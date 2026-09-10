"""
marley/ops/async_stream.py
==========================
BudgetedAsyncStreamer â€” Double-buffered async weight streaming for Wan2.1 DiT blocks.

Architecture
------------
All 30 DiT transformer blocks reside on CPU (pinned memory).
Two GPU slots (slot_A, slot_B) are pre-allocated on VRAM.
Two CUDA streams manage execution:
  - copy_stream  : H2D weight transfers (pinned CPU â†’ GPU slot)
  - compute_stream: block forward passes (GPU)

Bidirectional Event Ownership Protocol (Race Condition Fix)
-----------------------------------------------------------
The original single-event design allowed copy_stream to overwrite a slot
still being read by compute_stream. The fixed protocol uses BOTH directions:

  Iteration i (block[i] in slot_compute, block[i+1] being copied to slot_prefetch):

    1. Launch block[i].forward() on compute_stream
    2. Record compute_done_events[i] on compute_stream
    3. copy_stream.wait_event(compute_done_events[i])   â† NEW: copy waits to release old slot
    4. If i+2 < N: prefetch block[i+2] â†’ old slot_compute (now slot_prefetch after swap)
    5. compute_stream.wait_event(copy_done_events[i+1]) â† compute waits for next block ready
    6. swap: slot_compute, slot_prefetch = slot_prefetch, slot_compute
    7. bind block[i+1] to new slot_compute

Step 3 prevents copy_stream from writing block[i+2] into slot_compute while
compute_stream is still mid-forward on block[i] using that same slot.

References
----------
- docs/F3_CALIBRATION_CHANGES_02.md â€” full rationale for all design decisions
- docs/private/F3_F4_TECHNICAL_OPINION_01.md â€” consultant advisory letters
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# StreamMetrics â€” 7 mandatory F3 metrics
# ---------------------------------------------------------------------------

@dataclass
class StreamMetrics:
    """
    All seven metrics mandated by the F3 benchmark specification.

    Attributes
    ----------
    total_denoise_time_s : float
        Wall-clock time (seconds) for the full denoising loop (all steps Ã— all blocks).
    total_h2d_transfer_time_s : float
        Cumulative wall-clock time spent on H2D weight transfers (measured per-block).
    effective_overlap_pct : float
        Percentage of H2D transfer time that overlapped with compute (0â€“100).
    peak_vram_mb : float
        Peak VRAM consumed during the run (NVML physical, in MB). Populated externally.
    prefetch_latency_s : float
        Cumulative time compute_stream spent stalled waiting for a late prefetch.
    forced_sync_count : int
        Number of explicit synchronization calls outside the event protocol.
    time_per_frame_s : float
        Average wall-clock time per output frame (total_denoise_time_s / num_frames).
    condition : str
        "sync" or "async" â€” identifies which execution mode produced these metrics.
    step_times_s : List[float]
        Wall-clock time per denoising step (len == num_steps).
    nan_inf_detected : bool
        True if any NaN or Inf was found in block outputs during the run.
    """
    total_denoise_time_s: float = 0.0
    total_h2d_transfer_time_s: float = 0.0
    effective_overlap_pct: float = 0.0
    peak_vram_mb: float = 0.0          # populated by NVML sampler in benchmark script
    prefetch_latency_s: float = 0.0    # REAL accumulated GPU stall (seconds) since the
                                       # measurement-integrity fix (Enmienda 1)
    forced_sync_count: int = 0
    time_per_frame_s: float = 0.0
    condition: str = "unknown"
    step_times_s: List[float] = field(default_factory=list)
    nan_inf_detected: bool = False
    # Per-block GPU timeline (len == num_blocks). Populated for auditable overlap.
    # Empty lists when the execution mode does not measure the corresponding signal.
    per_block_copy_s: List[float] = field(default_factory=list)
    per_block_compute_s: List[float] = field(default_factory=list)
    per_block_stall_s: List[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Condition          : {self.condition}",
            f"  Total denoise time : {self.total_denoise_time_s*1000:.1f} ms",
            f"  H2D transfer total : {self.total_h2d_transfer_time_s*1000:.1f} ms",
            f"  Effective overlap  : {self.effective_overlap_pct:.1f}%",
            f"  Peak VRAM (NVML)   : {self.peak_vram_mb:.0f} MB",
            f"  Prefetch latency   : {self.prefetch_latency_s*1000:.2f} ms",
            f"  Forced syncs       : {self.forced_sync_count}",
            f"  Time per frame     : {self.time_per_frame_s*1000:.1f} ms",
            f"  NaN/Inf detected   : {self.nan_inf_detected}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BudgetedAsyncStreamer
# ---------------------------------------------------------------------------

class BudgetedAsyncStreamer:
    """
    Manages async streaming of Wan2.1 DiT blocks between CPU (pinned) and GPU (double-buffer).

    Design Decisions (see docs/F3_CALIBRATION_CHANGES_02.md)
    -------------------------------------------------------
    1. Does NOT use enable_sequential_cpu_offload() â€” Accelerate's AlignDevicesHook
       intercepts transfers on the default stream, bypassing copy_stream entirely.
    2. Bidirectional CUDA events eliminate the buffer ownership race condition.
    3. One GPU slot = one complete copy of all block parameters.
    4. param.data patching is used to route block.forward() through the GPU slot
       without modifying the block's original CPU state.

    Parameters
    ----------
    blocks : List[nn.Module]
        The 30 WanTransformerBlock instances, all residing on CPU.
    device : torch.device
        Target CUDA device.
    dtype : torch.dtype
        Weight dtype (fp16 in baseline F3).
    """

    def __init__(
        self,
        blocks: List[nn.Module],
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("BudgetedAsyncStreamer requires a CUDA device.")

        self.blocks = blocks
        self.device = device
        self.dtype = dtype
        self.num_blocks = len(blocks)

        # CUDA streams
        self.compute_stream = torch.cuda.Stream(device=device, priority=-1)   # high priority
        self.copy_stream = torch.cuda.Stream(device=device, priority=0)       # normal priority

        # Pre-allocated CUDA events (N events for copy_done, N for compute_done)
        # Created once and reused across steps to avoid per-step allocation overhead.
        self.copy_done_events: List[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=True) for _ in range(self.num_blocks)
        ]
        self.compute_done_events: List[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=True) for _ in range(self.num_blocks)
        ]

        # Host (CPU pinned) weight storage: host_weights[block_idx] = {name: cpu_tensor}
        self.host_weights: List[Dict[str, torch.Tensor]] = []

        # GPU double-buffer slots: gpu_slots[slot_idx] = {name: gpu_tensor}
        # Slot 0 = slot_A, Slot 1 = slot_B
        self.gpu_slots: List[Dict[str, torch.Tensor]] = [{}, {}]

        # Original param data references (for restoration after benchmarks)
        # original_data[block_idx] = {name: original_tensor}
        self._original_data: List[Dict[str, torch.Tensor]] = []

        # Setup
        self._prepare_host_weights()
        self._init_gpu_slots()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _prepare_host_weights(self) -> None:
        """
        Copy all block parameters to pinned CPU memory.

        Pinned (page-locked) memory enables DMA transfers from the CUDA driver
        without involving the OS page table, which is required for async H2D
        transfers that don't block the CPU.
        """
        for block_idx, block in enumerate(self.blocks):
            pinned: Dict[str, torch.Tensor] = {}
            original: Dict[str, torch.Tensor] = {}
            for name, param in block.named_parameters():
                cpu_tensor = param.detach().to("cpu", dtype=self.dtype)
                try:
                    pinned_tensor = cpu_tensor.pin_memory()
                except RuntimeError:
                    # Fallback for environments where pinning fails
                    pinned_tensor = cpu_tensor
                pinned[name] = pinned_tensor
                original[name] = param.data  # save original reference
            self.host_weights.append(pinned)
            self._original_data.append(original)

    def _init_gpu_slots(self) -> None:
        """
        Allocate two GPU buffer sets (slot_A = 0, slot_B = 1).

        Each slot contains empty GPU tensors with the same shapes as the first
        block's parameters. All blocks share the same architecture, so one
        shape template works for all 30 blocks.
        """
        template = self.host_weights[0]
        for slot_idx in range(2):
            slot: Dict[str, torch.Tensor] = {}
            for name, cpu_t in template.items():
                slot[name] = torch.empty(
                    cpu_t.shape,
                    dtype=self.dtype,
                    device=self.device,
                )
            self.gpu_slots[slot_idx] = slot

    # ------------------------------------------------------------------
    # Block transfer operations
    # ------------------------------------------------------------------

    def _load_block_sync(self, block_idx: int, slot: int) -> float:
        """
        Synchronously copy block[block_idx] parameters to gpu_slots[slot].

        Uses the default CUDA stream (no async). Returns elapsed seconds.
        This is the Condition A (Marley Sync) transfer primitive.
        """
        t0 = time.perf_counter()
        for name, cpu_t in self.host_weights[block_idx].items():
            self.gpu_slots[slot][name].copy_(cpu_t, non_blocking=False)
        torch.cuda.synchronize(self.device)
        return time.perf_counter() - t0

    def _prefetch_block_async(self, block_idx: int, slot: int) -> None:
        """
        Asynchronously copy block[block_idx] parameters to gpu_slots[slot] on copy_stream.

        Records copy_done_events[block_idx] after the transfer is enqueued.
        The event fires when copy_stream reaches that point (transfer complete).
        """
        with torch.cuda.stream(self.copy_stream):
            for name, cpu_t in self.host_weights[block_idx].items():
                self.gpu_slots[slot][name].copy_(cpu_t, non_blocking=True)
            self.copy_done_events[block_idx].record(self.copy_stream)

    def _bind_block_to_slot(self, block_idx: int, slot: int) -> None:
        """
        Redirect block[block_idx] parameter data pointers to gpu_slots[slot].

        After binding, block[block_idx].forward() uses the GPU slot tensors.
        Does NOT move any data â€” only changes param.data references.
        """
        slot_data = self.gpu_slots[slot]
        for name, param in self.blocks[block_idx].named_parameters():
            param.data = slot_data[name]

    def _restore_block_to_cpu(self, block_idx: int) -> None:
        """
        Restore block[block_idx] parameter data pointers to original CPU tensors.
        """
        for name, param in self.blocks[block_idx].named_parameters():
            param.data = self._original_data[block_idx][name]

    def restore_all_to_cpu(self) -> None:
        """Restore all blocks to their original CPU state. Call after benchmarking."""
        for i in range(self.num_blocks):
            self._restore_block_to_cpu(i)

    # ------------------------------------------------------------------
    # Shape probe
    # ------------------------------------------------------------------

    def probe_shapes(
        self,
        transformer: nn.Module,
        latent_shape: Tuple[int, ...] = (1, 16, 5, 60, 104),
        text_len: int = 226,
        text_dim: int = 4096,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]:
        """
        Capture exact WanTransformerBlock.forward() input shapes via a forward hook.

        Strategy:
          1. Register a forward hook on transformer.blocks[0].
          2. Run transformer.forward() with dummy CPU tensors.
          3. The hook captures the exact args blocks[0] receives (hidden_states,
             encoder_hidden_states, temb, rotary_emb) then raises StopIteration
             to abort before any subsequent blocks execute.
          4. Return CPU copies of all captured tensors.

        Why NOT use patch_embedding:
          transformer.patch_embedding is a Conv3D that collapses spatial HÃ—W into
          channels, returning (B, C, T) â€” NOT the (B, seq_len, dim) format the
          blocks receive. WanTransformer.forward transposes/reshapes internally.

        Why NOT use transformer.rope:
          transformer.rope() returns a tuple (cos, sin), not the tensor the blocks
          receive. WanTransformer.forward processes the rope output before passing
          it to each block.
        """
        dtype = self.dtype
        captured: dict = {}

        class _ShapesCaptured(Exception):
            pass

        def _capture_hook(module, args, kwargs, output):
            # args[0]=hidden_states, [1]=encoder_hidden_states, [2]=temb, [3]=rotary_emb
            captured["hidden_states"] = args[0].detach().cpu() if hasattr(args[0], "detach") else args[0]
            captured["encoder_hidden_states"] = args[1].detach().cpu() if hasattr(args[1], "detach") else args[1]
            captured["temb"] = args[2].detach().cpu() if hasattr(args[2], "detach") else args[2]
            # rotary_emb may be tensor or tuple depending on WanTransformer version
            re = args[3]
            if isinstance(re, torch.Tensor):
                captured["rotary_emb"] = re.detach().cpu()
            elif isinstance(re, (tuple, list)):
                captured["rotary_emb"] = tuple(
                    x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in re
                )
            else:
                captured["rotary_emb"] = re
            raise _ShapesCaptured()

        hook = transformer.blocks[0].register_forward_hook(_capture_hook, with_kwargs=True)
        try:
            dummy_latent = torch.randn(*latent_shape, dtype=dtype)
            dummy_text = torch.randn(1, text_len, text_dim, dtype=dtype)
            dummy_ts = torch.tensor([500.0])
            with torch.no_grad():
                transformer(
                    hidden_states=dummy_latent,
                    encoder_hidden_states=dummy_text,
                    timestep=dummy_ts,
                    return_dict=False,
                )
        except _ShapesCaptured:
            pass  # Expected exit path
        except Exception as e:
            print(f"[probe_shapes] Warning during forward probe: {type(e).__name__}: {e}")
        finally:
            hook.remove()

        if not captured:
            raise RuntimeError(
                "probe_shapes: failed to capture block inputs. "
                "Check that transformer.blocks[0] is accessible and the model is loaded."
            )

        return (
            captured["hidden_states"],
            captured["encoder_hidden_states"],
            captured["temb"],
            captured["rotary_emb"],
        )

    # ------------------------------------------------------------------
    # Condition A â€” Marley Synchronous Baseline
    # ------------------------------------------------------------------

    def execute_sync_loop(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        num_steps: int,
        num_frames: int = 17,
        nan_check: bool = True,
    ) -> StreamMetrics:
        """
        Condition A: Marley Synchronous baseline.

        For each step, iterates through all 30 DiT blocks sequentially:
          1. Synchronous H2D transfer of block weights to GPU slot 0.
          2. Bind block to slot 0.
          3. Forward pass on compute_stream.
          4. Synchronize. Restore block to CPU.

        This establishes the A in the A/B comparison. Any speedup in execute_async_loop()
        is attributable to async prefetching, not to any other architectural difference.

        Parameters
        ----------
        hidden_states : pre-moved to device before calling.
        num_steps : number of denoising steps to simulate (default 30 for F3-B).
        num_frames : number of output frames (17 for 480p baseline).
        nan_check : if True, checks for NaN/Inf in block outputs at each step.
        """
        m = StreamMetrics(condition="sync")
        device = self.device

        h = hidden_states.clone().to(device)
        enc = encoder_hidden_states.to(device)
        t = temb.to(device)
        # rotary_emb may be a Tensor or a tuple of tensors (cos, sin)
        if isinstance(rotary_emb, torch.Tensor):
            r = rotary_emb.to(device)
        else:
            r = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in rotary_emb)

        total_h2d_time = 0.0
        step_times = []
        # Per-block copy timeline (host-measured). Compute/stall timelines are not
        # measured in the synchronous path (overlap is 0 by definition), so those
        # lists stay zero-filled for schema consistency with the async path.
        per_block_copy_s: List[float] = [0.0] * self.num_blocks

        wall_start = time.perf_counter()

        with torch.no_grad():
            for step in range(num_steps):
                step_start = time.perf_counter()

                for block_idx in range(self.num_blocks):
                    # --- Synchronous H2D transfer ---
                    transfer_t = self._load_block_sync(block_idx, slot=0)
                    total_h2d_time += transfer_t
                    per_block_copy_s[block_idx] += transfer_t

                    # --- Bind and forward ---
                    self._bind_block_to_slot(block_idx, slot=0)
                    with torch.cuda.stream(self.compute_stream):
                        h = self.blocks[block_idx](h, enc, t, r)
                        m.forced_sync_count += 1  # one sync per block in sync mode

                    self.compute_stream.synchronize()

                    # --- NaN/Inf check ---
                    if nan_check and (torch.isnan(h).any() or torch.isinf(h).any()):
                        m.nan_inf_detected = True

                    # --- Restore block to CPU ---
                    self._restore_block_to_cpu(block_idx)

                step_time = time.perf_counter() - step_start
                step_times.append(step_time)

        wall_end = time.perf_counter()

        # --- Populate metrics ---
        m.total_denoise_time_s = wall_end - wall_start
        m.total_h2d_transfer_time_s = total_h2d_time
        m.effective_overlap_pct = 0.0  # sync: no overlap by definition
        m.prefetch_latency_s = 0.0
        m.step_times_s = step_times
        m.time_per_frame_s = m.total_denoise_time_s / max(num_frames, 1)
        m.per_block_copy_s = per_block_copy_s
        m.per_block_compute_s = [0.0] * self.num_blocks
        m.per_block_stall_s = [0.0] * self.num_blocks
        # peak_vram_mb populated by NVML sampler in benchmark script

        return m

    # ------------------------------------------------------------------
    # Condition B â€” Marley Asynchronous Double-Buffered
    # ------------------------------------------------------------------

    def execute_async_loop(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        num_steps: int,
        num_frames: int = 17,
        nan_check: bool = True,
        sync_baseline_h2d_s: Optional[float] = None,
    ) -> StreamMetrics:
        """
        Condition B: Marley Asynchronous double-buffered scheduler.

        Implements the bidirectional event ownership protocol:
          - copy_stream handles H2D prefetch of block[i+1] while compute_stream
            processes block[i].
          - compute_done_events[i]: compute_stream â†’ copy_stream
            (copy_stream must wait before reusing the old compute slot).
          - copy_done_events[i+1]: copy_stream â†’ compute_stream
            (compute_stream must wait before reading the newly loaded slot).

        This eliminates the race condition present in single-direction event designs.

        Parameters and return value are identical to execute_sync_loop().
        """
        m = StreamMetrics(condition="async")
        device = self.device

        h = hidden_states.clone().to(device)
        enc = encoder_hidden_states.to(device)
        t = temb.to(device)
        # rotary_emb may be a Tensor or a tuple of tensors (cos, sin)
        if isinstance(rotary_emb, torch.Tensor):
            r = rotary_emb.to(device)
        else:
            r = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in rotary_emb)

        total_h2d_time = 0.0
        total_prefetch_latency = 0.0
        step_times = []

        # Measurement-integrity accumulators (Enmienda 1 â€” real overlap/stall).
        # Per-block GPU timelines are accumulated across ALL steps so the reported
        # overlap is empirical and auditable, never a formula artifact.
        measured_async_copy_s = 0.0    # Î£ GPU-measured H2D copies of prefetched blocks (>=1)
        per_block_copy_s: List[float] = [0.0] * self.num_blocks
        per_block_compute_s: List[float] = [0.0] * self.num_blocks
        per_block_stall_s: List[float] = [0.0] * self.num_blocks

        # CUDA events for timing H2D overlap measurement
        ev_copy_start: List[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=True) for _ in range(self.num_blocks)
        ]
        ev_compute_start: List[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=True) for _ in range(self.num_blocks)
        ]
        ev_compute_end: List[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=True) for _ in range(self.num_blocks)
        ]

        wall_start = time.perf_counter()

        with torch.no_grad():
            for step in range(num_steps):
                step_start = time.perf_counter()

                # ---------------------------------------------------------------
                # INITIALIZATION: synchronously load block[0] into slot_compute,
                # then start async prefetch of block[1] into slot_prefetch.
                # ---------------------------------------------------------------
                slot_compute = 0
                slot_prefetch = 1

                # Block 0: sync load into slot_compute
                h2d_t = self._load_block_sync(0, slot=slot_compute)
                total_h2d_time += h2d_t
                per_block_copy_s[0] += h2d_t
                self._bind_block_to_slot(0, slot=slot_compute)

                # Block 1: async prefetch into slot_prefetch
                if self.num_blocks > 1:
                    with torch.cuda.stream(self.copy_stream):
                        ev_copy_start[1].record(self.copy_stream)
                    self._prefetch_block_async(1, slot=slot_prefetch)

                # Per-step references for the real stall measurement (Enmienda 1).
                # Indexed by block id (1..N-1); measured after the step synchronize.
                stall_start_evs: List[Optional[torch.cuda.Event]] = [None] * self.num_blocks

                # ---------------------------------------------------------------
                # MAIN LOOP
                # ---------------------------------------------------------------
                for i in range(self.num_blocks):

                    # --- STEP 1: Launch block[i].forward() on compute_stream ---
                    with torch.cuda.stream(self.compute_stream):
                        ev_compute_start[i].record(self.compute_stream)
                        h = self.blocks[i](h, enc, t, r)
                        self.compute_done_events[i].record(self.compute_stream)
                        ev_compute_end[i].record(self.compute_stream)

                    # --- STEP 2 (RACE CONDITION FIX): copy_stream waits for compute_done[i]
                    #     before it can overwrite slot_compute (which becomes slot_prefetch
                    #     after the slot swap below). ---
                    self.copy_stream.wait_event(self.compute_done_events[i])

                    # --- STEP 3: Prefetch block[i+2] into the now-released slot_compute ---
                    if i + 2 < self.num_blocks:
                        with torch.cuda.stream(self.copy_stream):
                            ev_copy_start[i + 2].record(self.copy_stream)
                        self._prefetch_block_async(i + 2, slot=slot_compute)

                    # --- STEP 4: compute_stream waits for copy_done[i+1] before
                    #     it can execute block[i+1] (which is in slot_prefetch). ---
                    if i + 1 < self.num_blocks:
                        # Measure stall (prefetch latency): keep the compute-side start
                        # event referenced so its elapsed vs. copy_done[i+1] is computed
                        # right after this step's end-of-step synchronize (the event
                        # arrays are reused across steps and must be measured in-step).
                        stall_ev_start = torch.cuda.Event(enable_timing=True)
                        with torch.cuda.stream(self.compute_stream):
                            stall_ev_start.record(self.compute_stream)
                        stall_start_evs[i + 1] = stall_ev_start
                        self.compute_stream.wait_event(self.copy_done_events[i + 1])

                    # --- STEP 5: Swap slots ---
                    slot_compute, slot_prefetch = slot_prefetch, slot_compute

                    # --- STEP 6: Bind next block to new slot_compute ---
                    if i + 1 < self.num_blocks:
                        self._bind_block_to_slot(i + 1, slot=slot_compute)

                    # --- NaN/Inf check ---
                    if nan_check:
                        # Non-blocking check: just flag, synchronize at step end
                        pass  # checked below after synchronize

                # Synchronize all streams at end of step (single sync per step)
                torch.cuda.synchronize(device)

                # --- Accumulate THIS step's GPU measurements (Enmienda 1) ---
                # Events are reused across steps, so elapsed times must be computed
                # now â€” before the next step re-records them.
                for j in range(1, self.num_blocks):
                    # Real prefetch stall: how much compute actually waited for block j.
                    stall_ev = stall_start_evs[j]
                    if stall_ev is not None:
                        stall_ms = stall_ev.elapsed_time(self.copy_done_events[j])
                        if stall_ms > 0.0:  # copy finished first â†’ negative â†’ no stall
                            total_prefetch_latency += stall_ms / 1000.0
                            per_block_stall_s[j] += stall_ms / 1000.0
                    # GPU-measured H2D copy duration of prefetched block j.
                    copy_ms = ev_copy_start[j].elapsed_time(self.copy_done_events[j])
                    if copy_ms > 0.0:
                        measured_async_copy_s += copy_ms / 1000.0
                        per_block_copy_s[j] += copy_ms / 1000.0
                # GPU-measured compute duration of each block.
                for i in range(self.num_blocks):
                    compute_ms = ev_compute_start[i].elapsed_time(ev_compute_end[i])
                    if compute_ms > 0.0:
                        per_block_compute_s[i] += compute_ms / 1000.0

                if nan_check and (torch.isnan(h).any() or torch.isinf(h).any()):
                    m.nan_inf_detected = True

                step_time = time.perf_counter() - step_start
                step_times.append(step_time)

                # Restore all blocks to CPU after this step
                for idx in range(self.num_blocks):
                    self._restore_block_to_cpu(idx)

        wall_end = time.perf_counter()

        # ---------------------------------------------------------------
        # Compute overlap and H2D metrics â€” EMPIRICAL (Enmienda 1)
        # ---------------------------------------------------------------
        # Total H2D work is now measured, not assumed:
        #   - blocks 1..N-1 per step: GPU-measured async copy duration (ev_copy_start
        #     â†’ copy_done). This is the transfer work that CAN overlap with compute.
        #   - block 0 per step: synchronous host-timed load (inherently serial at the
        #     top of each step).
        measured_copy_s = measured_async_copy_s + per_block_copy_s[0]
        if measured_copy_s > 0.0:
            total_h2d_time = measured_copy_s
        elif sync_baseline_h2d_s is not None and sync_baseline_h2d_s > 0:
            # Fallback only if GPU event timing was unavailable.
            total_h2d_time = sync_baseline_h2d_s
        else:
            single_block_bytes = sum(p.numel() * p.element_size() for p in self.blocks[0].parameters())
            total_transfer_bytes = num_steps * self.num_blocks * single_block_bytes
            total_h2d_time = total_transfer_bytes / (7.5 * 1024**3)

        # Effective overlap = fraction of prefetchable H2D work (blocks >= 1) that
        # did NOT force the compute stream to stall.
        if measured_async_copy_s > 0.0:
            stall_ratio = total_prefetch_latency / measured_async_copy_s
            hidden_ratio = max(0.0, min(1.0, 1.0 - stall_ratio))
            m.effective_overlap_pct = hidden_ratio * 100.0
        else:
            m.effective_overlap_pct = 0.0

        # ---------------------------------------------------------------
        # Populate metrics
        # ---------------------------------------------------------------
        m.total_denoise_time_s = wall_end - wall_start
        m.total_h2d_transfer_time_s = total_h2d_time
        m.prefetch_latency_s = total_prefetch_latency
        m.step_times_s = step_times
        m.time_per_frame_s = m.total_denoise_time_s / max(num_frames, 1)
        m.forced_sync_count = num_steps  # one synchronize() per step at step end
        m.per_block_copy_s = per_block_copy_s
        m.per_block_compute_s = per_block_compute_s
        m.per_block_stall_s = per_block_stall_s

        return m

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        """
        Free GPU slot tensors and clear internal state.
        Always call restore_all_to_cpu() before release() if blocks will be reused.
        """
        for slot in self.gpu_slots:
            slot.clear()
        self.host_weights.clear()
        torch.cuda.empty_cache()


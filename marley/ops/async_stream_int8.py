"""
marley/ops/async_stream_int8.py
================================
INT8BudgetedStreamer — Double-buffered async weight streaming with INT8 linear projections.

Isolation experiment (Phase F3 + INT8). Reuses the proven BudgetedAsyncStreamer scheduler
(double-buffer + bidirectional CUDA event ownership) UNCHANGED, and swaps only the four
low-level transfer primitives so that the H2D payload carried across PCIe for each DiT
block's linear-projection weights is INT8 (≈ half the FP16 bytes) instead of FP16.

Compute remains FP16
--------------------
Per F1.7, the DiT linear projections are ~96.4% of per-block parameters. We quantize those
2D weight matrices per output-channel with a symmetric INT8 scale:

    w_int8 = round(w / scale_row),   scale_row = max|w_row| / 127

Host stores the INT8 payload + a small per-row FP16 scale. On the GPU, each FP16 compute
slot first receives the INT8 integers (a ~50% byte reduction across PCIe) and is then
dequantized in place (multiply each row by its scale) so that block.forward() executes in
exact FP16 precision. Non-linear parameters (norms, biases — 1-D) remain FP16 end to end.

This isolates the single independent variable mandated by the F3+INT8 plan:
    transferred byte volume per block
while keeping compute precision and numerical path directly comparable to the frozen
F3 FP16 baseline.

Race-condition protection is inherited verbatim from marley/ops/async_stream.py:
  compute_stream → copy_stream  (copy waits for compute_done before overwriting a slot)
  copy_stream → compute_stream  (compute waits for copy_done before reading a slot)
The dequantize step is folded into the transfer primitive, so copy_done_events fire only
AFTER the slot holds final dequantized FP16 weights.

References
----------
- docs/F3_INT8_EXPERIMENT_PLAN_01.md  — isolation experiment design
- docs/F3_F4_TECHNICAL_OPINION_01.md  — INT8-as-production-baseline directive
- docs/F3_CALIBRATION_CHANGES_02.md   — measurement-integrity (Enmiendas 1-3)
- marley/ops/async_stream.py          — BudgetedAsyncStreamer / StreamMetrics
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from marley.ops.async_stream import BudgetedAsyncStreamer, StreamMetrics


class INT8BudgetedStreamer(BudgetedAsyncStreamer):
    """
    BudgetedAsyncStreamer variant whose H2D payload for linear-projection weights is INT8.

    Compute runs in FP16 after an on-device per-row dequantize. All scheduler logic
    (streams, events, double-buffer, bind/restore) is inherited unchanged; only the
    transfer primitives (_load_block_sync / _prefetch_block_async) and the two setup
    methods that build host/GPU storage are overridden.

    Parameters
    ----------
    blocks : List[nn.Module]
        The 30 WanTransformerBlock instances, all residing on CPU (FP16).
    device : torch.device
        Target CUDA device.
    """

    def __init__(
        self,
        blocks: List[nn.Module],
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("INT8BudgetedStreamer requires a CUDA device.")
        self.blocks = blocks
        self.device = device
        self.dtype = dtype
        self.num_blocks = len(blocks)

        # Per-block INT8 scale map: quantized_layer_names[block][name] -> is 2-D quantized
        self._q2d_names: List[List[str]] = [[] for _ in range(self.num_blocks)]
        # Host per-row FP16 scales (pinned), keyed (block, name).
        self.host_scale: List[Dict[str, torch.Tensor]] = []
        # GPU per-slot scale scratch (dequant multipliers), keyed (slot, name).
        self._scale_gpu: List[Dict[str, torch.Tensor]] = [{}, {}]
        # Blocks per-slot template shape (all blocks share architecture).
        self._template_shape: Dict[str, torch.Size] = {}

        # CUDA streams (same priorities as base).
        self.compute_stream = torch.cuda.Stream(device=device, priority=-1)
        self.copy_stream = torch.cuda.Stream(device=device, priority=0)

        # Reused events sized by block count.
        self.copy_done_events: List[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=True) for _ in range(self.num_blocks)
        ]
        self.compute_done_events: List[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=True) for _ in range(self.num_blocks)
        ]

        # Host weight storage + original CPU references.
        self.host_weights: List[Dict[str, torch.Tensor]] = []
        self._original_data: List[Dict[str, torch.Tensor]] = []
        # GPU double-buffer compute slots (FP16), same layout as base.
        self.gpu_slots: List[Dict[str, torch.Tensor]] = [{}, {}]
        self._prepare_host_weights()
        self._init_gpu_slots()

    # ------------------------------------------------------------------
    # Setup overrides
    # ------------------------------------------------------------------

    def _prepare_host_weights(self) -> None:
        """
        Build pinned host storage:
          - 2-D floating point weights (linear projections) -> pinned INT8 + pinned FP16 scale
          - everything else (norms / biases / 1-D)          -> pinned FP16
        Also records original CPU param.data for later restoration.
        """
        for block_idx, block in enumerate(self.blocks):
            pinned: Dict[str, torch.Tensor] = {}
            original: Dict[str, torch.Tensor] = {}
            scales: Dict[str, torch.Tensor] = {}
            for name, param in block.named_parameters():
                cpu_t = param.detach().to("cpu", dtype=self.dtype)
                original[name] = param.data
                if param.ndim == 2 and param.numel() > 0:
                    # Per-output-row symmetric INT8 quantization.
                    wf = cpu_t.float()
                    amax = wf.abs().amax(dim=1).clamp_min(1e-12)  # [out]
                    scale = amax / 127.0                          # [out]
                    wq = torch.clamp(torch.round(wf / scale.view(-1, 1)), -127, 127)
                    wq_int8 = wq.to(torch.int8).contiguous()
                    pinned_t = wq_int8.pin_memory()
                    scale_t = scale.half().contiguous().pin_memory()
                    pinned[name] = pinned_t
                    scales[name] = scale_t
                    self._q2d_names[block_idx].append(name)
                    self._template_shape[name] = wq_int8.shape
                else:
                    pinned_t = cpu_t.pin_memory()
                    pinned[name] = pinned_t
                    self._template_shape[name] = cpu_t.shape
            self.host_weights.append(pinned)
            self.host_scale.append(scales)
            self._original_data.append(original)

    def _init_gpu_slots(self) -> None:
        """
        Allocate two GPU buffer sets (slot_A = 0, slot_B = 1).

        Each slot holds FP16 tensors (compute dtype) with the exact block parameter shapes.
        Quantized layers are FP16 tensors into which the INT8 integers are copied (dtype
        conversion) and then dequantized in place. Per-slot scale scratch tensors are sized
        from the host scale of block 0 (all blocks share architecture).
        """
        # Host weights for block 0 encode per-layer shapes; allocate FP16 compute tensors.
        for slot_idx in range(2):
            slot: Dict[str, torch.Tensor] = {}
            for name, _shape in self._template_shape.items():
                slot[name] = torch.empty(
                    _shape, dtype=self.dtype, device=self.device
                )
            self.gpu_slots[slot_idx] = slot

        # Per-slot scale scratch, keyed by layer name (shape = per-row scale length).
        for name in self.host_scale[0]:
            out_len = self.host_scale[0][name].shape[0]
            for slot_idx in range(2):
                self._scale_gpu[slot_idx][name] = torch.empty(
                    (out_len,), dtype=self.dtype, device=self.device
                )

    def _apply_dequant(self, block_idx: int, slot_idx: int) -> None:
        """
        Dequantize INT8 integers (already copied into the FP16 compute slot) in place by
        multiplying each output row by its stored per-row FP16 scale. Runs on the caller's
        current CUDA stream; must be invoked after the INT8 payload has been copied.
        """
        slot = self.gpu_slots[slot_idx]
        scale_dst = self._scale_gpu[slot_idx]
        for name in self._q2d_names[block_idx]:
            # Pull the per-row scale to GPU, then multiply the FP16 slot row-wise.
            scale_dst[name].copy_(self.host_scale[block_idx][name], non_blocking=True)
            slot[name].mul_(scale_dst[name].view(-1, 1))

    # ------------------------------------------------------------------
    # Transfer primitive overrides (INT8 payload + on-device dequantize)
    # ------------------------------------------------------------------

    def _load_block_sync(self, block_idx: int, slot: int) -> float:
        """
        Synchronously move block[block_idx] to gpu_slots[slot] on the default stream:
          copy INT8 payload (linear proj) + FP16 (others), then dequantize in place.
        Returns elapsed seconds (host-timed; used for the sync baseline H2D metric).
        """
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for name, cpu_t in self.host_weights[block_idx].items():
            self.gpu_slots[slot][name].copy_(cpu_t, non_blocking=False)
        self._apply_dequant(block_idx, slot)
        torch.cuda.synchronize(self.device)
        t1.record()
        t1.synchronize()
        return t0.elapsed_time(t1) / 1000.0

    def _prefetch_block_async(self, block_idx: int, slot: int) -> None:
        """
        Asynchronously move block[block_idx] to gpu_slots[slot] on copy_stream:
          INT8/FP16 copy + on-device dequantize, THEN record copy_done_events[block_idx].
        Recording after dequant guarantees the slot is compute-ready when compute_stream
        observes the event.
        """
        with torch.cuda.stream(self.copy_stream):
            for name, cpu_t in self.host_weights[block_idx].items():
                self.gpu_slots[slot][name].copy_(cpu_t, non_blocking=True)
            self._apply_dequant(block_idx, slot)
            self.copy_done_events[block_idx].record(self.copy_stream)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def host_payload_bytes(self, block_idx: int = 0) -> int:
        """Bytes actually moved across PCIe for one block (INT8 weights + FP16 others)."""
        total = 0
        for name, cpu_t in self.host_weights[block_idx].items():
            if name in self._q2d_names[block_idx]:
                total += cpu_t.numel() * 1 + self.host_scale[block_idx][name].numel() * 2
            else:
                total += cpu_t.numel() * cpu_t.element_size()
        return total

    def fp16_equivalent_bytes(self, block_idx: int = 0) -> int:
        """Bytes the frozen FP16 baseline would move for the same block."""
        return sum(
            p.numel() * 2 for p in self.blocks[block_idx].parameters()
        )

    def restore_all_to_cpu(self) -> None:
        for i in range(self.num_blocks):
            for name, param in self.blocks[i].named_parameters():
                param.data = self._original_data[i][name]

    def release(self) -> None:
        for slot in self.gpu_slots:
            slot.clear()
        for slot in self._scale_gpu:
            slot.clear()
        self.host_weights.clear()
        self.host_scale.clear()
        torch.cuda.empty_cache()

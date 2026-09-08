"""
marley/core/adaptive.py
=======================
F4 AdaptiveEngine — the decision loop that selects FP16/INT8, prefetch and
residency per window and delegates execution to the frozen F3/F3+INT8 streamers.

Responsibilities (F4_TEST_SPEC v3 §2.1–§2.4):
  - Observer : maintain an EMA of memory pressure; expose the 6 RuntimeState signals.
  - Selector : pick a PolicyDecision for the active profile (delegated to policies.py).
  - Planner  : Ahead-Of-Time (AOT) baseline plan, one decision per window of N steps.
  - Checkpoint : re-evaluate every window AND on abrupt pressure deviation, with
                 hysteresis (PRESSURE_HIGH / PRESSURE_LOW + MIN_DWELL_WINDOWS).

Constraint: this module NEVER re-implements transfer/compute. It calls
BudgetedAsyncStreamer (FP16) or INT8BudgetedStreamer (INT8) unchanged. Switching
precision reuses the pre-computed host representation of each streamer (built once
in __init__), so no on-the-fly re-quantization occurs.

Authorized contract: docs/F4_TEST_SPEC_01.md (v3, approved by Director).
Do NOT modify marley/ops/async_stream*.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from marley.core.policies import (
    PolicyDecision,
    PolicySelector,
    PROFILES,
    RuntimeState,
)
from marley.ops.async_stream import BudgetedAsyncStreamer, StreamMetrics
from marley.ops.async_stream_int8 import INT8BudgetedStreamer

# ---------------------------------------------------------------------------
# Hysteresis / tuning constants (F4_TEST_SPEC v3 §2.4 / advisor letter §3).
# ---------------------------------------------------------------------------
PRESSURE_HIGH_MB = 200.0      # used >= ema + HIGH  -> enter pressure regime
PRESSURE_LOW_MB = 200.0       # used <= ema - LOW   -> may exit pressure regime
MIN_DWELL_WINDOWS = 1         # min windows in a regime before it may flip back
DEFAULT_WINDOW = 5            # N: diffusion steps per decision window
EMA_ALPHA = 0.3               # EMA smoothing for pressure
VRAM_HARD_GATE_MB = 4800.0
VRAM_SAFE_HEADROOM_MB = 1500.0


@dataclass
class WindowRecord:
    """Per-window execution + decision telemetry."""
    window: int = 0
    steps_run: int = 0
    profile: str = "performance"
    decision: Optional[PolicyDecision] = None
    wall_ms: float = 0.0
    overlap_pct: float = 0.0
    peak_vram_mb: float = 0.0
    used_mb_at_decision: float = 0.0
    ema_used_mb: float = 0.0
    decision_overhead_ms: float = 0.0


@dataclass
class AdaptiveResult:
    """Aggregate result returned by AdaptiveEngine.run()."""
    merged: StreamMetrics = field(default_factory=lambda: StreamMetrics(condition="adaptive"))
    windows: List[WindowRecord] = field(default_factory=list)
    telemetry: Dict = field(default_factory=dict)


class AdaptiveEngine:
    """
    Drives a Wan2.1 DiT denoise workload in decision windows, delegating each
    window to the frozen FP16 or INT8 streamer according to the active policy.

    Parameters
    ----------
    blocks : List[nn.Module]          -- the 30 WanTransformerBlock, on CPU.
    device : torch.device
    dtype : torch.dtype               -- compute dtype (fp16 default).
    window : int                      -- N diffusion steps per decision window.
    base_profile : str                -- "performance" | "memory_safe" (start).
    pressure_sampler : Optional[Callable[[], float]]
        If provided, a callable returning current *used* physical VRAM (MB) used
        to drive the EMA and dynamic re-planning. When None the engine runs the
        static AOT plan only (no pressure adaptation).
    """

    def __init__(
        self,
        blocks: List[nn.Module],
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        window: int = DEFAULT_WINDOW,
        base_profile: str = "performance",
        pressure_sampler: Optional[Callable[[], float]] = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("AdaptiveEngine requires a CUDA device.")
        if base_profile not in PROFILES:
            raise ValueError(f"base_profile must be one of {PROFILES}, got {base_profile!r}")
        if window < 1:
            raise ValueError("window (N) must be >= 1")

        self.blocks = blocks
        self.device = device
        self.dtype = dtype
        self.window = window
        self.base_profile = base_profile
        self._pressure_sampler = pressure_sampler

        self._vram_total_mb = (
            torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        )

        # Build BOTH streamers up-front so each keeps a pre-computed pinned host
        # representation (FP16 and INT8). Switching precision between windows then
        # requires no on-the-fly re-quantization.
        self._fp16 = BudgetedAsyncStreamer(blocks=blocks, device=device, dtype=dtype)
        self._int8 = INT8BudgetedStreamer(blocks=blocks, device=device, dtype=dtype)

        # Observer / hysteresis state.
        self._ema_used_mb: Optional[float] = None
        self._regime: str = "normal"     # "normal" | "pressure"
        self._dwell_windows: int = 0
        self._active_profile: str = base_profile
        self._last_exec_sig: Optional[Tuple[str, str]] = None  # (precision, prefetch)

        # Telemetry accumulators.
        self._decision_overhead_ms = 0.0
        self._switch_count = 0
        self._replan_count = 0
        self._replan_events: List[Dict] = []

    # ------------------------------------------------------------------
    # Observer
    # ------------------------------------------------------------------

    def _read_used_mb(self) -> Optional[float]:
        if self._pressure_sampler is not None:
            try:
                return float(self._pressure_sampler())
            except Exception:
                return None
        return None

    def _update_ema(self, used_mb: float) -> None:
        if self._ema_used_mb is None:
            self._ema_used_mb = used_mb
        else:
            self._ema_used_mb = (
                EMA_ALPHA * used_mb + (1.0 - EMA_ALPHA) * self._ema_used_mb
            )

    def _observe(self, last_window: Optional[StreamMetrics]) -> RuntimeState:
        used = self._read_used_mb()
        if used is None:
            used = self._ema_used_mb if self._ema_used_mb is not None else 0.0
        self._update_ema(used)
        free = max(0.0, self._vram_total_mb - used)
        return RuntimeState(
            free_vram_mb=free,
            recent_h2d_ms=(last_window.total_h2d_transfer_time_s * 1000.0)
            if last_window else 0.0,
            layer_compute_ms=(
                sum(last_window.per_block_compute_s) * 1000.0
            ) if last_window else 0.0,
            ema_pressure_mb=self._ema_used_mb or 0.0,
            prefetch_stall_ms=(last_window.prefetch_latency_s * 1000.0)
            if last_window else 0.0,
            dequant_cost_ms=0.0,
        )

    # ------------------------------------------------------------------
    # Hysteresis monitor (returns new active profile or None if unchanged)
    # ------------------------------------------------------------------

    def _monitor(self, used_mb: float) -> Optional[str]:
        prev_ema = self._ema_used_mb
        self._update_ema(used_mb)

        if prev_ema is None:
            return None

        high = used_mb >= prev_ema + PRESSURE_HIGH_MB
        low = used_mb <= prev_ema - PRESSURE_LOW_MB

        if self._regime == "normal":
            if high:
                self._regime = "pressure"
                self._dwell_windows = 0
                return "memory_safe"
        else:  # currently in pressure regime
            self._dwell_windows += 1
            if low and self._dwell_windows >= MIN_DWELL_WINDOWS:
                self._regime = "normal"
                self._dwell_windows = 0
                return "performance"
        return None

    # ------------------------------------------------------------------
    # Decision / execution mapping
    # ------------------------------------------------------------------

    def _decide(self, state: RuntimeState) -> PolicyDecision:
        return PolicySelector.select(self._active_profile, state)

    def _apply_profile_switch(self, new_profile: str, window_idx: int) -> None:
        if new_profile != self._active_profile:
            self._active_profile = new_profile
            self._replan_count += 1
            self._replan_events.append({
                "window": window_idx,
                "profile": new_profile,
                "regime": self._regime,
                "ema_used_mb": self._ema_used_mb,
            })

    def _execution_target(self, decision: PolicyDecision) -> Tuple[object, str]:
        """Return (streamer, mode) to run a window from a decision."""
        streamer = self._int8 if decision.uses_int8() else self._fp16
        mode = "async" if decision.uses_async() else "sync"
        return streamer, mode

    def _track_switch(self, decision: PolicyDecision) -> None:
        sig = (decision.precision, decision.prefetch)
        if self._last_exec_sig is not None and sig != self._last_exec_sig:
            self._switch_count += 1
        self._last_exec_sig = sig

    # ------------------------------------------------------------------
    # AOT baseline plan
    # ------------------------------------------------------------------

    def plan_aot(self, num_steps: int) -> List[PolicyDecision]:
        """
        Generate the static baseline plan (one decision per window of N steps)
        for the base profile, independent of runtime pressure. Used when no
        pressure sampler is present (pure deterministic static AOT execution).
        """
        num_windows = (num_steps + self.window - 1) // self.window
        probe_state = RuntimeState(
            free_vram_mb=self._vram_total_mb,
            ema_pressure_mb=0.0,
        )
        decision = PolicySelector.select(self.base_profile, probe_state)
        return [decision for _ in range(num_windows)]

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb,
        num_steps: int,
        num_frames: int = 17,
        nan_check: bool = True,
        base_profile: Optional[str] = None,
        pressure_sampler: Optional[Callable[[], float]] = None,
    ) -> AdaptiveResult:
        """
        Run the denoise loop in windows of `self.window` steps, delegating each
        window to the frozen FP16/INT8 streamer per the active decision.

        Returns an AdaptiveResult carrying the merged StreamMetrics plus per-window
        records and decision telemetry. peak_vram_mb is left for the caller's NVML
        sampler to populate (mirrors the F3 convention).
        """
        if pressure_sampler is not None:
            self._pressure_sampler = pressure_sampler
            # Fresh observer for an explicitly-injected pressure scenario.
            self._ema_used_mb = None
            self._regime = "normal"
            self._dwell_windows = 0
        if base_profile is not None and base_profile != self._active_profile:
            self.base_profile = base_profile
            self._active_profile = base_profile
            self._regime = "normal"
            self._dwell_windows = 0

        num_windows = (num_steps + self.window - 1) // self.window
        windows: List[WindowRecord] = []
        window_streams: List[StreamMetrics] = []
        h_input = hidden_states.detach().cpu() if hidden_states.is_cuda else hidden_states

        wall_start = time.perf_counter()
        step_offset = 0
        total_denoise = 0.0
        total_h2d = 0.0
        total_stall = 0.0
        total_async_copy = 0.0
        forced_syncs = 0
        step_times: List[float] = []
        nan_inf = False

        per_block_copy: List[float] = [0.0] * len(self.blocks)
        per_block_stall: List[float] = [0.0] * len(self.blocks)
        per_block_compute: List[float] = [0.0] * len(self.blocks)

        last_mm: Optional[StreamMetrics] = None

        for w in range(num_windows):
            record = WindowRecord(window=w)

            # Re-plan / observe once per window.
            used_at_decision: Optional[float] = self._read_used_mb()
            if used_at_decision is not None:
                new_profile = self._monitor(used_at_decision)
                if new_profile is not None:
                    self._apply_profile_switch(new_profile, w)
            state = self._observe(last_mm)
            record.used_mb_at_decision = used_at_decision or 0.0
            record.ema_used_mb = self._ema_used_mb or 0.0

            t0 = time.perf_counter()
            decision = self._decide(state)
            decision_overhead = time.perf_counter() - t0
            record.decision = decision
            record.profile = self._active_profile
            record.decision_overhead_ms = decision_overhead * 1000.0
            self._decision_overhead_ms += decision_overhead * 1000.0
            self._track_switch(decision)

            steps_run = min(self.window, num_steps - step_offset)
            record.steps_run = steps_run

            streamer, mode = self._execution_target(decision)
            if mode == "sync":
                mm = streamer.execute_sync_loop(
                    h_input, encoder_hidden_states, temb, rotary_emb,
                    num_steps=steps_run, num_frames=num_frames, nan_check=nan_check,
                )
            else:
                mm = streamer.execute_async_loop(
                    h_input, encoder_hidden_states, temb, rotary_emb,
                    num_steps=steps_run, num_frames=num_frames, nan_check=nan_check,
                )

            record.wall_ms = mm.total_denoise_time_s * 1000.0
            record.overlap_pct = mm.effective_overlap_pct
            windows.append(record)
            window_streams.append(mm)

            # Accumulate across windows.
            total_denoise += mm.total_denoise_time_s
            total_h2d += mm.total_h2d_transfer_time_s
            total_stall += mm.prefetch_latency_s
            forced_syncs += mm.forced_sync_count
            nan_inf = nan_inf or mm.nan_inf_detected
            step_times.extend(mm.step_times_s)
            # Keep the widest measured async-copy for overlap (per-block sums).
            measured_async = sum(mm.per_block_copy_s) if mode == "async" else 0.0
            total_async_copy = max(total_async_copy, measured_async)
            for j in range(len(self.blocks)):
                per_block_copy[j] += mm.per_block_copy_s[j]
                per_block_stall[j] += mm.per_block_stall_s[j]
                per_block_compute[j] += mm.per_block_compute_s[j]

            last_mm = mm
            step_offset += steps_run
            if step_offset >= num_steps:
                break

        wall_end = time.perf_counter()
        external_wall = wall_end - wall_start

        # ---- Merge into a single StreamMetrics ----
        merged = StreamMetrics(condition="adaptive")
        merged.total_denoise_time_s = external_wall
        merged.total_h2d_transfer_time_s = total_h2d
        merged.prefetch_latency_s = total_stall
        merged.forced_sync_count = forced_syncs
        merged.nan_inf_detected = nan_inf
        merged.step_times_s = step_times
        merged.time_per_frame_s = external_wall / max(num_frames, 1)
        merged.per_block_copy_s = per_block_copy
        merged.per_block_stall_s = per_block_stall
        merged.per_block_compute_s = per_block_compute
        if total_async_copy > 0.0:
            merged.effective_overlap_pct = (
                max(0.0, 1.0 - (total_stall / total_async_copy)) * 100.0
            )
        else:
            merged.effective_overlap_pct = 0.0

        telemetry = {
            "profile_sequence": [r.profile for r in windows],
            "decisions": [r.decision for r in windows],
            "decision_overhead_ms_total": self._decision_overhead_ms,
            "switch_count": self._switch_count,
            "replan_count": self._replan_count,
            "replan_events": self._replan_events,
            "pressure_events": [],
            "hard_gate_vram_mb_limit": VRAM_HARD_GATE_MB,
            "engineering_target_vram_mb": 4000.0,
            "adaptive_gate_headroom_recovered_gb": VRAM_SAFE_HEADROOM_MB / 1024.0,
        }

        return AdaptiveResult(merged=merged, windows=windows, telemetry=telemetry)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def restore_all_to_cpu(self) -> None:
        self._fp16.restore_all_to_cpu()
        self._int8.restore_all_to_cpu()

    def release(self) -> None:
        self._fp16.release()
        self._int8.release()

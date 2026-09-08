"""
marley/core/policies.py
=======================
F4 decision primitives — pure (no CUDA), deterministic policy objects.

This module defines the *what* of the F4 decision layer:
  - PolicyDecision : a 3-axis decision (precision, prefetch, residency).
  - RuntimeState   : the 6 observable signals the engine feeds the selector.
  - PolicySelector : maps (profile, state) -> PolicyDecision.

It performs NO GPU work and performs NO I/O. It is deliberately separate from
marley/ops/*.py so the decision logic can be unit-tested without CUDA and so the
frozen F3/F3+INT8 execution primitives are never duplicated here.

Authorized contract: docs/F4_TEST_SPEC_01.md (v3, approved by Director).
Do NOT modify marley/ops/async_stream*.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Precision = Literal["fp16", "int8"]
PrefetchMode = Literal["aggressive", "conservative", "off"]
ResidencyMode = Literal["keep", "prefetch", "evict"]

# Allowed value sets (exposed for contract checks / unit tests).
PRECISIONS = ("fp16", "int8")
PREFETCH_MODES = ("aggressive", "conservative", "off")
RESIDENCY_MODES = ("keep", "prefetch", "evict")
PROFILES = ("performance", "memory_safe")

# ---------------------------------------------------------------------------
# Tuning constants for the memory_safe profile (per F4_TEST_SPEC v3, §2.2/§4).
# free_vram_mb below SAFE_MIN_FREE_MB is treated as "under pressure" for the
# memory_safe precision choice.
# ---------------------------------------------------------------------------
SAFE_MIN_FREE_MB = 1500.0


@dataclass(frozen=True)
class PolicyDecision:
    """
    One F4 control decision on the 3 decision axes.

    Attributes
    ----------
    precision : "fp16" | "int8"       -- weight transfer representation.
    prefetch  : "aggressive" | "conservative" | "off" -- overlap aggressiveness.
    residency : "keep" | "prefetch" | "evict" -- GPU residency intent.

    Residency is an explicit 3-state field (advisor letter §4): the interface
    must mirror the 3-axis design rather than collapse residency to a bool.
    """

    precision: Precision
    prefetch: PrefetchMode
    residency: ResidencyMode

    def __post_init__(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {self.precision!r}")
        if self.prefetch not in PREFETCH_MODES:
            raise ValueError(f"prefetch must be one of {PREFETCH_MODES}, got {self.prefetch!r}")
        if self.residency not in RESIDENCY_MODES:
            raise ValueError(f"residency must be one of {RESIDENCY_MODES}, got {self.residency!r}")

    def uses_int8(self) -> bool:
        return self.precision == "int8"

    def uses_async(self) -> bool:
        return self.prefetch != "off"


@dataclass(frozen=True)
class RuntimeState:
    """
    The 6 observable decision inputs (F4 plan §3). Frozen & cheap to construct.

    Attributes
    ----------
    free_vram_mb : float    -- free physical VRAM headroom (NVML) at decision time.
    recent_h2d_ms : float   -- recent cumulative PCIe H2D transfer latency.
    layer_compute_ms : float-- last block compute elapsed time (event timeline).
    ema_pressure_mb : float -- EMA of memory pressure (background WDDM), used MB.
    prefetch_stall_ms : float -- measured real prefetch stall in the last window.
    dequant_cost_ms : float -- on-device dequant cost on the critical path (~324 ms total).
    """

    free_vram_mb: float = 0.0
    recent_h2d_ms: float = 0.0
    layer_compute_ms: float = 0.0
    ema_pressure_mb: float = 0.0
    prefetch_stall_ms: float = 0.0
    dequant_cost_ms: float = 0.0


class PolicySelector:
    """
    Pure mapping from (profile, state) to a PolicyDecision.

    Rules (F4_TEST_SPEC v3 §2.2 / §4.4):
      performance : INT8 + aggressive prefetch + residency keep (max VRAM for min latency).
      memory_safe : residency evict always (early eviction is its identity);
                    precision fp16 when headroom is fine, int8 when free VRAM is low;
                    prefetch conservative when fine, off when under pressure.
    """

    @staticmethod
    def select(profile: str, state: RuntimeState) -> PolicyDecision:
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")

        if profile == "performance":
            return PolicyDecision(precision="int8", prefetch="aggressive", residency="keep")

        # memory_safe profile.
        under_pressure = state.free_vram_mb < SAFE_MIN_FREE_MB
        if under_pressure:
            # Tighten: reduce transfer volume + drop prefetch to free VRAM quickly.
            return PolicyDecision(precision="int8", prefetch="off", residency="evict")
        # Healthy headroom: keep speed but still evict early for safety margin.
        return PolicyDecision(precision="fp16", prefetch="conservative", residency="evict")

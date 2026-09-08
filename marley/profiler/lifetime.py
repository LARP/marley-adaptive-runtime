"""
Marley Runtime - Phase F1 - Tensor Lifetime Profiler
======================================================
Traces the lifecycle (enter/exit) of the primary Wan2.1-T2V pipeline
components (DiT transformer blocks, text encoders, VAE) and attributes
physical GPU residency (NVML) to whichever component is actively driving
execution, while the PyTorch caching allocator statistics bound the delta.

Two complementary signals (per ROADMAP §2):
  1. NVML physical residency  (ground-truth; sampled on a background thread)
  2. torch.cuda.allocated / reserved deltas at each forward boundary

Kill-gate fallback (ROADMAP Phase F1): if fine-grained dynamic tracing ever
proves too intrusive, the static weight-footprint analysis (`static_weights`)
always remains available as a pure-analytical model with no runtime overhead.

This module is framework-agnostic: it hooks plain `torch.nn.Module`s and needs
no knowledge of the diffusers pipeline internals.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# NVML singleton (same strategy as f0_baseline_real.py)
# ---------------------------------------------------------------------------
_NVML_HANDLE = None
_NVML_OK = False
try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_OK = True
except Exception:  # pragma: no cover - depends on driver
    _NVML_OK = False


def _nvml_used_mb() -> float:
    if not _NVML_OK:
        return 0.0
    return pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE).used / 2**20


# ---------------------------------------------------------------------------
# Static analytical footprint
# ---------------------------------------------------------------------------
def static_weights(module: torch.nn.Module) -> Dict[str, float]:
    """Pure-analytical byte footprint of a module's parameters by storage dtype.

    Returns {dtype_str: total_bytes}. Always usable, no CUDA/NVML required.
    """
    by_dtype: Dict[str, float] = {}
    for p in module.parameters():
        if not p.requires_grad:
            continue
        key = str(p.dtype)
        by_dtype[key] = by_dtype.get(key, 0.0) + float(p.numel() * p.element_size())
    return by_dtype


def static_weights_bytes(module: torch.nn.Module) -> float:
    return float(sum(static_weights(module).values()))


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# Per-component runtime record
# ---------------------------------------------------------------------------
@dataclass
class ComponentStat:
    name: str
    kind: str = "component"  # 'diT_block' | 'text_encoder' | 'vae' | 'component'
    forward_calls: int = 0
    total_time_s: float = 0.0
    max_time_s: float = 0.0
    alloc_delta_peak_mb: float = 0.0  # max (alloc_after - alloc_before) per call
    reserved_delta_peak_mb: float = 0.0
    nvml_peak_attributed_mb: float = 0.0  # physical residency while this drove exec
    weights_bytes: float = 0.0
    num_params: int = 0

    @property
    def weights_mb(self) -> float:
        return self.weights_bytes / 2**20


# ---------------------------------------------------------------------------
# Background NVML sampler that attributes residency to the active component
# ---------------------------------------------------------------------------
class _NvmlAttributor:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._active: List[str] = []  # stack; top = current driver of execution
        self.global_peak_mb = 0.0
        self.by_name: Dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def push(self, name: str) -> None:
        with self._lock:
            self._active.append(name)

    def pop(self) -> None:
        with self._lock:
            if self._active:
                self._active.pop()

    def snapshot(self) -> Tuple[float, Dict[str, float]]:
        """One NVML read attributed to the current top-of-stack component."""
        used = _nvml_used_mb()
        with self._lock:
            if used > self.global_peak_mb:
                self.global_peak_mb = used
            if self._active:
                cur = self._active[-1]
                if used > self.by_name.get(cur, 0.0):
                    self.by_name[cur] = used
            # shallow copies for the caller
            return used, dict(self.by_name)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.snapshot()
            time.sleep(self.interval_s)

    def start(self) -> None:
        if not _NVML_OK:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return dict(self.by_name)


# ---------------------------------------------------------------------------
# Public profiler facade
# ---------------------------------------------------------------------------
class TensorLifetimeProfiler:
    """Hooks named submodules and records their lifecycle + residency.

    attach(components) expects a sequence of (name, module) pairs. Nested
    submodule forwards fire independently, so hooking individual DiT blocks
    yields per-block residency; hooking a parent only yields coarse groups.
    """

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.stats: Dict[str, ComponentStat] = {}
        self.attributor = _NvmlAttributor(interval_s)
        self._handles: List = []
        self._order: List[str] = []

    def attach(
        self,
        components: Sequence[Tuple[str, torch.nn.Module]],
        kind: str = "component",
        recurse: bool = False,
    ) -> None:
        """Hook each component.

        By default a single pre/post hook pair is registered on the component's
        own `forward`. Some pipelines invoke a component via a method other
        than `__call__` (e.g. the VAE is decoded through `vae.decode(...)`),
        which bypasses the top-level forward hook. Set `recurse=True` to also
        hook every leaf submodule so such calls are still captured and
        attributed to the same component stat.
        """
        for name, module in components:
            self.stats[name] = ComponentStat(
                name=name,
                kind=kind,
                weights_bytes=static_weights_bytes(module),
                num_params=count_params(module),
            )
            self._order.append(name)
            if recurse:
                targets = self._leaf_modules(module) or [module]
            else:
                targets = [module]
            for m in targets:
                self._handles.append(m.register_forward_pre_hook(self._make_pre(name)))
                self._handles.append(m.register_forward_hook(self._make_post(name)))

    @staticmethod
    def _leaf_modules(module: torch.nn.Module):
        children = list(module.children())
        if not children:
            return []
        leaves = []
        for child in children:
            sub_leaves = TensorLifetimeProfiler._leaf_modules(child)
            leaves.extend(sub_leaves if sub_leaves else [child])
        return leaves

    def _make_pre(self, name: str):
        def pre_hook(module, args):
            stat = self.stats[name]
            torch.cuda.synchronize()
            stat._t0 = time.monotonic()
            stat._alloc_before = torch.cuda.memory_allocated() / 2**20
            stat._reserved_before = torch.cuda.memory_reserved() / 2**20
            self.attributor.push(name)
            return None  # do not modify inputs

        return pre_hook

    def _make_post(self, name: str):
        def post_hook(module, args, output):
            stat = self.stats[name]
            try:
                self.attributor.pop()
            finally:
                torch.cuda.synchronize()
                t1 = time.monotonic()
            dt = t1 - stat._t0
            alloc_after = torch.cuda.memory_allocated() / 2**20
            reserved_after = torch.cuda.memory_reserved() / 2**20

            stat.forward_calls += 1
            stat.total_time_s += dt
            stat.max_time_s = max(stat.max_time_s, dt)
            stat.alloc_delta_peak_mb = max(
                stat.alloc_delta_peak_mb, alloc_after - stat._alloc_before
            )
            stat.reserved_delta_peak_mb = max(
                stat.reserved_delta_peak_mb, reserved_after - stat._reserved_before
            )
            return output

        return post_hook

    def start(self) -> None:
        self.attributor.start()

    def stop(self) -> None:
        self.attributor.stop()
        # fold attributed NVML peaks into stats
        for name, peak in self.attributor.by_name.items():
            if name in self.stats:
                self.stats[name].nvml_peak_attributed_mb = max(
                    self.stats[name].nvml_peak_attributed_mb, peak
                )

    @property
    def global_nvml_peak_mb(self) -> float:
        return self.attributor.global_peak_mb

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ---- report helpers ---------------------------------------------------
    def rows(self) -> List[ComponentStat]:
        return [self.stats[n] for n in self._order]

    def to_dict(self) -> dict:
        return {
            "global_nvml_peak_mb": round(self.global_nvml_peak_mb, 1),
            "attributed_total_nvml_peak_mb": round(
                sum(s.nvml_peak_attributed_mb for s in self.rows()), 1
            ),
            "components": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "forward_calls": s.forward_calls,
                    "total_time_s": round(s.total_time_s, 3),
                    "max_time_s": round(s.max_time_s, 3),
                    "weights_mb": round(s.weights_mb, 2),
                    "num_params": s.num_params,
                    "alloc_delta_peak_mb": round(s.alloc_delta_peak_mb, 2),
                    "reserved_delta_peak_mb": round(s.reserved_delta_peak_mb, 2),
                    "nvml_peak_attributed_mb": round(s.nvml_peak_attributed_mb, 1),
                }
                for s in self.rows()
            ],
        }

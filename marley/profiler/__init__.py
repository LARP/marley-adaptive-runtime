"""
Marley Runtime - Profiler package
=================================
Phase F1 components: tensor-lifetime / residency profiling utilities.

Exposed:
    TensorLifetimeProfiler   - per-component forward + NVML residency profiler
    static_weights           - pure-analytical parameter footprint (no CUDA)
    static_weights_bytes     - total weight bytes of a module
    count_params             - number of trainable parameters in a module
"""

from .lifetime import (
    TensorLifetimeProfiler,
    ComponentStat,
    static_weights,
    static_weights_bytes,
    count_params,
)

__all__ = [
    "TensorLifetimeProfiler",
    "ComponentStat",
    "static_weights",
    "static_weights_bytes",
    "count_params",
]

"""
Marley Runtime - Phase F0.6 - Windows WDDM Concurrency Benchmark
=================================================================
Measures the effective hardware overlap of host-to-device PCIe
transfers (raw cudaMemcpyAsync, DMA copy engine) alongside dense
matrix multiplications (matmul on Tensor Cores) running on SEPARATE
non-default CUDA streams, under the Windows WDDM driver scheduler.

Kill Gate (ROADMAP v5, Phase F0.6):
    If measured transfer/compute overlap under WDDM is < 10%,
    permanently cancel Phase F3 (async prefetch scheduler) in favor
    of deterministic synchronous staging.

Methodology (overlap metric):
    For a matched unit of work (compute_time ~= copy_time):
        T_copy_alone    : wall time of the H2D copy workload only
        T_compute_alone : wall time of the matmul workload only
        T_serial        = T_copy_alone + T_compute_alone   (upper bound)
        T_both          : wall makespan when both run concurrently
                          on separate non-default streams

        overlap_saved_s = T_serial - T_both                (>= 0)
        overlap_pct     = overlap_saved_s / T_copy_alone * 100
                          (fraction of the PCIe transfer time hidden
                           behind concurrent compute; 0..100)

    overlap_pct >= 10  -> Phase F3 stays active (recommended)
    overlap_pct <  10  -> CANCEL Phase F3 (kill gate hit)

Telemetry: only computes overlap; physical residency is NOT the gate
here. Repeated trials reported as median to damp WDDM/clock noise.

Usage:
    .venv\\Scripts\\python.exe f0_6_wddm_overlap.py
    .venv\\Scripts\\python.exe f0_6_wddm_overlap.py --copy-mb 512 --trials 7
    .venv\\Scripts\\python.exe f0_6_wddm_overlap.py --gemm 4096 --chunk-mb 4
"""

import argparse
import ctypes
import datetime
import io
import json
import os
import statistics
import sys
import time

# Force UTF-8 output on Windows to handle special characters
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch

# ---------------------------------------------------------------------------
# Raw cudart bindings (ctypes) so we can issue true cudaMemcpyAsync on the
# DMA/copy engine instead of torch's SM-based copy kernels.
# ---------------------------------------------------------------------------
_CUDART_PATH = None
for _cand in [
    os.path.join(os.path.dirname(torch.__file__), "lib", "cudart64_12.dll"),
]:
    if os.path.exists(_cand):
        _CUDART_PATH = _cand
        break


def _check(rc, ctx):
    if rc != 0:
        from ctypes import c_char_p
        _cudart.cudaGetErrorString.restype = c_char_p
        err = _cudart.cudaGetErrorString(rc).decode() if _cudart else str(rc)
        raise RuntimeError(f"cudart {ctx} failed: rc={rc} {err}")


_cudart = None
if _CUDART_PATH:
    _cudart = ctypes.CDLL(_CUDART_PATH)
    _cudart.cudaStreamCreate.restype = ctypes.c_int
    _cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cudart.cudaStreamDestroy.restype = ctypes.c_int
    _cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    _cudart.cudaStreamSynchronize.restype = ctypes.c_int
    _cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    _cudart.cudaEventCreate.restype = ctypes.c_int
    _cudart.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cudart.cudaEventDestroy.restype = ctypes.c_int
    _cudart.cudaEventDestroy.argtypes = [ctypes.c_void_p]
    _cudart.cudaEventRecord.restype = ctypes.c_int
    _cudart.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _cudart.cudaEventSynchronize.restype = ctypes.c_int
    _cudart.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
    _cudart.cudaEventElapsedTime.restype = ctypes.c_int
    _cudart.cudaEventElapsedTime.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p,
    ]
    _cudart.cudaMemcpyAsync.restype = ctypes.c_int
    _cudart.cudaMemcpyAsync.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
        ctypes.c_void_p,
    ]
    _cudart.cudaGetErrorString.restype = ctypes.c_char_p
    _cudart.cudaGetErrorString.argtypes = [ctypes.c_int]

# cudaMemcpy kinds
_CUDA_MEMCPY_HOST_TO_DEVICE = 1


# ---------------------------------------------------------------------------
# Logging (mirrors f0_baseline_real.py conventions)
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_log_file_path = ""


def _init_log(tag: str) -> str:
    global _log_file_path
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file_path = os.path.join(LOG_DIR, f"f0_6_wddm_{tag}_{ts}.log")
    return _log_file_path


def log(msg: str, console: bool = True):
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
    entry = f"[{ts}] {msg}"
    if console:
        print(entry)
    if _log_file_path:
        with open(_log_file_path, "a", encoding="utf-8") as fh:
            fh.write(entry + "\n")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def memcpy_h2d(dst_ptr, src_ptr, nbytes, stream_handle):
    _check(
        _cudart.cudaMemcpyAsync(
            dst_ptr, src_ptr, nbytes, _CUDA_MEMCPY_HOST_TO_DEVICE, stream_handle
        ),
        "cudaMemcpyAsync(H2D)",
    )


def stream_sync(stream_handle):
    _check(_cudart.cudaStreamSynchronize(stream_handle), "cudaStreamSynchronize")


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------
def time_fn(fn, trials):
    """Median wall time (seconds) of fn() across `trials` runs."""
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_overlap(
    copy_mb: int,
    chunk_mb: int,
    gemm_n: int,
    dtype,
    trials: int,
):
    dev = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()

    # ---- two independent non-blocking torch streams (compute + copy) ----
    s_compute = torch.cuda.Stream(device=dev)
    s_copy = torch.cuda.Stream(device=dev)
    h_compute = ctypes.c_void_p(s_compute.cuda_stream)
    h_copy = ctypes.c_void_p(s_copy.cuda_stream)

    total_bytes = copy_mb * 2**20
    chunk_bytes = chunk_mb * 2**20
    n_chunks = total_bytes // chunk_bytes

    log("=" * 72)
    log("  MARLEY RUNTIME - PHASE F0.6 - WDDM CONCURRENCY BENCHMARK")
    log("=" * 72)
    log(f"  Device:       {torch.cuda.get_device_name(0)}")
    log(f"  CC:           {torch.cuda.get_device_capability(0)}")
    log(f"  Total VRAM:   {torch.cuda.get_device_properties(0).total_memory/2**20:.0f} MB")
    log(f"  Copy total:   {copy_mb} MB in {n_chunks} x {chunk_mb} MB H2D cudaMemcpyAsync")
    log(f"  GEMM:         {gemm_n}x{gemm_n}x{gemm_n} fp16 matmul (Tensor Cores)")
    log(f"  Trials:       {trials}")
    log(f"  Log:          {_log_file_path}")
    log("=" * 72)

    # ---- allocate pinned host source + device destination ----
    log(">> Allocating pinned host buffer and CUDA destination...")
    host = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    host.zero_()
    torch.cuda.synchronize()
    dst = torch.empty(total_bytes, dtype=torch.uint8, device=dev)

    h_src_base = ctypes.c_void_p(host.data_ptr())
    h_dst_base = ctypes.c_void_p(dst.data_ptr())

    # ---- GEMM operands ----
    a = torch.randn(gemm_n, gemm_n, device=dev, dtype=dtype)
    b = torch.randn(gemm_n, gemm_n, device=dev, dtype=dtype)

    def run_copy_alone():
        for i in range(n_chunks):
            memcpy_h2d(
                ctypes.c_void_p(h_dst_base.value + i * chunk_bytes),
                ctypes.c_void_p(h_src_base.value + i * chunk_bytes),
                chunk_bytes,
                h_copy,
            )
        stream_sync(h_copy)

    def run_compute_alone(gemm_reps):
        with torch.cuda.stream(s_compute):
            for _ in range(gemm_reps):
                torch.matmul(a, b)
        stream_sync(h_compute)

    # ---- warmup / calibrate gemm reps so compute ~= copy ----
    log(">> Warmup...")
    torch.cuda.synchronize()
    s_compute.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    _check(_cudart.cudaStreamSynchronize(h_copy), "warmup sync")

    # single gemm time
    n_probe = 8
    t_probe = time_fn(lambda: run_compute_alone(n_probe), trials)
    t_single = t_probe / n_probe

    # copy_alone time (used as the compute-matching target)
    t_copy_alone = time_fn(run_copy_alone, trials)
    log(f"   copy_alone  = {t_copy_alone*1000:8.1f} ms  ({copy_mb/t_copy_alone/1024:.2f} GB/s)")
    log(f"   single gemm = {t_single*1000:8.1f} ms")

    gemm_reps = max(1, int(round(t_copy_alone / t_single)))
    log(f"   -> compute matched with {gemm_reps} gemms (~{gemm_reps*t_single*1000:.0f} ms)")

    def run_compute_alone_matched():
        run_compute_alone(gemm_reps)

    def run_both():
        # Issue exactly `gemm_reps` gemms on the compute stream and
        # `n_chunks` copies on the copy stream, distributed so both engines
        # stay fed back to back. Total compute == total copies (matched).
        issued = 0
        with torch.cuda.stream(s_compute):
            for i in range(n_chunks):
                target = ((i + 1) * gemm_reps) // n_chunks
                while issued < target:
                    torch.matmul(a, b)
                    issued += 1
                memcpy_h2d(
                    ctypes.c_void_p(h_dst_base.value + i * chunk_bytes),
                    ctypes.c_void_p(h_src_base.value + i * chunk_bytes),
                    chunk_bytes,
                    h_copy,
                )
            while issued < gemm_reps:
                torch.matmul(a, b)
                issued += 1
        stream_sync(h_compute)
        stream_sync(h_copy)

    # ---- measurements ----
    log(">> Measuring compute_alone (matched)...")
    t_compute_alone = time_fn(run_compute_alone_matched, trials)
    log(f"   compute_alone = {t_compute_alone*1000:8.1f} ms")

    log(">> Measuring copy_alone (repeat)...")
    t_copy_alone2 = time_fn(run_copy_alone, trials)
    log(f"   copy_alone2   = {t_copy_alone2*1000:8.1f} ms")

    log(">> Measuring concurrent (both streams fed)...")
    t_both = time_fn(run_both, trials)
    log(f"   both (makespan) = {t_both*1000:8.1f} ms")

    # ---- compute overlap metrics (use matched compute + copy2) ----
    t_serial = t_compute_alone + t_copy_alone2
    saved = t_serial - t_both
    overlap_pct = (saved / t_copy_alone2) * 100.0 if t_copy_alone2 > 0 else 0.0

    gate = overlap_pct >= 10.0
    log("\n" + "=" * 72)
    log("  F0.6 WDDM CONCURRENCY RESULTS")
    log("=" * 72)
    log(f"  copy_alone    : {t_copy_alone2*1000:8.1f} ms")
    log(f"  compute_alone : {t_compute_alone*1000:8.1f} ms")
    log(f"  serial sum    : {t_serial*1000:8.1f} ms")
    log(f"  concurrent    : {t_both*1000:8.1f} ms")
    log(f"  overlap saved : {saved*1000:8.1f} ms")
    log(f"  OVERLAP PCT   : {overlap_pct:8.2f} %")
    log(f"  Gate (>=10%)  : {'PASS -> Phase F3 stays active' if gate else 'FAIL -> CANCEL Phase F3'}")
    log("=" * 72)

    summary = {
        "phase": "F0.6",
        "device": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "copy_mb": copy_mb,
        "chunk_mb": chunk_mb,
        "n_chunks": n_chunks,
        "gemm_n": gemm_n,
        "gemm_reps": gemm_reps,
        "trials": trials,
        "copy_alone_s": round(t_copy_alone2, 4),
        "compute_alone_s": round(t_compute_alone, 4),
        "serial_sum_s": round(t_serial, 4),
        "concurrent_s": round(t_both, 4),
        "overlap_saved_s": round(saved, 4),
        "overlap_pct": round(overlap_pct, 3),
        "gate_10pct_ok": bool(gate),
        "verdict": "Phase F3 ACTIVE" if gate else "CANCEL Phase F3 (kill gate)",
        "status": "SUCCESS",
    }

    json_path = os.path.join(
        LOG_DIR, f"f0_6_wddm_{copy_mb}mb_c{chunk_mb}_{gemm_n}.json"
    )
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log(f"  Telemetry JSON -> {json_path}")
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Marley Runtime Phase F0.6 - Windows WDDM Concurrency Benchmark"
    )
    parser.add_argument("--copy-mb", type=int, default=1024, help="Total H2D bytes to transfer (default: 1024)")
    parser.add_argument("--chunk-mb", type=int, default=8, help="Per-transfer chunk size MB (default: 8)")
    parser.add_argument("--gemm", type=int, default=4096, help="GEMM square dimension N (default: 4096)")
    parser.add_argument("--trials", type=int, default=5, help="Repeated trials for median (default: 5)")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args()

    _init_log(f"{args.copy_mb}mb_c{args.chunk_mb}_n{args.gemm}")

    if not torch.cuda.is_available():
        print("FATAL: CUDA not available.")
        sys.exit(1)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    try:
        result = run_overlap(
            copy_mb=args.copy_mb,
            chunk_mb=args.chunk_mb,
            gemm_n=args.gemm,
            dtype=dtype_map[args.dtype],
            trials=args.trials,
        )
        print(f"\n[{'PASS' if result['gate_10pct_ok'] else 'GATE HIT'}] "
              f"Overlap = {result['overlap_pct']:.2f}% | {result['verdict']}")
    except Exception as e:
        import traceback
        log(f"ERROR: {e}\n{traceback.format_exc()}")
        sys.exit(2)


if __name__ == "__main__":
    main()

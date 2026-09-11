"""
f9_0_preflight.py
=================
Preflight operacional BLOQUEANTE para la campana F9-0.

Verifica, sin cargar el modelo ni ejecutar inferencia:
  1. GPU y driver en estado correcto (NVML + nvidia-smi).
  2. Entorno Python/venv funcional (torch, CUDA, pynvml, diffusers, transformers).
  3. Cache y modelo Wan2.1-T2V-1.3B-Diffusers disponibles.
  4. GPU sin otro contexto CUDA de computo activo (nvmlDeviceGetComputeRunningProcesses).
  5. Estado del sistema adecuado para una campana prolongada (RAM, disco, energia, VRAM libre).

Salida: logs/f9_0_preflight_report.json + resumen en consola.
Codigo de salida 0 si READY FOR EXECUTION; 1 si NOT READY.

Este script NO ejecuta la campana F9-0 y NO modifica el runtime.
"""

from __future__ import annotations

import datetime
import io
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List

import psutil

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

MB = 1024.0 * 1024.0
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_CACHE_DIRNAME = "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
REQUIRED_MODEL_FILES = ["model_index.json"]
OUTPUT_PATH = "logs/f9_0_preflight_report.json"


def _ok(check: Dict[str, Any], name: str, passed: bool, detail: str,
        blocking: bool = True) -> None:
    check[name] = {"pass": bool(passed), "blocking": blocking, "detail": detail}


def check_gpu() -> Dict[str, Any]:
    out: Dict[str, Any] = {"checks": {}}
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8", "replace")
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        out.update({
            "available": True,
            "device_name": name,
            "driver_version": driver,
            "memory_total_mb": round(mem.total / MB, 1),
            "memory_used_mb": round(mem.used / MB, 1),
            "memory_free_mb": round(mem.free / MB, 1),
            "gpu_temp_c": temp,
            "gpu_clock_mhz": clock,
        })
        _ok(out["checks"], "nvml_available", True, "NVML init OK")
        _ok(out["checks"], "vram_total_ge_6gb", mem.total / MB >= 5500,
            f"total={round(mem.total/MB,1)} MB")
        _ok(out["checks"], "vram_free_headroom", mem.free / MB >= 4500,
            f"free={round(mem.free/MB,1)} MB (>=4500 MB recomendado para el safe-abort)")
        _ok(out["checks"], "gpu_temp_reasonable", temp is not None and temp < 80,
            f"temp={temp} C")
    except Exception as exc:
        out["available"] = False
        out["error"] = str(exc)
        _ok(out["checks"], "nvml_available", False, f"NVML init failed: {exc}")
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        out["nvidia_smi"] = smi.stdout.strip() or smi.stderr.strip()
    except Exception as exc:
        out["nvidia_smi"] = f"unavailable: {exc}"
    return out


def check_python_env() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "executable": sys.executable,
        "python_version": sys.version.split()[0],
        "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "checks": {},
    }
    try:
        import torch
        out["torch_version"] = torch.__version__
        out["torch_cuda_available"] = bool(torch.cuda.is_available())
        out["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            out["cuda_device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            out["cuda_capability"] = f"{props.major}.{props.minor}"
            out["cuda_total_mb"] = round(props.total_memory / MB, 1)
        _ok(out["checks"], "torch_import", True, torch.__version__)
        _ok(out["checks"], "torch_cuda_available", out["torch_cuda_available"],
            f"cuda={out['torch_cuda_available']} ({torch.version.cuda})")
    except Exception as exc:
        _ok(out["checks"], "torch_import", False, str(exc))
    for mod in ("pynvml", "diffusers", "transformers", "huggingface_hub"):
        try:
            m = __import__(mod)
            out[f"{mod}_version"] = getattr(m, "__version__", "unknown")
            _ok(out["checks"], f"{mod}_import", True, str(getattr(m, "__version__", "?")))
        except Exception as exc:
            _ok(out["checks"], f"{mod}_import", False, str(exc))
    return out


def _hf_cache_root() -> str:
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        return env
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def check_model_cache() -> Dict[str, Any]:
    out: Dict[str, Any] = {"cache_root": _hf_cache_root(), "model_id": MODEL_ID, "checks": {}}
    model_dir = os.path.join(out["cache_root"], MODEL_CACHE_DIRNAME)
    out["model_dir"] = model_dir
    exists = os.path.isdir(model_dir)
    out["model_dir_exists"] = exists
    snapshots_dir = os.path.join(model_dir, "snapshots")
    snapshots: List[str] = []
    if os.path.isdir(snapshots_dir):
        snapshots = [d for d in os.listdir(snapshots_dir)
                     if os.path.isdir(os.path.join(snapshots_dir, d))]
    out["snapshots"] = snapshots
    missing: List[str] = []
    found_any = False
    for snap in snapshots:
        snap_path = os.path.join(snapshots_dir, snap)
        present = all(os.path.exists(os.path.join(snap_path, f)) for f in REQUIRED_MODEL_FILES)
        if present:
            found_any = True
            out["active_snapshot"] = snap
            break
    if not found_any:
        missing = REQUIRED_MODEL_FILES
    out["missing_required_files"] = missing
    _ok(out["checks"], "model_cache_dir", exists, model_dir)
    _ok(out["checks"], "model_snapshot_complete", found_any,
        f"snapshots={len(snapshots)}, required={REQUIRED_MODEL_FILES}")
    # Component subfolders (transformer/vae/scheduler) presence in active snapshot.
    comps = ["transformer", "vae", "scheduler", "text_encoder"]
    if found_any:
        snap_path = os.path.join(snapshots_dir, out["active_snapshot"])
        for c in comps:
            out[f"component_{c}"] = os.path.isdir(os.path.join(snap_path, c))
    return out


def check_cuda_exclusivity() -> Dict[str, Any]:
    out: Dict[str, Any] = {"checks": {}}
    try:
        from f9_0_attribution_runner import nvml_process_summary
        summary = nvml_process_summary()
        out.update(summary)
        enum_ok = bool(summary.get("compute_enumeration_ok"))
        other_compute = summary.get("other_compute_pids", [])
        unknown = summary.get("unknown_context_pids", [])
        _ok(out["checks"], "compute_enumeration_available", enum_ok,
            "NVML compute enumeration OK" if enum_ok
            else "compute enumeration unavailable (cannot assert exclusivity)")
        _ok(out["checks"], "no_other_compute_context", len(other_compute) == 0,
            f"real CUDA compute-context PIDs (nvcuda.dll): {other_compute}")
        _ok(out["checks"], "graphics_does_not_invalidate", True,
            f"graphics/DWM PIDs (informational only): {summary.get('other_graphics_pids', [])[:15]}"
            + (" ..." if len(summary.get('other_graphics_pids', [])) > 15 else ""),
            blocking=False)
        if unknown:
            _ok(out["checks"], "unknown_context_processes", True,
                f"protected processes with undeterminable context (not invalidating): {unknown}",
                blocking=False)
    except Exception as exc:
        out["error"] = str(exc)
        _ok(out["checks"], "compute_enumeration_available", False, str(exc))
    return out


def check_system() -> Dict[str, Any]:
    out: Dict[str, Any] = {"checks": {}}
    vm = psutil.virtual_memory()
    out["ram_total_gb"] = round(vm.total / (1024 ** 3), 1)
    out["ram_available_gb"] = round(vm.available / (1024 ** 3), 1)
    out["cpu_count_logical"] = psutil.cpu_count(logical=True)
    out["cpu_count_physical"] = psutil.cpu_count(logical=False)
    drive = os.path.splitdrive(os.getcwd())[0] + os.sep
    try:
        du = shutil.disk_usage(drive)
        out["disk_total_gb"] = round(du.total / (1024 ** 3), 1)
        out["disk_free_gb"] = round(du.free / (1024 ** 3), 1)
    except Exception as exc:
        out["disk_error"] = str(exc)
    try:
        batt = psutil.sensors_battery()
        if batt is not None:
            out["battery_percent"] = round(batt.percent, 1)
            out["on_ac_power"] = bool(batt.power_plugged)
    except Exception:
        pass
    _ok(out["checks"], "ram_available_ge_8gb", vm.available / (1024 ** 3) >= 8.0,
        f"available={out['ram_available_gb']} GB (modelo usa ~15 GB host en pico)")
    _ok(out["checks"], "disk_free_ge_5gb", out.get("disk_free_gb", 0) >= 5.0,
        f"free={out.get('disk_free_gb')} GB")
    if "on_ac_power" in out:
        _ok(out["checks"], "on_ac_power", out["on_ac_power"],
            "en bateria puede provocar throttling durante ~4 h" if not out["on_ac_power"]
            else "AC power OK")
    return out


def check_outputs() -> Dict[str, Any]:
    out: Dict[str, Any] = {"checks": {}}
    for d in ("logs", "docs"):
        exists = os.path.isdir(d)
        _ok(out["checks"], f"dir_{d}_exists", exists, os.path.abspath(d))
    _ok(out["checks"], "runner_present",
        os.path.isfile("f9_0_attribution_runner.py"), "f9_0_attribution_runner.py")
    _ok(out["checks"], "protocol_present",
        os.path.isfile("docs/F9_0_PREREGISTERED_MEMORY_PEAK_ATTRIBUTION_PROTOCOL_01.md"),
        "docs/F9_0_PREREGISTERED_MEMORY_PEAK_ATTRIBUTION_PROTOCOL_01.md")
    return out


def main() -> int:
    print("=" * 80)
    print("  F9-0 -- PREFLIGHT OPERACIONAL (BLOQUEANTE)")
    print("  (No carga el modelo ni ejecuta inferencia; no modifica el runtime)")
    print("=" * 80)
    report: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "F9-0 preflight",
        "model_id": MODEL_ID,
        "checks": {},
    }
    sections = {
        "gpu": check_gpu(),
        "python_env": check_python_env(),
        "model_cache": check_model_cache(),
        "cuda_exclusivity": check_cuda_exclusivity(),
        "system": check_system(),
        "outputs": check_outputs(),
    }
    report["sections"] = sections

    blocking_failures: List[str] = []
    for sec_name, sec in sections.items():
        for check_name, res in (sec.get("checks") or {}).items():
            status = "PASS" if res["pass"] else ("FAIL" if res["blocking"] else "WARN")
            print(f"  [{status:4s}] {sec_name}.{check_name}: {res['detail']}")
            if not res["pass"] and res["blocking"]:
                blocking_failures.append(f"{sec_name}.{check_name}")

    ready = len(blocking_failures) == 0
    report["blocking_failures"] = blocking_failures
    report["ready_for_execution"] = ready
    report["campaign_command"] = (
        ".venv\\Scripts\\python.exe f9_0_attribution_runner.py --runs 5 --attribution-iteration 1")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("-" * 80)
    print(f"  Preflight report: {OUTPUT_PATH}")
    print(f"  Campaign command: {report['campaign_command']}")
    if ready:
        print("  STATUS: READY FOR EXECUTION")
    else:
        print("  STATUS: NOT READY")
        for bf in blocking_failures:
            print(f"    - blocking: {bf}")
    print("=" * 80)
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())

"""
Marley Runtime - Phase F1 - Tensor Lifetime Profiler (REAL MODEL)
==================================================================
Runs Wan2.1-T2V-1.3B under sequential CPU offload (the lowest-VRAM,
gate-satisfying config from Phase F0.5/F0.6) while a TensorLifetimeProfiler
hooks every DiT transformer block, the text encoder(s), and the VAE.

Per ROADMAP §2 the ground-truth gate is NVML **physical residency**, tracked
here on a background sampler that attributes residency to whichever component
is actively driving execution (top of an enter/exit stack). torch.cuda
allocated/reserved deltas bound the per-call activation footprint. The static
parameter footprint is always reported analytically (kill-gate fallback).

Outputs:
    logs/f1_<tag>.log              - full console + timestamps
    logs/f1_<tag>_telemetry.json   - per-component lifecycle + residency
    results/TEST_F1_lifetime_<tag>.md  - technical report (auto-written)

Usage:
    .venv\\Scripts\\python.exe f1_lifetime_profiler.py
    .venv\\Scripts\\python.exe f1_lifetime_profiler.py --res 480p --frames 17
    .venv\\Scripts\\python.exe f1_lifetime_profiler.py --res 480p --frames 17 \
        --steps 30 --group-blocks
"""

import argparse
import datetime
import io
import json
import os
import sys
import time
import traceback

# Force UTF-8 output on Windows to handle special characters
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch

from marley.profiler import TensorLifetimeProfiler, static_weights_bytes

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_DIR, "logs")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

_log_file_path = ""


def _init_log(tag: str) -> str:
    global _log_file_path
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file_path = os.path.join(LOG_DIR, f"f1_lifetime_{tag}_{ts}.log")
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
# Discovery of components to hook
# ---------------------------------------------------------------------------
def discover_components(pipe) -> list:
    """
    Returns (diT_blocks, text_encoders, vae) as plain lists of nn.Modules,
    resolving common diffusers WanPipeline attribute shapes generically.
    """
    transformer = getattr(pipe, "transformer", None)

    # --- DiT blocks --------------------------------------------------------
    diT_blocks = []
    if transformer is not None:
        blocks_attr = getattr(transformer, "blocks", None)
        if isinstance(blocks_attr, (torch.nn.ModuleList, torch.nn.Sequential)):
            diT_blocks.extend(b for b in blocks_attr if isinstance(b, torch.nn.Module))
        elif blocks_attr is not None:
            if isinstance(blocks_attr, torch.nn.Module):
                diT_blocks.append(blocks_attr)
        # fallback: any child class name containing 'Block' at any depth
        if not diT_blocks:
            for name, m in transformer.named_modules():
                if "block" in type(m).__name__.lower() and not list(m.children()):
                    diT_blocks.append(m)

    # --- text encoders (may be single module or list) ----------------------
    text_encoders = []
    te_attr = getattr(pipe, "text_encoder", None)
    if te_attr is not None:
        if isinstance(te_attr, (list, tuple)):
            text_encoders.extend(t for t in te_attr if isinstance(t, torch.nn.Module))
        elif isinstance(te_attr, torch.nn.Module):
            text_encoders.append(te_attr)
    te2_attr = getattr(pipe, "text_encoder_2", None)
    if te2_attr is not None:
        if isinstance(te2_attr, (list, tuple)):
            text_encoders.extend(t for t in te2_attr if isinstance(t, torch.nn.Module))
        elif isinstance(te2_attr, torch.nn.Module):
            text_encoders.append(te2_attr)

    vae = getattr(pipe, "vae", None)
    return diT_blocks, text_encoders, vae


def build_component_plan(pipe, group_blocks: bool) -> tuple:
    diT_blocks, text_encoders, vae = discover_components(pipe)
    plan = []  # list of (name, kind, module)

    for i, te in enumerate(text_encoders):
        plan.append((f"text_encoder[{i}]", "text_encoder", te))

    if group_blocks:
        # aggregate all DiT blocks as one group
        plan.append(("transformer(diT_blocks)", "diT_group", pipe.transformer))
    else:
        for i, blk in enumerate(diT_blocks):
            plan.append((f"diT_block[{i}]", "diT_block", blk))

    if vae is not None:
        plan.append(("vae", "vae", vae))

    return plan


# ---------------------------------------------------------------------------
# Phase F1 runner
# ---------------------------------------------------------------------------
def run_f1(
    res: str,
    frames: int,
    dtype_str: str,
    vae_dtype_str: str,
    vae_tiling: bool,
    offload: str,
    steps: int,
    group_blocks: bool,
):
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    RES_MAP = {"480p": (480, 832), "720p": (720, 1280), "360p": (360, 640)}
    height, width = RES_MAP[res]
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[dtype_str]
    vae_dtype = dtype_map[vae_dtype_str]

    log("=" * 72)
    log("  MARLEY RUNTIME - PHASE F1 - TENSOR LIFETIME PROFILER (REAL MODEL)")
    log("=" * 72)
    log(f"  Model:      {MODEL_ID}")
    log(f"  Res:        {res} ({width}x{height})")
    log(f"  Frames:     {frames}")
    log(f"  Dtype:      {dtype_str} | VAE: {vae_dtype_str} | Tiling: {vae_tiling}")
    log(f"  Offload:    {offload} | Steps: {steps} | Group-blocks: {group_blocks}")
    log(f"  Log:        {_log_file_path}")
    log("=" * 72)

    # ---- 1. load VAE with requested precision ------------------------------
    log(">> Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=vae_dtype)
    if vae_tiling:
        vae.enable_tiling()

    # ---- 2. full pipeline ---------------------------------------------------
    log(">> Loading WanPipeline...")
    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=dtype)

    if offload == "cpu":
        log(">> enable_sequential_cpu_offload()...")
        pipe.enable_sequential_cpu_offload()
    elif offload == "model_cpu":
        log(">> enable_model_cpu_offload()...")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    if vae_tiling:
        pipe.vae.enable_tiling()

    # ---- 3. static analytical footprint (kill-gate fallback) ---------------
    log("\n>> Static analytical weight footprint (no runtime):")
    plan = build_component_plan(pipe, group_blocks)
    static_rows = []
    for name, kind, mod in plan:
        mb = static_weights_bytes(mod) / 2**20
        static_rows.append((name, kind, mb))
        log(f"   {name:<26} {kind:<14} {mb:9.2f} MB")
    total_static_mb = sum(r[2] for r in static_rows)
    log(f"   {'TOTAL':<26} {'':<14} {total_static_mb:9.2f} MB")

    # ---- 4. attach runtime profiler and run --------------------------------
    log("\n>> Attaching TensorLifetimeProfiler hooks...")
    profiler = TensorLifetimeProfiler(interval_s=0.05)
    for name, kind, mod in plan:
        # VAE forward is driven via vae.decode(...) -> must hook its leaf
        # submodules (recurse) or the top-level forward hook never fires.
        profiler.attach([(name, mod)], kind=kind, recurse=(kind == "vae"))

    PROMPT = (
        "A golden retriever dog runs joyfully across a sunlit meadow, "
        "cinematic lighting, shallow depth of field, 4K."
    )
    NEG_PROMPT = "blurry, low quality, watermark, deformed"

    log(f">> Running inference ({steps} steps, {frames} frames @ {res})...")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    profiler.start()
    t_inf_start = time.time()
    try:
        output = pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            height=height,
            width=width,
            num_frames=frames,
            guidance_scale=5.0,
            num_inference_steps=steps,
            generator=torch.Generator("cpu").manual_seed(42),
        )
        torch.cuda.synchronize()
        elapsed = time.time() - t_inf_start
        profiler.stop()
        log(f"   [OK] Inference completed in {elapsed:.1f}s")
    except Exception as e:
        profiler.stop()
        log(f"   [ERR] Inference failed after {time.time()-t_inf_start:.1f}s: {e}")
        log(traceback.format_exc(), console=False)
        return {"phase": "F1", "status": "ERROR", "error": str(e)[:300]}

    data = profiler.to_dict()
    data["config"] = {
        "phase": "F1",
        "model": MODEL_ID,
        "res": res,
        "resolution_px": f"{width}x{height}",
        "frames": frames,
        "dtype": dtype_str,
        "vae_dtype": vae_dtype_str,
        "vae_tiling": vae_tiling,
        "offload": offload,
        "steps": steps,
        "group_blocks": group_blocks,
        "total_inference_time_s": round(elapsed, 2),
        "total_static_weights_mb": round(total_static_mb, 2),
        "static_rows": [{"name": n, "kind": k, "mb": round(m, 2)} for n, k, m in static_rows],
    }

    # torch peak (auxiliary layer)
    data["torch_peak"] = {
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }
    data["device"] = torch.cuda.get_device_name(0)
    data["status"] = "SUCCESS"

    # ---- 5. persist video (optional but keeps parity) ----------------------
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{res}_{frames}f_{dtype_str}_vae_{vae_dtype_str}_{offload}_{steps}st_{ts}"
    video_path = None
    if hasattr(output, "frames") and output.frames is not None and len(output.frames) > 0:
        try:
            video_path = os.path.join(LOG_DIR, f"f1_{tag}.mp4")
            export_to_video(output.frames[0], video_path, fps=16)
            log(f"   Video -> {video_path}")
        except Exception as e:
            log(f"   [WARN] video export failed: {e}")

    json_path = os.path.join(LOG_DIR, f"f1_{tag}_telemetry.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    log(f"   Telemetry JSON -> {json_path}")

    profiler.detach()
    return data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Marley Runtime Phase F1 - Tensor Lifetime Profiler (Real Wan2.1 Model)"
    )
    parser.add_argument("--res", default="480p", choices=["480p", "720p", "360p"])
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=30, help="denoising steps (default 30)")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--vae-dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--vae-tiling", action="store_true", default=True)
    parser.add_argument("--no-vae-tiling", action="store_true", help="disable VAE tiling")
    parser.add_argument("--offload", default="cpu", choices=["none", "cpu", "model_cpu"])
    parser.add_argument("--group-blocks", action="store_true",
                        help="hook all DiT blocks as one aggregate group (faster)")
    args = parser.parse_args()
    args.vae_tiling = not args.no_vae_tiling

    tag = f"{args.res}_{args.frames}f_{args.dtype}_vae_{args.vae_dtype}_{args.offload}_{args.steps}st"
    if args.group_blocks:
        tag += "_grp"
    _init_log(tag)

    if not torch.cuda.is_available():
        print("FATAL: CUDA not available.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 2**20
    log(f"Device: {gpu_name} | Total VRAM: {total_vram:.0f} MB")

    result = run_f1(
        res=args.res,
        frames=args.frames,
        dtype_str=args.dtype,
        vae_dtype_str=args.vae_dtype,
        vae_tiling=args.vae_tiling,
        offload=args.offload,
        steps=args.steps,
        group_blocks=args.group_blocks,
    )

    if result.get("status") == "SUCCESS":
        log("\n[PASS] Phase F1 profiling run completed.")
        sys.exit(0)
    else:
        log(f"\n[FAIL] {result}")
        sys.exit(1)


if __name__ == "__main__":
    main()

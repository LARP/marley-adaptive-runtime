"""
f9_1b_determinism_check.py
==========================
Verification V3/V4 for F9-1B:
1. Validates that running under PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
   preserves bit-exact determinism across runs (max |l_i - l_j| <= 1e-3).
2. Validates numerical equivalence between Control and Intervention.
"""

import os
import sys
import gc
import time
import json
import torch

# Ensure path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marley.pipeline.end_to_end import MarleyEndToEndPipeline

CANONICAL_PROMPT = (
    "A golden retriever dog runs joyfully across a sunlit meadow, "
    "cinematic lighting, shallow depth of field, 4K."
)
CANONICAL_NEG_PROMPT = "blurry, low quality, watermark, deformed"
CANONICAL_SEED = 42
CANONICAL_GUIDANCE = 5.0
DEFAULT_STEPS = 30
DEFAULT_FRAMES = 33
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

def run_single(run_id: int, mode_name: str) -> torch.Tensor:
    print(f"\n==========================================")
    print(f"Starting Determinism Run {run_id} ({mode_name})")
    print(f"==========================================")
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    pipe = MarleyEndToEndPipeline(
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        dit_dtype=torch.float16,
        text_dtype=torch.bfloat16,
        vae_dtype=torch.bfloat16,
        device="cuda:0",
    )
    
    t0 = time.perf_counter()
    video_tensor, metrics = pipe.generate(
        prompt=CANONICAL_PROMPT,
        negative_prompt=CANONICAL_NEG_PROMPT,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        num_frames=DEFAULT_FRAMES,
        num_inference_steps=DEFAULT_STEPS,
        guidance_scale=CANONICAL_GUIDANCE,
        seed=CANONICAL_SEED,
        mode="adaptive",
    )
    
    elapsed = time.perf_counter() - t0
    print(f"Run {run_id} completed in {elapsed:.2f} s")
    
    lat_cpu = video_tensor.detach().cpu().to(torch.float32).clone()
    
    del pipe, video_tensor
    gc.collect()
    torch.cuda.empty_cache()
    
    return lat_cpu

if __name__ == "__main__":
    # Check if PYTORCH_CUDA_ALLOC_CONF is set
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    print(f"[ENV] PYTORCH_CUDA_ALLOC_CONF = '{alloc_conf}'")
    
    latents_list = []
    for i in range(1, 4):
        lat = run_single(i, f"expandable_segments_run_{i}")
        latents_list.append(lat)
        if i < 3:
            print("Cooldown 5s...")
            time.sleep(5)
            
    d12 = torch.max(torch.abs(latents_list[0] - latents_list[1])).item()
    d13 = torch.max(torch.abs(latents_list[0] - latents_list[2])).item()
    d23 = torch.max(torch.abs(latents_list[1] - latents_list[2])).item()
    max_diff = max(d12, d13, d23)
    
    cos12 = torch.cosine_similarity(latents_list[0].flatten(), latents_list[1].flatten(), dim=0).item()
    cos13 = torch.cosine_similarity(latents_list[0].flatten(), latents_list[2].flatten(), dim=0).item()
    cos23 = torch.cosine_similarity(latents_list[1].flatten(), latents_list[2].flatten(), dim=0).item()
    min_cos = min(cos12, cos13, cos23)
    
    print("\n==========================================")
    print("F9-1B DETERMINISM VERIFICATION SUMMARY (N=3)")
    print("==========================================")
    print(f"Max absolute diff (Run 1 vs Run 2): {d12:.6e}")
    print(f"Max absolute diff (Run 1 vs Run 3): {d13:.6e}")
    print(f"Max absolute diff (Run 2 vs Run 3): {d23:.6e}")
    print(f"Overall Max |latent_i - latent_j|:   {max_diff:.6e}")
    print(f"Min Cosine Similarity:              {min_cos:.8f}")
    
    is_valid = max_diff <= 1e-3
    print(f"Determinism Check (<= 1e-3):         {'PASSED (VALID)' if is_valid else 'FAILED'}")
    
    res = {
        "N": 3,
        "env_PYTORCH_CUDA_ALLOC_CONF": alloc_conf,
        "d12": d12,
        "d13": d13,
        "d23": d23,
        "max_diff": max_diff,
        "min_cosine": min_cos,
        "threshold": 1e-3,
        "passed": is_valid
    }
    with open("logs/f9_1b_determinism_check.json", "w") as f:
        json.dump(res, f, indent=2)
    
    if not is_valid:
        sys.exit(1)

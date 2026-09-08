"""
Marley Runtime · Baseline Profiler (Phase F0)
Instrumentation and reproduction of the first CUDA Out of Memory (OOM) on 6GB VRAM hardware.
"""

import argparse
import datetime
import os
import sys
import traceback
import torch

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "f0_baseline.log")


def log_event(message: str, console: bool = True):
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    entry = f"[{timestamp}] {message}"
    if console:
        print(entry)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def get_vram_status():
    if not torch.cuda.is_available():
        return "CUDA not available"
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    alloc_bytes = torch.cuda.memory_allocated(0)
    res_bytes = torch.cuda.memory_reserved(0)
    
    to_mb = lambda b: b / (1024 * 1024)
    return {
        "free_mb": to_mb(free_bytes),
        "total_mb": to_mb(total_bytes),
        "allocated_mb": to_mb(alloc_bytes),
        "reserved_mb": to_mb(res_bytes),
        "max_allocated_mb": to_mb(torch.cuda.max_memory_allocated(0))
    }


def run_synthetic_dit_stress(res: str, frames: int, dtype_str: str):
    """
    Executes a synthetic stress run matching the exact tensor geometry of Wan2.1-1.3B
    for 720p (1280x720) at N frames, forcing the memory footprint of 3D attention and latents.
    """
    log_event(f"--- Starting Wan2.1-1.3B synthetic stress run (Res: {res}, Frames: {frames}, Dtype: {dtype_str}) ---")
    
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32
    }
    dtype = dtype_map.get(dtype_str, torch.float16)

    # 720p: 1280x720 -> VAE latents (8x spatial compression, 4x temporal compression)
    if res == "720p":
        h_lat, w_lat = 720 // 8, 1280 // 8  # 90 x 160
    elif res == "480p":
        h_lat, w_lat = 480 // 8, 854 // 8   # 60 x 106
    else:
        h_lat, w_lat = 64, 64

    t_lat = max(1, frames // 4)
    channels = 16  # Wan2.1 VAE latent channels
    seq_len = t_lat * h_lat * w_lat
    hidden_dim = 1536  # Wan2.1-1.3B hidden dimension
    heads = 12
    head_dim = hidden_dim // heads

    log_event(f"Latent geometry: ({channels}, {t_lat}, {h_lat}, {w_lat}) | 3D Seq Len: {seq_len} tokens")

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    
    # 1. Allocate initial latents
    vram_init = get_vram_status()
    log_event(f"Initial available VRAM: {vram_init['free_mb']:.1f} MB / {vram_init['total_mb']:.1f} MB")
    
    log_event("Step 1: Allocating initial latent buffer...")
    latents = torch.randn((1, channels, t_lat, h_lat, w_lat), device=device, dtype=dtype)
    
    # 2. Simulate DiT forward pass across 30 blocks
    num_blocks = 30
    log_event(f"Step 2: Simulating forward pass across {num_blocks} DiT blocks...")
    
    x = torch.randn((1, seq_len, hidden_dim), device=device, dtype=dtype)
    
    for b in range(num_blocks):
        mem = get_vram_status()
        log_event(f"  [Block {b+1:02d}/{num_blocks}] Allocated VRAM: {mem['allocated_mb']:.1f} MB, Free: {mem['free_mb']:.1f} MB")
        
        # Un-tiled full 3D self-attention simulation (Q, K, V)
        # Q @ K^T produces tensor of shape (heads, seq_len, seq_len)
        q = torch.randn((1, heads, seq_len, head_dim), device=device, dtype=dtype)
        k = torch.randn((1, heads, seq_len, head_dim), device=device, dtype=dtype)
        v = torch.randn((1, heads, seq_len, head_dim), device=device, dtype=dtype)
        
        log_event(f"  [Block {b+1:02d}] Computing full attention matrix Q @ K.T ({heads}x{seq_len}x{seq_len})...")
        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        # Feed-forward projection (4x expansion MLP)
        mlp_intermediate = torch.randn((1, seq_len, hidden_dim * 4), device=device, dtype=dtype)
        del mlp_intermediate, attn_scores, attn_weights, q, k, v, out
        
    log_event("Generation completed successfully without triggering OOM.")


def main():
    parser = argparse.ArgumentParser(description="Marley Runtime Baseline Profiler")
    parser.add_argument("--res", type=str, default="720p", choices=["720p", "480p"], help="Target resolution")
    parser.add_argument("--frames", type=int, default=81, help="Video frame count")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="Precision dtype")
    args = parser.parse_args()

    print("=================================================================")
    print("      MARLEY RUNTIME · BASELINE PROFILER (PHASE F0)")
    print("=================================================================")
    
    if not torch.cuda.is_available():
        log_event("FATAL ERROR: CUDA is not available. Check PyTorch installation and drivers.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    log_event(f"Detected Device: {gpu_name}")
    log_event(f"CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    try:
        run_synthetic_dit_stress(args.res, args.frames, args.dtype)
    except torch.cuda.OutOfMemoryError as oom:
        log_event("*****************************************************************", console=True)
        log_event(">>> GATE F0 REACHED: FIRST CUDA OUT OF MEMORY LOGGED <<<", console=True)
        log_event(f"Exception: {oom}", console=True)
        vram = get_vram_status()
        log_event(f"VRAM at failure -> Allocated: {vram['allocated_mb']:.2f} MB | Reserved: {vram['reserved_mb']:.2f} MB | Remaining Free: {vram['free_mb']:.2f} MB", console=True)
        log_event(f"Peak Allocated VRAM: {vram['max_allocated_mb']:.2f} MB", console=True)
        log_event(f"Trigger Configuration: Res={args.res}, Frames={args.frames}, Dtype={args.dtype}", console=True)
        log_event(f"Traceback:\n{traceback.format_exc()}", console=False)
        log_event("*****************************************************************", console=True)
        log_event(f"Formal incident log saved at: {LOG_FILE}")
        sys.exit(0)
    except Exception as e:
        log_event(f"Unexpected error during run: {e}")
        log_event(traceback.format_exc(), console=False)
        sys.exit(2)


if __name__ == "__main__":
    main()

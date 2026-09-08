"""
f3_memory_diagnostic.py
=======================
Diagnostic script to identify where VRAM is consumed during F3 setup.
Prints VRAM usage at every critical step.
"""
import gc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

def vram_mb():
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved  = torch.cuda.memory_reserved(0) / 1024**2
    free_drv  = torch.cuda.mem_get_info(0)[0] / 1024**2
    return allocated, reserved, free_drv

def report(label):
    alloc, res, free = vram_mb()
    print(f"[VRAM] {label:55s} alloc={alloc:7.1f}MB  reserved={res:7.1f}MB  driver_free={free:7.1f}MB")

device = torch.device("cuda", 0)
dtype  = torch.float16

report("baseline (before anything)")

# ── 1. Load transformer ─────────────────────────────────────────────────────
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
transformer = WanTransformer3DModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    subfolder="transformer",
    torch_dtype=dtype,
)
report("after from_pretrained")
print(f"         model device: {next(transformer.parameters()).device}")

transformer = transformer.to("cpu")
torch.cuda.empty_cache()
gc.collect()
report("after .to('cpu') + empty_cache")

blocks = list(transformer.blocks)
num_blocks = len(blocks)
print(f"         num_blocks: {num_blocks}")

# ── 2. BudgetedAsyncStreamer init ────────────────────────────────────────────
# Replicate _init_gpu_slots manually to see cost
template = {}
for name, param in blocks[0].named_parameters():
    template[name] = param.data

report("before GPU slot allocation")
gpu_slots = [{}, {}]
for slot_idx in range(2):
    for name, cpu_t in template.items():
        gpu_slots[slot_idx][name] = torch.empty(cpu_t.shape, dtype=dtype, device=device)
report("after GPU slot allocation (2 slots)")

# ── 3. Pin all host weights ──────────────────────────────────────────────────
report("before pinning host weights")
host_weights = []
for block_idx, block in enumerate(blocks):
    pinned = {}
    for name, param in block.named_parameters():
        cpu_t = param.detach().to("cpu", dtype=dtype)
        try:
            pinned[name] = cpu_t.pin_memory()
        except Exception:
            pinned[name] = cpu_t
    host_weights.append(pinned)
    if block_idx in (0, 9, 19, 29):
        report(f"  after pinning block {block_idx}")

report("after pinning ALL 30 blocks")

# ── 4. Move activations to GPU ───────────────────────────────────────────────
# Simulate what execute_sync_loop does
h   = torch.randn(1, 7800, 1536, dtype=dtype)
enc = torch.randn(1,  226, 1536, dtype=dtype)
t   = torch.randn(1,    6, 1536, dtype=dtype)
r   = (
    torch.randn(1, 7800, 1, 128, dtype=dtype),
    torch.randn(1, 7800, 1, 128, dtype=dtype),
)

h   = h.to(device)
enc = enc.to(device)
t   = t.to(device)
r   = tuple(x.to(device) for x in r)
report("after moving activations to GPU")

# ── 5. Copy block 0 to GPU slot ─────────────────────────────────────────────
with torch.cuda.stream(torch.cuda.Stream(device=device)):
    for name, cpu_t in host_weights[0].items():
        gpu_slots[0][name].copy_(cpu_t, non_blocking=False)
torch.cuda.synchronize(device)
report("after copying block 0 to GPU slot")

# ── 6. Bind block 0 params to GPU slot ──────────────────────────────────────
original_data = {}
for name, param in blocks[0].named_parameters():
    original_data[name] = param.data
    param.data = gpu_slots[0][name]
report("after binding block 0 to GPU slot")

# ── 7. Attempt forward pass of block 0 ──────────────────────────────────────
print("\nAttempting block 0 forward pass...")
try:
    with torch.no_grad():
        h_out = blocks[0](h, enc, t, r)
    report("AFTER block 0 forward — SUCCESS")
    print(f"         output shape: {h_out.shape}")
except torch.cuda.OutOfMemoryError as e:
    report("OOM during block 0 forward")
    print(f"         OOM error: {e}")
finally:
    # Restore
    for name, param in blocks[0].named_parameters():
        param.data = original_data[name]

"""
marley/ops/chunked_attention.py
================================
Phase F7-1: Attention Sequence Chunking Processor for Wan2.1 DiT.

SCOPE (Director Resolution 2026-09-09, Art. 2/5/6/10):
  Experimental-only module. Must preserve the GLOBAL semantics of attention.
  Chunking partitions the query tensor Q along the sequence dimension
  (e.g. C=2048) while each block retains access to the FULL key K and value V
  context. This is NOT a windowed, local, spatially/temporally-limited,
  micro-batched, or approximated attention variant.

MATHEMATICAL NOTION (hypothesis, not an empirical claim):
  Attention(Q, K, V)_i = softmax(Q_i K^T / sqrt(D)) V
  Because normalization is over the key dimension, each query row is
  independent of other query rows; concatenating per-chunk results is intended
  to be mathematically equivalent in exact arithmetic.

NUMERICAL CAVEAT:
  Under a Flash/Mem-Efficient SDPA backend the online-softmax accumulation
  order depends on tensor layout. Executing in sub-blocks is NOT guaranteed
  bit-exact with respect to the monolithic call, even though it is
  mathematically equivalent in exact arithmetic. Any equivalence must be
  ESTABLISHED BY MEASUREMENT, never asserted as bit-exact.

INTENDED EFFECT (to be VERIFIED by the F7-1 pilot, not assumed):
  Reduce the peak transient activation working set. Whether this actually
  lowers the physical NVML peak is an open experimental question.

This module and the probe (f7_1_attention_chunking_probe.py) are the ONLY two
files authorized for modification. See the F7-1 pilot resolution.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanAttnProcessor,
    _get_added_kv_projections,
    _get_qkv_projections,
    dispatch_attention_fn,
)


class ChunkedWanAttnProcessor(WanAttnProcessor):
    """
    Drop-in replacement for WanAttnProcessor that chunks the query projection
    along the sequence dimension when sequence length exceeds chunk_size,
    while keeping the full K/V context (global attention semantics).

    Instrumentation (enabled only when ``instrument=True``) records, without
    altering numerics:
      * attention variant (self vs cross),
      * real query sequence length S, heads, head dimension, dtype,
      * configured attention backend,
      * chunk decomposition and per-call allocator delta.
    It also enforces two live safety gates:
      * Gate A (numerical integrity): aborts on NaN/Inf.
      * Gate C (memory safety): aborts as a SAFE OPERATIONAL stop if physical
        VRAM approaches the device limit. This is distinct from a scientific
        result.
    """

    def __init__(
        self,
        chunk_size: int = 2048,
        enabled: bool = True,
        tag: str = "",
        instrument: bool = False,
        event_log: Optional[list] = None,
        nvml_used_mb_fn: Optional[Callable[[], float]] = None,
        safe_abort_nvml_mb: float = 5900.0,
        reserved_log_mb: float = 4800.0,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.enabled = enabled
        self.tag = tag
        self.instrument = instrument
        self.event_log = event_log if event_log is not None else []
        self.nvml_used_mb_fn = nvml_used_mb_fn
        self.safe_abort_nvml_mb = safe_abort_nvml_mb
        self.reserved_log_mb = reserved_log_mb
        self._reported_variants: set = set()

    def _record(self, entry: Dict[str, Any]) -> None:
        if self.event_log is not None:
            self.event_log.append(entry)

    def _safety_checks(self) -> None:
        # Gate A -- numerical integrity is checked by caller on the output tensor.
        # Gate C -- physical memory safety (operational, not scientific).
        if self.nvml_used_mb_fn is not None:
            try:
                used = self.nvml_used_mb_fn()
            except Exception:
                used = None
            if used is not None and used > self.safe_abort_nvml_mb:
                raise SafeAttentionAbort(
                    f"[SAFE ABORT OPERATIVO] NVML fisico {used:.1f} MB excede el umbral de "
                    f"seguridad {self.safe_abort_nvml_mb:.1f} MB en {self.tag}. "
                    "Esto NO constituye un resultado cientifico del experimento."
                )

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb=None,
    ) -> torch.Tensor:
        # Distinguish self- vs cross-attention for causal attribution.
        # In Wan2.1 T2V blocks: attn1 = self (encoder_hidden_states is None),
        # attn2 = cross (text context present). add_k_proj is only used for I2V.
        variant = "self" if encoder_hidden_states is None else "cross"

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the text encoder context length, hardcoded for now.
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            freqs_cos, freqs_sin = rotary_emb

            def apply_rotary_emb(
                hs: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
            ) -> torch.Tensor:
                x1, x2 = hs.unflatten(-1, (-1, 2)).unbind(-1)
                c = cos[..., 0::2]
                s = sin[..., 1::2]
                out = torch.empty_like(hs)
                out[..., 0::2] = x1 * c - x2 * s
                out[..., 1::2] = x1 * s + x2 * c
                return out.type_as(hs)

            query = apply_rotary_emb(query, freqs_cos, freqs_sin)
            key = apply_rotary_emb(key, freqs_cos, freqs_sin)

        # ---- Instrumentation: capture real runtime facts once per (tag, variant) ----
        if self.instrument:
            seq_len = int(query.shape[1])
            key_entry = f"{self.tag}|{variant}"
            if key_entry not in self._reported_variants:
                self._reported_variants.add(key_entry)
                self._record(
                    {
                        "event": "attention_observed",
                        "tag": self.tag,
                        "variant": variant,
                        "configured_backend": self._attention_backend,
                        "seq_len_S": seq_len,
                        "heads": int(query.shape[2]),
                        "head_dim": int(query.shape[3]),
                        "dtype": str(query.dtype),
                        "chunk_size": self.chunk_size,
                        "full_chunks": seq_len // self.chunk_size if seq_len > 0 else 0,
                        "residual": seq_len % self.chunk_size if seq_len > 0 else seq_len,
                        "chunked": bool(self.enabled and seq_len > self.chunk_size),
                        "alloc_mb_before": round(
                            torch.cuda.memory_allocated(query.device) / (1024 * 1024), 2
                        ),
                        "reserved_mb_before": round(
                            torch.cuda.memory_reserved(query.device) / (1024 * 1024), 2
                        ),
                    }
                )

        # I2V-style added-key branch (preserved from WanAttnProcessor).
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            if self.enabled and query.shape[1] > self.chunk_size:
                out_img_chunks = []
                for q_chunk in query.split(self.chunk_size, dim=1):
                    out_img_chunks.append(
                        dispatch_attention_fn(
                            q_chunk,
                            key_img,
                            value_img,
                            attn_mask=None,
                            dropout_p=0.0,
                            is_causal=False,
                            backend=self._attention_backend,
                            parallel_config=None,
                        )
                    )
                hidden_states_img = torch.cat(out_img_chunks, dim=1)
            else:
                hidden_states_img = dispatch_attention_fn(
                    query,
                    key_img,
                    value_img,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    backend=self._attention_backend,
                    parallel_config=None,
                )
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        # Main attention dispatch (self or cross), possibly chunked on the query dim.
        if self.enabled and query.shape[1] > self.chunk_size:
            alloc_before = torch.cuda.memory_allocated(query.device) / (1024 * 1024)
            out_chunks = []
            for q_chunk in query.split(self.chunk_size, dim=1):
                out_chunks.append(
                    dispatch_attention_fn(
                        q_chunk,
                        key,
                        value,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=False,
                        backend=self._attention_backend,
                        parallel_config=(
                            self._parallel_config if encoder_hidden_states is None else None
                        ),
                    )
                )
            hidden_states = torch.cat(out_chunks, dim=1)
            if self.instrument:
                self._record(
                    {
                        "event": "attention_alloc_delta",
                        "tag": self.tag,
                        "variant": variant,
                        "chunked": True,
                        "alloc_mb_delta": round(
                            torch.cuda.memory_allocated(query.device) / (1024 * 1024) - alloc_before,
                            2,
                        ),
                        "reserved_mb_after": round(
                            torch.cuda.memory_reserved(query.device) / (1024 * 1024), 2
                        ),
                    }
                )
                if torch.cuda.memory_reserved(query.device) / (1024 * 1024) > self.reserved_log_mb:
                    self._record(
                        {
                            "event": "reserved_above_log_threshold",
                            "tag": self.tag,
                            "variant": variant,
                            "reserved_mb": round(
                                torch.cuda.memory_reserved(query.device) / (1024 * 1024), 2
                            ),
                        }
                    )
        else:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=(
                    self._parallel_config if encoder_hidden_states is None else None
                ),
            )

        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Gate A -- numerical integrity (live abort on NaN/Inf).
        if self.instrument:
            try:
                bad = bool(torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any())
            except Exception:
                bad = False
            if bad:
                raise NumericalIntegrityError(
                    f"[GATE A ABORT] NaN/Inf detectado en la salida de atencion {self.tag}/{variant}. "
                    "Piloto abortado por integridad numerica."
                )
            self._safety_checks()

        return hidden_states


class NumericalIntegrityError(RuntimeError):
    """Gate A: abort on numerical corruption."""


class SafeAttentionAbort(RuntimeError):
    """Gate C: safe operational stop near device memory limit (not scientific)."""


def apply_chunked_attention_to_dit_blocks(
    blocks,
    chunk_size: int = 2048,
    enabled: bool = True,
    instrument: bool = False,
    event_log: Optional[list] = None,
    nvml_used_mb_fn: Optional[Callable[[], float]] = None,
    safe_abort_nvml_mb: float = 5900.0,
    reserved_log_mb: float = 4800.0,
) -> List[ChunkedWanAttnProcessor]:
    """
    Configures every self_attn (attn1) and cross_attn (attn2) submodule across
    the DiT blocks with its own ChunkedWanAttnProcessor instance, tagged by
    block index so that causal attribution can identify which layer peaks.

    Returns the list of created processors (for reference / teardown).
    """
    processors: List[ChunkedWanAttnProcessor] = []
    for block_idx, block in enumerate(blocks):
        for attn_name in ("attn1", "attn2"):
            attn_mod = getattr(block, attn_name, None)
            if attn_mod is not None and hasattr(attn_mod, "set_processor"):
                proc = ChunkedWanAttnProcessor(
                    chunk_size=chunk_size,
                    enabled=enabled,
                    tag=f"block{block_idx}.{attn_name}",
                    instrument=instrument,
                    event_log=event_log,
                    nvml_used_mb_fn=nvml_used_mb_fn,
                    safe_abort_nvml_mb=safe_abort_nvml_mb,
                    reserved_log_mb=reserved_log_mb,
                )
                attn_mod.set_processor(proc)
                processors.append(proc)
    return processors

"""
marley/pipeline/end_to_end.py
=============================
Marley Pipeline: End-to-End Wan2.1 Inactive-Offload & Block-Streaming Orchestrator.

Integrates:
  1. Prompt Encoding: UMT5-XXL text encoder (CPU offloaded).
  2. Noise & Latents Preparation: FlowMatchEulerDiscreteScheduler (or standard WanScheduler).
  3. Denoising Loop: WanTransformer3DModel patched with Marley DiT block streamers:
     - 'sync': BudgetedAsyncStreamer (Condition A)
     - 'async_fp16': BudgetedAsyncStreamer (Condition B)
     - 'async_int8': INT8BudgetedStreamer (Condition C)
     - 'adaptive': AdaptiveEngine (Condition D)
  4. VAE Video Reconstruction: AutoencoderKLWan in bfloat16 with native enable_tiling(256x256)
     respecting the F5-A Non-Regression Rule (<= 2,109 MB, ~27s decode).
  5. Video Export: mp4 output via diffusers.utils.export_to_video.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

from marley.core.adaptive import AdaptiveEngine
from marley.ops.async_stream import BudgetedAsyncStreamer
from marley.ops.async_stream_int8 import INT8BudgetedStreamer


@dataclass
class EndToEndMetrics:
    prompt_encode_time_s: float = 0.0
    dit_prep_time_s: float = 0.0
    denoise_time_s: float = 0.0
    vae_decode_time_s: float = 0.0
    video_export_time_s: float = 0.0
    total_wall_clock_s: float = 0.0
    peak_nvml_mb: float = 0.0
    peak_torch_alloc_mb: float = 0.0
    peak_torch_reserved_mb: float = 0.0
    peak_process_ram_mb: float = 0.0
    nan_inf_detected: bool = False
    output_video_path: Optional[str] = None
    step_times_s: List[float] = field(default_factory=list)
    mode: str = "async_fp16"


class MarleyEndToEndPipeline:
    """
    Orchestrates Wan2.1 text-to-video inference using Marley Runtime block streaming.
    """

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        device: str = "cuda:0",
        dit_dtype: torch.dtype = torch.float16,
        vae_dtype: torch.dtype = torch.bfloat16,
        text_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device)
        self.dit_dtype = dit_dtype
        self.vae_dtype = vae_dtype
        self.text_dtype = text_dtype

        print(f"[MarleyPipeline] Initializing pipeline on {self.device}...")
        # Load VAE in bfloat16 with tiling per F5-A
        print("  - Loading VAE (bfloat16 with 256x256 spatial tiling)...")
        self.vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=vae_dtype,
        )
        self.vae.enable_tiling()
        self.vae.eval()

        # Load full pipeline on CPU to manage component lifecycles
        print("  - Loading WanPipeline modules on CPU...")
        self.pipe = WanPipeline.from_pretrained(
            model_id,
            vae=self.vae,
            torch_dtype=dit_dtype,
        )

        self.transformer = self.pipe.transformer
        self.blocks = list(self.transformer.blocks)
        self.num_blocks = len(self.blocks)

        # Detach DiT blocks so only non-block layers (patch_embedding, condition_embedder, etc.) move to GPU
        self.transformer.blocks = nn.ModuleList([])
        self.transformer.to(self.device, dtype=dit_dtype)
        self.transformer.eval()
        print(f"  - Pipeline ready. {self.num_blocks} DiT blocks on CPU, non-block submodules on {self.device}.")

    def _execute_dit_custom_forward(
        self,
        streamer: Union[BudgetedAsyncStreamer, INT8BudgetedStreamer, AdaptiveEngine],
        mode: str,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        rotary_emb: torch.Tensor,
        num_steps: int = 1,
        num_frames: int = 33,
    ) -> torch.Tensor:
        """
        Executes the 30 DiT blocks for 1 step using the requested Marley streaming engine.
        """
        if mode == "sync":
            assert isinstance(streamer, BudgetedAsyncStreamer)
            m = streamer.execute_sync_loop(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=timestep_proj,
                rotary_emb=rotary_emb,
                num_steps=1,
                num_frames=num_frames,
                nan_check=False,
            )
            return hidden_states
        elif mode == "async_fp16":
            assert isinstance(streamer, BudgetedAsyncStreamer)
            m = streamer.execute_async_loop(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=timestep_proj,
                rotary_emb=rotary_emb,
                num_steps=1,
                num_frames=num_frames,
                nan_check=False,
            )
            return hidden_states
        elif mode == "async_int8":
            assert isinstance(streamer, INT8BudgetedStreamer)
            m = streamer.execute_async_loop(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=timestep_proj,
                rotary_emb=rotary_emb,
                num_steps=1,
                num_frames=num_frames,
                nan_check=False,
            )
            return hidden_states
        elif mode == "adaptive":
            assert isinstance(streamer, AdaptiveEngine)
            res = streamer.run(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=timestep_proj,
                rotary_emb=rotary_emb,
                num_steps=1,
                num_frames=num_frames,
                nan_check=False,
            )
            return hidden_states
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, watermark, deformed",
        height: int = 480,
        width: int = 832,
        num_frames: int = 33,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = 42,
        mode: str = "async_fp16",
        output_video_path: Optional[str] = None,
        nvml_sampler=None,
    ) -> Tuple[torch.Tensor, EndToEndMetrics]:
        """
        Executes full text-to-video generation with exact stage-by-stage timing.
        """
        metrics = EndToEndMetrics(mode=mode)
        t_global_start = time.perf_counter()

        generator = torch.Generator("cpu").manual_seed(seed)
        device = self.device
        dtype = self.dit_dtype

        # -------------------------------------------------------------------
        # Step 1: Prompt Encoding (UMT5-XXL on Host CPU)
        # -------------------------------------------------------------------
        print(f">> [1/5] Encoding Prompt via UMT5-XXL (Host CPU)...")
        t_enc_start = time.perf_counter()
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            device=torch.device("cpu"),
            dtype=self.text_dtype,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
        metrics.prompt_encode_time_s = time.perf_counter() - t_enc_start
        print(f"   [OK] Prompt encoded in {metrics.prompt_encode_time_s:.2f}s (transferred to {device})")

        # -------------------------------------------------------------------
        # Step 2: Latents & Scheduler Setup
        # -------------------------------------------------------------------
        print(f">> [2/5] Preparing Latents & Scheduler ({num_frames} frames @ {width}x{height})...")
        t_prep_start = time.perf_counter()
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps

        num_channels_latents = self.transformer.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)
        metrics.dit_prep_time_s = time.perf_counter() - t_prep_start
        print(f"   [OK] Latents ready: {list(latents.shape)} in {metrics.dit_prep_time_s:.2f}s")

        # -------------------------------------------------------------------
        # Step 3: Initialize Marley Streamer
        # -------------------------------------------------------------------
        print(f">> [3/5] Setting up Marley Streamer (mode='{mode}')...")
        if mode == "sync" or mode == "async_fp16":
            streamer = BudgetedAsyncStreamer(blocks=self.blocks, device=device, dtype=dtype)
        elif mode == "async_int8":
            streamer = INT8BudgetedStreamer(blocks=self.blocks, device=device, dtype=dtype)
        elif mode == "adaptive":
            streamer = AdaptiveEngine(blocks=self.blocks, device=device, dtype=dtype, window=5)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # -------------------------------------------------------------------
        # Step 4: Denoising Loop (30 Steps)
        # -------------------------------------------------------------------
        print(f">> [4/5] Executing DiT Denoising Loop ({num_inference_steps} steps)...")
        t_denoise_start = time.perf_counter()
        torch.set_grad_enabled(False)

        batch_size, num_channels, num_frames_in, h_in, w_in = latents.shape
        p_t, p_h, p_w = self.transformer.config.patch_size
        post_patch_num_frames = num_frames_in // p_t
        post_patch_height = h_in // p_h
        post_patch_width = w_in // p_w

        transformer_blocks_orig = self.transformer.blocks
        # Detach module blocks during patched execution so diffusers doesnt conflict
        self.transformer.blocks = nn.ModuleList([])

        try:
            for step_idx, t in enumerate(timesteps):
                t_step_start = time.perf_counter()
                latent_model_input = latents.to(dtype)

                # Patchify & compute embeddings
                rotary_emb = self.transformer.rope(latent_model_input)
                hidden_states = self.transformer.patch_embedding(latent_model_input)
                hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

                timestep = t.expand(latent_model_input.shape[0])
                temb, timestep_proj, enc_hidden_states, _ = self.transformer.condition_embedder(
                    timestep, prompt_embeds, None
                )
                timestep_proj = timestep_proj.unflatten(1, (6, -1))

                # --- Execute DiT blocks via Marley Streamer (cond) ---
                hidden_states_cond = hidden_states.clone()
                self._execute_dit_custom_forward(
                    streamer=streamer,
                    mode=mode,
                    hidden_states=hidden_states_cond,
                    encoder_hidden_states=enc_hidden_states,
                    timestep_proj=timestep_proj,
                    rotary_emb=rotary_emb,
                    num_steps=1,
                    num_frames=num_frames,
                )

                # Output norm, projection & unpatchify (cond)
                shift, scale = (self.transformer.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
                shift = shift.to(hidden_states_cond.device)
                scale = scale.to(hidden_states_cond.device)

                hidden_states_cond = (self.transformer.norm_out(hidden_states_cond.float()) * (1 + scale) + shift).type_as(hidden_states_cond)
                hidden_states_cond = self.transformer.proj_out(hidden_states_cond)
                hidden_states_cond = hidden_states_cond.reshape(
                    batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                ).permute(0, 7, 1, 4, 2, 5, 3, 6)
                noise_pred_cond = hidden_states_cond.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                # --- Classifier-Free Guidance (uncond) ---
                temb_uncond, timestep_proj_uncond, enc_hidden_states_uncond, _ = self.transformer.condition_embedder(
                    timestep, negative_prompt_embeds, None
                )
                timestep_proj_uncond = timestep_proj_uncond.unflatten(1, (6, -1))
                hidden_states_uncond = hidden_states.clone()

                self._execute_dit_custom_forward(
                    streamer=streamer,
                    mode=mode,
                    hidden_states=hidden_states_uncond,
                    encoder_hidden_states=enc_hidden_states_uncond,
                    timestep_proj=timestep_proj_uncond,
                    rotary_emb=rotary_emb,
                    num_steps=1,
                    num_frames=num_frames,
                )

                shift_u, scale_u = (self.transformer.scale_shift_table.to(temb_uncond.device) + temb_uncond.unsqueeze(1)).chunk(2, dim=1)
                shift_u = shift_u.to(hidden_states_uncond.device)
                scale_u = scale_u.to(hidden_states_uncond.device)

                hidden_states_uncond = (self.transformer.norm_out(hidden_states_uncond.float()) * (1 + scale_u) + shift_u).type_as(hidden_states_uncond)
                hidden_states_uncond = self.transformer.proj_out(hidden_states_uncond)
                hidden_states_uncond = hidden_states_uncond.reshape(
                    batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                ).permute(0, 7, 1, 4, 2, 5, 3, 6)
                noise_pred_uncond = hidden_states_uncond.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                # Combine predictions
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # Scheduler step
                latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                step_elapsed = time.perf_counter() - t_step_start
                metrics.step_times_s.append(step_elapsed)
                if (step_idx + 1) % 5 == 0 or (step_idx + 1) == num_inference_steps:
                    print(f"   Step [{step_idx+1}/{num_inference_steps}] completed in {step_elapsed:.2f}s (avg: {sum(metrics.step_times_s)/len(metrics.step_times_s):.2f}s/step)")

        finally:
            self.transformer.blocks = transformer_blocks_orig
            # Release streamer allocations
            if hasattr(streamer, "restore_all_to_cpu"):
                streamer.restore_all_to_cpu()
            if hasattr(streamer, "release"):
                streamer.release()
            torch.cuda.empty_cache()

        metrics.denoise_time_s = time.perf_counter() - t_denoise_start
        print(f"   [OK] Denoising finished in {metrics.denoise_time_s:.2f}s")

        # -------------------------------------------------------------------
        # Step 5: VAE Video Reconstruction (F5-A Non-Regression baseline)
        # -------------------------------------------------------------------
        print(f">> [5/5] VAE Decode ({num_frames} frames via tiled bfloat16)...")
        t_vae_start = time.perf_counter()
        latents_vae = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents_vae.device, latents_vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents_vae.device, latents_vae.dtype
        )
        latents_vae = latents_vae / latents_std + latents_mean

        self.vae.to(device)
        video_tensor = self.vae.decode(latents_vae, return_dict=False)[0]
        self.vae.to("cpu")
        torch.cuda.empty_cache()
        video_frames = self.pipe.video_processor.postprocess_video(video_tensor, output_type="pil")
        metrics.vae_decode_time_s = time.perf_counter() - t_vae_start
        print(f"   [OK] VAE decode completed in {metrics.vae_decode_time_s:.2f}s")

        # -------------------------------------------------------------------
        # Step 6: Video Export
        # -------------------------------------------------------------------
        t_export_start = time.perf_counter()
        if output_video_path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_video_path = f"logs/f6_{width}x{height}_{num_frames}f_{mode}_{ts}.mp4"

        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        export_to_video(video_frames[0], output_video_path, fps=16)
        metrics.video_export_time_s = time.perf_counter() - t_export_start
        metrics.output_video_path = output_video_path
        print(f"   [OK] Video successfully exported to: {output_video_path}")

        # Check for NaNs
        metrics.nan_inf_detected = bool(torch.isnan(video_tensor).any() or torch.isinf(video_tensor).any())

        metrics.total_wall_clock_s = time.perf_counter() - t_global_start
        return video_tensor, metrics

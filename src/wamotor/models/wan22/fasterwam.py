from collections.abc import Sequence
from typing import Any

import torch

from .helpers.loader import load_wan22_ti2v_5b_components
from .jointwam import JointWAM
from .sparse_action_dit import DEFAULT_SPARSE_MOT_CONDITION_LAYERS, SparseActionDiT
from .sparse_mot import SparseMoT
from .speculative import speculative_denoise_action


class FasterWAM(JointWAM):
    """Final FasterWAM system built from SparseMoT and SparseActionDiT."""

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: int | None = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        condition_layers: Sequence[int] | None = None,
        mot_action_video_attention_layers: Sequence[int] | None = None,
        video_kv_fusion: str | None = None,
        video_kv_fusion_init: str = "current_layer",
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FasterWAM.")
        if "text_dim" not in video_dit_config:
            raise ValueError(
                "`video_dit_config['text_dim']` is required for FasterWAM."
            )
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "FasterWAM requires `video_dit_config['action_conditioned']=false`."
            )
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for FasterWAM.")

        if condition_layers is None:
            condition_layers = action_dit_config.get(
                "condition_layers", DEFAULT_SPARSE_MOT_CONDITION_LAYERS
            )
        normalized_condition_layers = SparseActionDiT._normalize_condition_layers(
            condition_layers,
            int(action_dit_config["num_layers"]),
        )
        action_dit_config = dict(action_dit_config)
        action_dit_config["condition_layers"] = list(normalized_condition_layers)

        if (
            mot_action_video_attention_layers is not None
            and tuple(mot_action_video_attention_layers) != normalized_condition_layers
        ):
            raise ValueError(
                "`mot_action_video_attention_layers` must be omitted or equal to `condition_layers` "
                "for FasterWAM."
            )

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = SparseActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )

        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("SparseActionDiT `num_layers` must match video expert.")
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError(
                "SparseActionDiT condition-layer `num_heads` must match video expert."
            )
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("SparseActionDiT `attn_head_dim` must match video expert.")
        if int(action_expert.num_heads) * int(action_expert.attn_head_dim) != int(
            video_expert.hidden_dim
        ):
            raise ValueError(
                "SparseActionDiT condition-layer attention dim must equal video attention dim: "
                f"{action_expert.num_heads} * {action_expert.attn_head_dim} != {video_expert.hidden_dim}."
            )

        mot = SparseMoT(
            mixtures={"video": video_expert, "action": action_expert},
            condition_layers=normalized_condition_layers,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            video_kv_fusion=video_kv_fusion,
            video_kv_fusion_init=video_kv_fusion_init,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            mot_action_video_attention_layers=normalized_condition_layers,
        )
        model.condition_layers = normalized_condition_layers
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN"
                if skip_dit_load_from_pretrain
                else action_dit_pretrained_path
            ),
            "condition_layers": list(normalized_condition_layers),
            "video_kv_fusion": video_kv_fusion,
            "video_kv_fusion_init": video_kv_fusion_init,
        }
        return model

    @torch.no_grad()
    def spec_action(
        self,
        prompt: str | None,
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        proprio: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        negative_prompt: str | None = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        w: int = 10,
        epsilon: float = 1e-2,
    ) -> dict[str, Any]:
        """Denoise actions with batched speculative Euler steps.

        At an accepted state, one velocity is used to draft the next ``w``
        states. All draft states are validated in one batched model call. A
        window contributes only its longest valid prefix; after a rejection,
        the first rejected transition is evaluated with the last exact velocity.
        The first transition always uses ordinary Euler
        integration and speculation starts from the second model timestep.
        """
        del negative_prompt, text_cfg_scale
        self.eval()

        if isinstance(w, bool) or not isinstance(w, int) or w <= 0:
            raise ValueError(f"`w` must be a positive integer, got {w!r}")
        epsilon = float(epsilon)
        if epsilon < 0.0 or epsilon != epsilon or epsilon == float("inf"):
            raise ValueError(
                f"`epsilon` must be a finite non-negative float, got {epsilon!r}"
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (
            input_image.ndim != 4
            or input_image.shape[0] != 1
            or input_image.shape[1] != 3
        ):
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(
                    f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        action_generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled
        )
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError(
                "`prompt` and `context/context_mask` are mutually exclusive."
            )
        if not use_prompt and not use_context:
            raise ValueError(
                "Either `prompt` or both `context/context_mask` must be provided."
            )

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError(
                    "`context` and `context_mask` must be both provided together."
                )
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            context_mask = context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, _ = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        timestep_video = (
            infer_timesteps_video[0]
            .unsqueeze(0)
            .to(dtype=latents_video.dtype, device=self.device)
        )
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_masks = self._build_mot_attention_masks(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_attention_mask = self._build_video_attention_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
        )

        infer_timesteps_action, infer_deltas_action = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        )
        latents_action = speculative_denoise_action(
            model=self,
            latents_action=latents_action,
            infer_timesteps_action=infer_timesteps_action,
            infer_deltas_action=infer_deltas_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_masks=attention_masks,
            video_seq_len=video_seq_len,
            w=w,
            epsilon=epsilon,
        )

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

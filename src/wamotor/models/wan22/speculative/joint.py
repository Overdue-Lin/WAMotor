from __future__ import annotations

from math import isfinite
from typing import Any

import torch

_COMPILED_PREDICTORS: dict[int, Any] = {}


def _get_predictor(model: Any):
    predictor = _COMPILED_PREDICTORS.get(id(model))
    if predictor is not None:
        return predictor
    if getattr(model, "device", None) is not None and model.device.type == "cuda":
        try:
            predictor = torch.compile(
                model._predict_joint_noise,
                mode="reduce-overhead",
                fullgraph=False,
            )
        except (AttributeError, RuntimeError):
            predictor = model._predict_joint_noise
    else:
        predictor = model._predict_joint_noise
    _COMPILED_PREDICTORS[id(model)] = predictor
    return predictor


@torch.no_grad()
def speculative_denoise_joint(
    model: Any,
    latents_video: torch.Tensor,
    latents_action: torch.Tensor,
    first_frame_latents: torch.Tensor,
    infer_timesteps_video: torch.Tensor,
    infer_deltas_video: torch.Tensor,
    infer_timesteps_action: torch.Tensor,
    infer_deltas_action: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    fuse_vae_embedding_in_latents: bool,
    gt_action: torch.Tensor | None = None,
    w: int = 10,
    epsilon: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Speculatively integrate the coupled JointWAM video/action states."""
    if isinstance(w, bool) or not isinstance(w, int) or w <= 0:
        raise ValueError(f"`w` must be a positive integer, got {w!r}")
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"`epsilon` must be finite and non-negative, got {epsilon!r}"
        ) from exc
    if epsilon < 0.0 or not isfinite(epsilon):
        raise ValueError(f"`epsilon` must be finite and non-negative, got {epsilon!r}")

    num_steps = min(
        infer_timesteps_video.shape[0],
        infer_timesteps_action.shape[0],
        infer_deltas_video.shape[0],
        infer_deltas_action.shape[0],
    )
    if num_steps == 0:
        return latents_video, latents_action
    batch_size = latents_video.shape[0]
    abs_deltas = infer_deltas_action.float().abs()
    tiny = torch.finfo(torch.float32).eps
    reference_delta = abs_deltas.median().clamp_min(tiny)
    thresholds = epsilon * (reference_delta / abs_deltas.clamp_min(tiny)) ** 2
    predict_joint_noise = _get_predictor(model)

    def predict(
        video: torch.Tensor,
        action: torch.Tensor,
        timestep_indices: torch.Tensor,
    ):
        repeats = timestep_indices.shape[0]
        timestep_video = infer_timesteps_video[timestep_indices].repeat_interleave(
            batch_size
        )
        timestep_action = infer_timesteps_action[timestep_indices].repeat_interleave(
            batch_size
        )
        pred_context = context.expand(repeats * batch_size, -1, -1)
        pred_mask = context_mask.expand(repeats * batch_size, -1)
        pred_gt_action = (
            None
            if gt_action is None
            else gt_action.expand(repeats * batch_size, -1, -1)
        )
        return predict_joint_noise(
            latents_video=video,
            latents_action=action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=pred_context,
            context_mask=pred_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            gt_action=pred_gt_action,
        )

    m = 0
    video_velocity: torch.Tensor | None = None
    action_velocity: torch.Tensor | None = None
    # Always take the first transition with ordinary Euler integration.
    video_velocity, action_velocity = predict(
        latents_video,
        latents_action,
        torch.zeros(1, dtype=torch.long, device=latents_video.device),
    )
    latents_video = model.infer_video_scheduler.step(
        video_velocity, infer_deltas_video[0], latents_video
    )
    latents_action = model.infer_action_scheduler.step(
        action_velocity, infer_deltas_action[0], latents_action
    )
    latents_video[:, :, 0:1] = first_frame_latents
    m = 1
    video_velocity = action_velocity = None

    while m < num_steps:
        if video_velocity is None:
            video_velocity, action_velocity = predict(
                latents_video,
                latents_action,
                torch.tensor([m], dtype=torch.long, device=latents_video.device),
            )
        if video_velocity is None or action_velocity is None:
            raise RuntimeError("Joint speculative state lost its velocity.")
        if m == num_steps - 1:
            latents_video = model.infer_video_scheduler.step(
                video_velocity, infer_deltas_video[m], latents_video
            )
            latents_action = model.infer_action_scheduler.step(
                action_velocity, infer_deltas_action[m], latents_action
            )
            latents_video[:, :, 0:1] = first_frame_latents
            return latents_video, latents_action

        end = min(m + w, num_steps - 1)
        count = end - m
        video_scale = torch.cumsum(infer_deltas_video[m:end].float(), dim=0)
        action_scale = torch.cumsum(infer_deltas_action[m:end].float(), dim=0)
        video_draft = (
            latents_video.unsqueeze(0)
            + video_scale.to(latents_video.dtype).view(count, 1, 1, 1, 1, 1)
            * video_velocity.unsqueeze(0)
        ).reshape(count * batch_size, *latents_video.shape[1:])
        action_draft = (
            latents_action.unsqueeze(0)
            + action_scale.to(latents_action.dtype).view(count, 1, 1, 1)
            * action_velocity.unsqueeze(0)
        ).reshape(count * batch_size, *latents_action.shape[1:])
        video_draft[:, :, 0:1] = first_frame_latents.expand(
            count * batch_size, -1, -1, -1, -1
        )

        validated_video, validated_action = predict(
            video_draft,
            action_draft,
            torch.arange(m + 1, end + 1, dtype=torch.long, device=latents_video.device),
        )
        action_mse = (
            validated_action.float()
            .reshape(count, batch_size, *latents_action.shape[1:])
            .sub(action_velocity.float().unsqueeze(0))
            .square()
            .mean(dim=-1)
        )
        threshold = thresholds[m + 1 : end + 1]
        valid = action_mse.lt(threshold.view(count, 1, 1)).all(dim=-1).all(dim=-1)
        invalid = torch.where(~valid)[0]
        if invalid.numel() == 0:
            accepted = count
        else:
            try:
                accepted = int(invalid[0].item())
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Invalid speculative acceptance index.") from exc
        if accepted == count:
            latents_video = video_draft[-batch_size:]
            latents_action = action_draft[-batch_size:]
            video_velocity = validated_video[-batch_size:]
            action_velocity = validated_action[-batch_size:]
            m = end
            continue

        if accepted:
            latents_video = video_draft[
                (accepted - 1) * batch_size : accepted * batch_size
            ]
            latents_action = action_draft[
                (accepted - 1) * batch_size : accepted * batch_size
            ]
            fallback_video = validated_video[
                (accepted - 1) * batch_size : accepted * batch_size
            ]
            fallback_action = validated_action[
                (accepted - 1) * batch_size : accepted * batch_size
            ]
        else:
            fallback_video = video_velocity
            fallback_action = action_velocity
        step = m + accepted
        latents_video = model.infer_video_scheduler.step(
            fallback_video, infer_deltas_video[step], latents_video
        )
        latents_action = model.infer_action_scheduler.step(
            fallback_action, infer_deltas_action[step], latents_action
        )
        latents_video[:, :, 0:1] = first_frame_latents
        m = step + 1
        video_velocity = validated_video[:batch_size] if accepted == 0 else None
        action_velocity = validated_action[:batch_size] if accepted == 0 else None

    return latents_video, latents_action

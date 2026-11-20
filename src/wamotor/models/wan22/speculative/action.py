from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

import torch

_COMPILED_PREDICTORS: dict[int, Any] = {}


def _get_predictor(model: Any):
    """Compile only the repeated speculative action model call."""
    key = id(model)
    predictor = _COMPILED_PREDICTORS.get(key)
    if predictor is not None:
        return predictor
    if getattr(model, "device", None) is not None and model.device.type == "cuda":
        try:
            predictor = torch.compile(
                model._predict_action_noise_with_cache,
                mode="reduce-overhead",
                fullgraph=False,
            )
        except (AttributeError, RuntimeError):
            predictor = model._predict_action_noise_with_cache
    else:
        predictor = model._predict_action_noise_with_cache
    _COMPILED_PREDICTORS[key] = predictor
    return predictor


@torch.no_grad()
def speculative_denoise_action(
    model: Any,
    latents_action: torch.Tensor,
    infer_timesteps_action: torch.Tensor,
    infer_deltas_action: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    video_kv_cache: Sequence[dict[str, torch.Tensor] | None],
    attention_masks: torch.Tensor | Sequence[torch.Tensor],
    video_seq_len: int,
    w: int = 10,
    epsilon: float = 1e-2,
) -> torch.Tensor:
    """Run batched speculative Euler integration for one action block."""
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

    num_steps = infer_timesteps_action.shape[0]
    if num_steps == 0:
        return latents_action
    batch_size = latents_action.shape[0]
    abs_deltas = infer_deltas_action.float().abs()
    tiny = torch.finfo(torch.float32).eps
    reference_delta = abs_deltas.median().clamp_min(tiny)
    thresholds = epsilon * (reference_delta / abs_deltas.clamp_min(tiny)) ** 2
    predict_action_noise = _get_predictor(model)

    def predict(latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        repeats = timesteps.shape[0]
        if repeats == 1:
            cache = video_kv_cache
        else:
            cache = [
                None
                if layer is None
                else {
                    key: value.expand(repeats * batch_size, *value.shape[1:])
                    for key, value in layer.items()
                }
                for layer in video_kv_cache
            ]
        return predict_action_noise(
            latents_action=latents,
            timestep_action=timesteps.repeat_interleave(batch_size),
            context=context.expand(repeats * batch_size, -1, -1),
            context_mask=context_mask.expand(repeats * batch_size, -1),
            video_kv_cache=cache,
            attention_mask=attention_masks,
            video_seq_len=video_seq_len,
        )

    m = 0
    velocity = None
    # Always take the first transition with ordinary Euler integration.
    velocity = predict(latents_action, infer_timesteps_action[:1])
    latents_action = model.infer_action_scheduler.step(
        velocity, infer_deltas_action[0], latents_action
    )
    m = 1
    velocity = None

    while m < num_steps:
        if velocity is None:
            velocity = predict(latents_action, infer_timesteps_action[m : m + 1])
        if m == num_steps - 1:
            return model.infer_action_scheduler.step(
                velocity, infer_deltas_action[m], latents_action
            )

        end = min(m + w, num_steps - 1)
        candidate_timesteps = infer_timesteps_action[m + 1 : end + 1]
        count = candidate_timesteps.shape[0]
        draft = (
            latents_action.unsqueeze(0)
            + torch.cumsum(infer_deltas_action[m:end].float(), dim=0)
            .to(dtype=latents_action.dtype)
            .view(count, 1, 1, 1)
            * velocity.unsqueeze(0)
        ).reshape(count * batch_size, *latents_action.shape[1:])
        validated = predict(draft, candidate_timesteps)
        mse = (
            validated.float()
            .reshape(count, batch_size, *latents_action.shape[1:])
            .sub(velocity.float().unsqueeze(0))
            .square()
            .mean(dim=-1)
        )
        threshold = thresholds[m + 1 : end + 1]
        valid = mse.lt(threshold.view(count, 1, 1)).all(dim=-1).all(dim=-1)
        invalid = torch.where(~valid)[0]
        accepted = count if invalid.numel() == 0 else invalid[0].item()
        if accepted == count:
            latents_action = draft[-batch_size:]
            velocity = validated[-batch_size:]
            m = end
            continue
        if accepted:
            latents_action = draft[(accepted - 1) * batch_size : accepted * batch_size]
            fallback_velocity = validated[
                (accepted - 1) * batch_size : accepted * batch_size
            ]
        else:
            fallback_velocity = velocity
        step = m + accepted
        latents_action = model.infer_action_scheduler.step(
            fallback_velocity, infer_deltas_action[step], latents_action
        )
        m = step + 1
        velocity = validated[:batch_size] if accepted == 0 else None

    return latents_action

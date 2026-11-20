from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
for path in (ROOT, ROOT / "src", ROOT / "experiments" / "libero"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Loaded in main so --help remains usable without the LIBERO environment.
torch: Any = None
DEFAULT_PROMPT: str = ""


def get_libero_dummy_action() -> Any:
    raise RuntimeError("LIBERO dependencies have not been loaded.")


PHASES = (
    "libero_input_ms",
    "vae_encode_ms",
    "visual_model_ms",
    "action_chunk_ms",
    "model_overall_ms",
    "overall_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark per-action-chunk inference latency on LIBERO observations."
    )
    parser.add_argument(
        "--model", required=True, help="Result label, e.g. fastwam or fasterwam."
    )
    parser.add_argument("--task", required=True, help="Hydra task config name.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument(
        "--output",
        default=str(ROOT / "evaluate_results" / "latency"),
    )
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=30)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument(
        "--action-infer-mode",
        default="auto",
        choices=[
            "auto",
            "joint_rollout",
            "one_pass_future_cache",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "no"], default="bf16")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--speculative",
        action="store_true",
        help="Use model.spec_action; fail if the model does not implement it.",
    )
    parser.add_argument("--spec-w", type=int, default=10)
    parser.add_argument("--spec-epsilon", type=float, default=5e-2)
    return parser.parse_args()


def model_dtype(name: str) -> Any:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[name]


def load_cfg(args: argparse.Namespace):
    hydra = __import__("hydra")
    compose = hydra.compose
    initialize_config_dir = hydra.initialize_config_dir
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(
            config_name="sim_libero",
            overrides=[
                f"task={args.task}",
                f"EVALUATION.task_suite_name={args.suite}",
                f"EVALUATION.task_id={args.task_id}",
                f"EVALUATION.num_inference_steps={args.inference_steps}",
                f"EVALUATION.action_horizon={args.action_horizon}",
                f"EVALUATION.device={args.device}",
                f"EVALUATION.dataset_stats_path={args.dataset_stats}",
                f"seed={args.seed}",
            ],
        )


def load_dependencies():
    torch = __import__("torch")
    hydra_utils = __import__("hydra.utils", fromlist=["instantiate"])
    libero = __import__("libero.libero", fromlist=["benchmark"])
    eval_single = __import__(
        "experiments.libero.eval_libero_single",
        fromlist=["_load_model_checkpoint", "_obs_to_model_input"],
    )
    libero_utils = __import__(
        "experiments.libero.libero_utils",
        fromlist=["LIBERO_ENV_RESOLUTION", "get_libero_dummy_action", "get_libero_env"],
    )
    dataset_utils = __import__(
        "wamotor.processing.utils.normalizer",
        fromlist=["load_dataset_stats_from_json"],
    )
    prompts = __import__("wamotor.prompts", fromlist=["DEFAULT_PROMPT"])
    return {
        "torch": torch,
        "instantiate": hydra_utils.instantiate,
        "benchmark": libero.benchmark,
        "load_model_checkpoint": eval_single._load_model_checkpoint,
        "obs_to_model_input": eval_single._obs_to_model_input,
        "libero_env_resolution": libero_utils.LIBERO_ENV_RESOLUTION,
        "dummy_action": libero_utils.get_libero_dummy_action,
        "get_libero_env": libero_utils.get_libero_env,
        "load_dataset_stats": dataset_utils.load_dataset_stats_from_json,
        "default_prompt": prompts.DEFAULT_PROMPT,
    }


class PhaseProfiler:
    """Records CUDA-event or synchronized CPU wall time for instrumented methods."""

    def __init__(self, device: Any):
        self.device = device
        self.cuda = device.type == "cuda"
        self.enabled = False
        self.active: set[str] = set()
        self.cpu_ms: dict[str, float] = {
            "vae_encode_ms": 0.0,
            "visual_model_ms": 0.0,
        }

    def reset(self) -> None:
        for key in self.cpu_ms:
            self.cpu_ms[key] = 0.0

    @contextmanager
    def record(self, phase: str):
        if not self.enabled or phase in self.active:
            yield
            return
        self.active.add(phase)
        try:
            if self.cuda:
                torch.cuda.synchronize(self.device)
            start_time = time.perf_counter()
            try:
                yield
            finally:
                if self.cuda:
                    torch.cuda.synchronize(self.device)
                self.cpu_ms[phase] += (time.perf_counter() - start_time) * 1000.0
        finally:
            self.active.remove(phase)

    def wrap(self, owner: Any, method_name: str, phase: str) -> None:
        original = getattr(owner, method_name, None)
        if original is None:
            return

        def wrapped(*args, **kwargs):
            with self.record(phase):
                return original(*args, **kwargs)

        setattr(owner, method_name, wrapped)

    def values(self) -> dict[str, float]:
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return dict(self.cpu_ms)


def synchronize(device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: Any, fn: Callable[[], Any]) -> tuple[Any, float]:
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0


def select_inference(model: Any, mode: str, cfg, speculative: bool = False):
    if mode == "auto":
        mode = str(cfg.get("default_action_infer_mode", "joint_rollout"))
    if speculative:
        fn = getattr(model, "spec_action", None)
        if fn is None:
            raise ValueError(
                "--speculative requires model method `spec_action(...)`; "
                "refusing to fall back to infer_action."
            )
        return mode, fn, inspect.signature(fn)
    if mode == "one_pass_future_cache":
        fn = getattr(model, "infer_action_one_pass_future_cache", None)
        if fn is None:
            raise ValueError(
                "action_infer_mode=one_pass_future_cache requires "
                "model.infer_action_one_pass_future_cache()."
            )
    else:
        fn = model.infer_action
    return mode, fn, inspect.signature(fn)


class _SpeculationTracker:
    """Observe actual accepted prefixes in a separate, untimed sampler replay."""

    def __init__(self, epsilon: float):
        try:
            self.epsilon = float(epsilon)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid speculative epsilon: {epsilon!r}") from exc
        self.accepted_steps: list[int] = []

    def reset(self) -> None:
        self.accepted_steps.clear()

    def collect(self, fn: Callable[[], Any]) -> None:
        """Read actual sampler decisions during an untimed deterministic replay."""
        from wamotor.models.wan22.speculative import (
            speculative_denoise_action,
            speculative_denoise_joint,
        )

        targets = {
            inspect.unwrap(speculative_denoise_action).__code__,
            inspect.unwrap(speculative_denoise_joint).__code__,
        }
        seen = set()
        self.reset()

        def trace(frame, event, arg):
            if frame.f_code not in targets:
                return None
            if event == "line":
                local = frame.f_locals
                # At this point both samplers have finalized the prefix count.
                if "accepted" in local and "count" in local:
                    line = inspect.getframeinfo(frame, context=1).code_context
                    if line and line[0].strip() == "if accepted == count:":
                        key = (id(frame), local["m"])
                        if key not in seen:
                            seen.add(key)
                            self.accepted_steps.append(local["accepted"])
            return trace

        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            fn()
        finally:
            sys.settrace(previous)

    def summary(self) -> dict[str, Any]:
        return {
            "average_spec_diffusion_steps": (
                statistics.fmean(self.accepted_steps) if self.accepted_steps else None
            ),
            "accepted_spec_diffusion_steps": list(self.accepted_steps),
            "spec_rounds": len(self.accepted_steps),
        }


def make_infer_kwargs(
    args: argparse.Namespace,
    cfg,
    signature: inspect.Signature,
    mode: str,
    image: Any,
    proprio: Any,
    description: str,
    iteration: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prompt": DEFAULT_PROMPT.format(task=description),
        "input_image": image,
        "action_horizon": args.action_horizon,
        "proprio": proprio,
        "negative_prompt": "",
        "text_cfg_scale": 1.0,
        "num_inference_steps": args.inference_steps,
        "sigma_shift": None,
        "seed": args.seed + iteration,
        "rand_device": "cpu",
        "tiled": False,
    }
    if args.speculative or "num_video_frames" in signature.parameters:
        kwargs["num_video_frames"] = (
            cfg.data.train.num_frames - 1
        ) // cfg.data.train.action_video_freq_ratio + 1
    if "action_infer_mode" in signature.parameters:
        kwargs["action_infer_mode"] = mode
    if args.speculative:
        if "w" in signature.parameters:
            kwargs["w"] = args.spec_w
        if "epsilon" in signature.parameters:
            kwargs["epsilon"] = args.spec_epsilon
    return kwargs


def prepare_observation(env: Any, initial_state: Any, num_steps_wait: int):
    env.reset()
    obs = env.set_init_state(initial_state)
    done = False
    for _ in range(num_steps_wait):
        obs, _, done, _ = env.step(get_libero_dummy_action())
        if done:
            break
    if done:
        raise RuntimeError(
            "LIBERO task completed during no-op observation preparation."
        )
    return obs


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for phase in PHASES:
        values = [row[phase] for row in rows]
        summary[phase] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def write_results(
    output_dir: Path,
    args: argparse.Namespace,
    mode: str,
    rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = summarize(rows)
    metadata = vars(args).copy()
    metadata["action_infer_mode_resolved"] = mode
    metadata["speculative"] = bool(args.speculative)
    spec_values = [
        row["average_spec_diffusion_steps"]
        for row in rows
        if row["average_spec_diffusion_steps"] is not None
    ]
    accepted_steps_per_inference = [
        row["accepted_spec_diffusion_steps"] for row in rows
    ]
    spec_metrics = {
        "average_spec_diffusion_steps": (
            statistics.fmean(spec_values) if spec_values else None
        ),
        "accepted_spec_diffusion_steps": accepted_steps_per_inference,
        "measured_inference_calls": len(spec_values),
    }
    metadata["spec_metrics"] = spec_metrics
    metadata["speculative_validation"] = "action_mse_only"
    metadata["phase_definition"] = {
        "libero_input_ms": "LIBERO observation image/proprio preprocessing",
        "vae_encode_ms": "input-image VAE encoding inside model inference",
        "visual_model_ms": "video pre_dit and video K/V prefill",
        "action_chunk_ms": "model_overall_ms - vae_encode_ms - visual_model_ms",
        "model_overall_ms": "complete model action inference call",
        "overall_ms": "libero_input_ms + model_overall_ms",
    }
    (output_dir / "config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    with (output_dir / "iterations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "config": metadata,
        "iterations": rows,
        "summary_ms": summary,
        "spec_metrics": spec_metrics,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "mean_ms", "std_ms", "min_ms", "max_ms"])
        for phase in PHASES:
            stats = summary[phase]
            writer.writerow(
                [phase, stats["mean"], stats["std"], stats["min"], stats["max"]]
            )


def main() -> None:
    args = parse_args()
    dependencies = load_dependencies()
    torch_module = dependencies["torch"]
    globals()["torch"] = torch_module
    instantiate = dependencies["instantiate"]
    benchmark = dependencies["benchmark"]
    _load_model_checkpoint = dependencies["load_model_checkpoint"]
    _obs_to_model_input = dependencies["obs_to_model_input"]
    libero_env_resolution = dependencies["libero_env_resolution"]
    dummy_action = dependencies["dummy_action"]
    get_libero_env = dependencies["get_libero_env"]
    load_dataset_stats = dependencies["load_dataset_stats"]
    default_prompt = dependencies["default_prompt"]
    globals()["DEFAULT_PROMPT"] = default_prompt
    globals()["get_libero_dummy_action"] = dummy_action
    if args.warmup_iters < 0 or args.benchmark_iters <= 0:
        raise ValueError("warmup-iters must be >= 0 and benchmark-iters must be > 0.")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    dataset_stats = Path(args.dataset_stats).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not dataset_stats.is_file():
        raise FileNotFoundError(dataset_stats)

    cfg = load_cfg(args)
    device = torch.device(args.device)
    dtype = model_dtype(args.dtype)
    model = instantiate(cfg.model, model_dtype=dtype, device=str(device))
    _load_model_checkpoint(model, str(checkpoint))
    model.eval()

    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats(str(dataset_stats)))

    mode, infer_fn, infer_signature = select_inference(
        model, args.action_infer_mode, cfg, speculative=args.speculative
    )
    if args.spec_w <= 0:
        raise ValueError("--spec-w must be positive.")
    if args.spec_epsilon < 0:
        raise ValueError("--spec-epsilon must be non-negative.")
    spec_tracker = _SpeculationTracker(args.spec_epsilon) if args.speculative else None
    profiler = PhaseProfiler(device)
    profiler.wrap(model, "_encode_input_image_latents_tensor", "vae_encode_ms")
    profiler.wrap(model.video_expert, "pre_dit", "visual_model_ms")
    profiler.wrap(model.mot, "prefill_video_cache", "visual_model_ms")

    suites = benchmark.get_benchmark_dict()
    if args.suite not in suites:
        raise KeyError(f"Unknown LIBERO suite {args.suite!r}.")
    suite = suites[args.suite]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    env, description = get_libero_env(task, libero_env_resolution, args.seed)
    video_h, video_w = cfg.data.train.video_size

    rows: list[dict[str, Any]] = []
    total_iters = args.warmup_iters + args.benchmark_iters
    try:
        for iteration in range(total_iters):
            obs = prepare_observation(
                env,
                initial_states[iteration % len(initial_states)],
                args.num_steps_wait,
            )
            (image, proprio, _), input_ms = timed_call(
                device,
                lambda obs=obs: _obs_to_model_input(
                    obs,
                    cfg,
                    processor,
                    width=video_w,
                    height=video_h,
                    device=str(device),
                    dtype=dtype,
                ),
            )
            infer_kwargs = make_infer_kwargs(
                args,
                cfg,
                infer_signature,
                mode,
                image,
                proprio,
                description,
                iteration,
            )
            profiler.reset()
            if spec_tracker is not None:
                spec_tracker.reset()
            profiler.enabled = True
            try:
                _, model_ms = timed_call(
                    device,
                    lambda infer_kwargs=infer_kwargs: infer_fn(**infer_kwargs),
                )
            finally:
                profiler.enabled = False
            phase_ms = profiler.values()
            if spec_tracker is not None:
                spec_tracker.collect(
                    lambda infer_kwargs=infer_kwargs: infer_fn(**infer_kwargs)
                )
            spec_stats = (
                spec_tracker.summary()
                if spec_tracker is not None
                else {
                    "average_spec_diffusion_steps": None,
                    "accepted_spec_diffusion_steps": [],
                    "spec_rounds": 0,
                }
            )
            if spec_stats["average_spec_diffusion_steps"] is not None:
                print(
                    "spec_avg="
                    f"{spec_stats['average_spec_diffusion_steps']:.3f} "
                    f"spec_rounds={spec_stats['spec_rounds']}"
                )
            action_ms = max(
                0.0,
                model_ms - phase_ms["vae_encode_ms"] - phase_ms["visual_model_ms"],
            )
            measured = iteration >= args.warmup_iters
            print(
                f"iter={iteration + 1}/{total_iters} measured={str(measured).lower()} "
                f"input={input_ms:.3f}ms vae={phase_ms['vae_encode_ms']:.3f}ms "
                f"visual={phase_ms['visual_model_ms']:.3f}ms "
                f"action={action_ms:.3f}ms model={model_ms:.3f}ms"
            )
            if measured:
                rows.append(
                    {
                        "iteration": iteration - args.warmup_iters,
                        "initial_state_index": iteration % len(initial_states),
                        "libero_input_ms": input_ms,
                        "vae_encode_ms": phase_ms["vae_encode_ms"],
                        "visual_model_ms": phase_ms["visual_model_ms"],
                        "action_chunk_ms": action_ms,
                        "model_overall_ms": model_ms,
                        "overall_ms": input_ms + model_ms,
                        "average_spec_diffusion_steps": spec_stats[
                            "average_spec_diffusion_steps"
                        ],
                        "accepted_spec_diffusion_steps": list(
                            spec_stats["accepted_spec_diffusion_steps"]
                        ),
                        "spec_rounds": spec_stats["spec_rounds"],
                    }
                )
    finally:
        env.close()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output).expanduser().resolve() / args.model / run_name
    write_results(output_dir, args, mode, rows)
    print(f"Latency benchmark finished: {output_dir}")
    for phase, stats in summarize(rows).items():
        print(f"{phase}: {stats['mean']:.3f} ms")
    if args.speculative:
        spec_values = [
            row["average_spec_diffusion_steps"]
            for row in rows
            if row["average_spec_diffusion_steps"] is not None
        ]
        average_spec = statistics.fmean(spec_values) if spec_values else None
        if average_spec is None:
            print("average_spec_diffusion_steps: N/A")
        else:
            print(f"average_spec_diffusion_steps: {average_spec:.3f}")


if __name__ == "__main__":
    main()

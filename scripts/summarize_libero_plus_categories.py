#!/usr/bin/env python3
"""Summarize LIBERO-Plus results by perturbation category."""

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

CATEGORIES = ("Camera", "Robot", "Lang", "Light", "Backg.", "Noise", "Layout")
VIEW_RE = re.compile(r"_view_(-?\d+)_(-?\d+)_(\d+)_(-?\d+)_(-?\d+)_initstate_(\d+)")


def load_task_map(path: Path) -> dict[str, list[str]]:
    text = path.read_text()
    return ast.literal_eval(text.split("libero_task_map =", 1)[1].strip())


def category(task_name: str) -> str:
    if "_language_" in task_name:
        return "Lang"
    if "_light_" in task_name:
        return "Light"
    if "_noise_" in task_name:
        return "Noise"
    if "_add_" in task_name or "_level" in task_name:
        return "Layout"
    if "_table_" in task_name or "_tb_" in task_name:
        return "Backg."

    match = VIEW_RE.search(task_name)
    if match:
        values = match.groups()
        # A canonical camera view plus a non-zero init state is the robot
        # initial-state perturbation; other view parameters are camera changes.
        if values[:5] == ("0", "0", "100", "0", "0") and values[5] != "0":
            return "Robot"
        return "Camera"
    raise ValueError(f"Cannot classify LIBERO-Plus task: {task_name}")


def summarize(input_dir: Path) -> dict:
    try:
        data = json.loads((input_dir / "summary.json").read_text())
        task_map_path = (
            Path(__file__).resolve().parents[1]
            / "third_party/LIBERO-plus/libero/libero/benchmark/libero_suite_task_map.py"
        )
        task_map = load_task_map(task_map_path)
        values = defaultdict(list)
        for key, result in data["task_results"].items():
            suite, task_id_text = key.rsplit("_", 1)
            task_id = int(task_id_text)
            task_name = task_map[suite][task_id]
            values[category(task_name)].append(
                {
                    "success_rate": float(result["successes"])
                    / result["total_episodes"]
                    * 100,
                    "episode_time": result.get("episode_time_mean"),
                }
            )
    except (
        OSError,
        KeyError,
        IndexError,
        ValueError,
        TypeError,
        ZeroDivisionError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            f"Invalid LIBERO-Plus result directory {input_dir}: {exc}"
        ) from exc

    stats = {}
    for name in CATEGORIES:
        samples = values[name]
        episode_times = [
            float(item["episode_time"])
            for item in samples
            if item["episode_time"] is not None
        ]
        stats[name] = {
            "success_rate": (
                sum(item["success_rate"] for item in samples) / len(samples)
                if samples
                else None
            ),
            "episode_time_mean": (
                sum(episode_times) / len(episode_times) if episode_times else None
            ),
            "tasks": len(samples),
        }
    all_values = [value for samples in values.values() for value in samples]
    all_episode_times = [
        float(item["episode_time"])
        for item in all_values
        if item["episode_time"] is not None
    ]
    stats["Average"] = {
        "success_rate": (
            sum(item["success_rate"] for item in all_values) / len(all_values)
            if all_values
            else None
        ),
        "episode_time_mean": (
            sum(all_episode_times) / len(all_episode_times)
            if all_episode_times
            else None
        ),
        "tasks": len(all_values),
    }
    return {"run_id": data.get("run_id"), "categories": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="LIBERO-Plus evaluation result directory",
    )
    args = parser.parse_args()
    output = args.input / "summary_additional.json"
    output.write_text(json.dumps(summarize(args.input), indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

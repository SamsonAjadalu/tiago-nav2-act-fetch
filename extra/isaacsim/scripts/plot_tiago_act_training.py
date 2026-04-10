#!/usr/bin/env python3
"""Plot TIAGo ACT training progress from saved LeRobot checkpoints.

This script is intentionally robust to the current logging setup:
- If scalar logs exist (TensorBoard / CSV / JSONL / W&B-style summaries), it plots them.
- If the run only saved checkpoints, it plots checkpoint step progression and model file size.

Default input directory: ``<TIAGO_ISAACSIM_DATA_ROOT>/tiago_act_policy`` (see ``tiago_repo_paths``).

Outputs a PNG and CSV summary into an `_analysis` folder under the run root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _try_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _iter_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def _read_checkpoint_series(root: Path) -> list[dict]:
    checkpoints_dir = root / "checkpoints"
    series: list[dict] = []
    if not checkpoints_dir.exists():
        return series

    for step_dir in sorted(checkpoints_dir.iterdir()):
        if not step_dir.is_dir():
            continue

        training_step_file = step_dir / "training_state" / "training_step.json"
        pretrained_dir = step_dir / "pretrained_model"
        model_file = pretrained_dir / "model.safetensors"
        config_file = pretrained_dir / "train_config.json"

        if not training_step_file.exists():
            continue

        try:
            training_data = _load_json(training_step_file)
        except Exception:
            continue

        step = _try_float(training_data.get("step"))
        if step is None:
            continue

        row = {
            "checkpoint_dir": step_dir.name,
            "step": step,
            "model_mb": model_file.stat().st_size / (1024 * 1024) if model_file.exists() else math.nan,
            "has_train_config": config_file.exists(),
        }
        series.append(row)

    series.sort(key=lambda row: row["step"])
    return series


def _parse_csv_metrics(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            return list(reader.fieldnames or []), []
        return list(reader.fieldnames or []), rows


def _discover_scalar_metrics(root: Path) -> dict[str, list[tuple[float, float]]]:
    """Best-effort parse for common log formats.

    Returns a mapping of metric name -> list of (step, value).
    """

    metrics: dict[str, list[tuple[float, float]]] = {}

    # CSV logs
    for csv_path in _iter_files(root, {".csv"}):
        try:
            fieldnames, rows = _parse_csv_metrics(csv_path)
        except Exception:
            continue
        if not rows:
            continue
        step_key = next((k for k in ("step", "global_step", "iteration", "iter") if k in fieldnames), None)
        if step_key is None:
            continue
        for field in fieldnames:
            if field == step_key:
                continue
            series: list[tuple[float, float]] = []
            for row in rows:
                x = _try_float(row.get(step_key))
                y = _try_float(row.get(field))
                if x is None or y is None:
                    continue
                series.append((x, y))
            if series:
                metrics[f"{csv_path.name}:{field}"] = series

    # JSONL logs
    for jsonl_path in _iter_files(root, {".jsonl"}):
        series_map: dict[str, list[tuple[float, float]]] = {}
        try:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        continue
                    step = _try_float(row.get("step") or row.get("global_step") or row.get("iter"))
                    if step is None:
                        continue
                    for key, value in row.items():
                        if key in {"step", "global_step", "iter"}:
                            continue
                        y = _try_float(value)
                        if y is None:
                            continue
                        series_map.setdefault(key, []).append((step, y))
        except Exception:
            continue
        for key, series in series_map.items():
            if series:
                metrics[f"{jsonl_path.name}:{key}"] = series

    return metrics


def _plot_checkpoint_progress(series: list[dict], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    steps = [row["step"] for row in series]
    indices = list(range(1, len(series) + 1))
    model_sizes = [row["model_mb"] for row in series]

    axes[0].plot(indices, steps, marker="o", linewidth=1.8)
    for idx, row in enumerate(series, start=1):
        axes[0].annotate(str(int(row["step"])), (idx, row["step"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    axes[0].set_title("LeRobot ACT checkpoint step progression")
    axes[0].set_xlabel("checkpoint index")
    axes[0].set_ylabel("training step")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, model_sizes, marker="o", linewidth=1.8, color="tab:orange")
    axes[1].set_title("Model file size vs training step")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("model.safetensors size (MB)")
    axes[1].grid(True, alpha=0.3)

    out_path = out_dir / "tiago_act_checkpoint_progress.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _plot_metrics(metrics: dict[str, list[tuple[float, float]]], out_dir: Path) -> Path:
    selected = sorted(metrics.items(), key=lambda item: len(item[1]), reverse=True)[:6]
    cols = 2
    rows = max(1, math.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows), constrained_layout=True)

    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = list(getattr(axes, "flatten", lambda: axes)())

    for ax in axes[len(selected):]:
        ax.axis("off")

    for ax, (name, series) in zip(axes, selected):
        xs = [p[0] for p in series]
        ys = [p[1] for p in series]
        ax.plot(xs, ys, linewidth=1.8)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)

    out_path = out_dir / "tiago_act_metrics.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TIAGo ACT training progress.")
    parser.add_argument(
        "--root",
        default=os.environ.get(
            "TIAGO_ACT_POLICY_ROOT",
            str(default_isaacsim_data_root() / "tiago_act_policy"),
        ),
        help="LeRobot ACT output root containing checkpoints/",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for the generated plots (defaults to <root>/_analysis)",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    series = _read_checkpoint_series(root)
    if not series:
        raise SystemExit(f"No ACT checkpoints found under: {root}")

    metrics = _discover_scalar_metrics(root)
    plots: list[Path] = []
    if metrics:
        plots.append(_plot_metrics(metrics, out_dir))
    plots.append(_plot_checkpoint_progress(series, out_dir))

    summary_path = out_dir / "tiago_act_checkpoint_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["checkpoint_dir", "step", "model_mb", "has_train_config"])
        writer.writeheader()
        writer.writerows(series)

    print(f"[OK] root: {root}")
    print(f"[OK] summary: {summary_path}")
    for plot in plots:
        print(f"[OK] plot: {plot}")
    if not metrics:
        print("[INFO] No scalar logs found; plotted checkpoint progression instead.")


if __name__ == "__main__":
    main()
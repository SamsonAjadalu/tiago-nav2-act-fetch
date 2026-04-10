#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def last_scalar(ea: EventAccumulator, tag: str) -> float | None:
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        return None
    vals = ea.Scalars(tag)
    if not vals:
        return None
    return float(vals[-1].value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ACT/PPO grasp-pose run metrics.")
    parser.add_argument("--run_dir", required=True, help="Run directory under logs/rl_games/.../<run>")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summaries = run_dir / "summaries"
    if not summaries.exists():
        raise SystemExit(f"Missing summaries dir: {summaries}")

    ea = EventAccumulator(str(summaries))
    ea.Reload()

    keys = [
        "Episode/success_rate",
        "Episode/failure_rate",
        "Episode/mean_d_pos",
        "Episode/mean_span_dist",
        "Episode/mean_centered",
        "Episode/mean_in_zone_frac",
        "rewards/iter",
        "shaped_rewards/iter",
        "episode_lengths/iter",
    ]

    print(f"run_dir: {run_dir}")
    for k in keys:
        v = last_scalar(ea, k)
        print(f"{k}: {v if v is not None else 'MISSING'}")


if __name__ == "__main__":
    main()

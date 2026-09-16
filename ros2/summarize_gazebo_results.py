#!/usr/bin/env python3
"""Summarise Gazebo evaluation CSVs written by drl_inference_node.py.

    python3 summarize_gazebo_results.py [results_gazebo]

Prints a markdown table (one row per policy CSV), writes
``<dir>/gazebo_summary.md`` and, when trajectory dumps exist
(``--traj-dir`` was used), ``<dir>/gazebo_traj_<policy>.png``.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nav2d.render import plot_trajectories  # noqa: E402


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results_gazebo")
    rows = []
    for csv in sorted(glob.glob(str(d / "*.csv"))):
        df = pd.read_csv(csv)
        if df.empty:
            continue
        succ = df[df.outcome == "success"]
        rows.append(dict(
            policy=Path(csv).stem, episodes=len(df),
            success=f"{(df.outcome == 'success').mean() * 100:.0f}%",
            collision=f"{(df.outcome == 'collision').mean() * 100:.0f}%",
            timeout=f"{(df.outcome == 'timeout').mean() * 100:.0f}%",
            steps_to_goal=f"{succ.steps.mean():.1f}" if len(succ) else "-",
            time_to_goal_s=f"{succ.time_s.mean():.1f}" if len(succ) else "-",
            path_eff=f"{succ.path_efficiency.mean():.2f}" if len(succ) else "-",
            mean_start_dist=f"{df.start_dist.mean():.2f}",
        ))
        traj_dir = d / f"traj_{Path(csv).stem}"
        if traj_dir.is_dir():
            eps = [np.load(f, allow_pickle=True).item() for f in sorted(glob.glob(str(traj_dir / "*.npy")))]
            if eps:
                plot_trajectories(eps, str(d / f"gazebo_traj_{Path(csv).stem}.png"),
                                  title=f"Gazebo: {Path(csv).stem} ({len(eps)} episodes)")
    if not rows:
        print(f"no CSVs in {d}")
        return
    table = pd.DataFrame(rows)
    try:
        md = table.to_markdown(index=False)
    except ImportError:  # tabulate not installed
        cols = list(table.columns)
        md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        md += "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in table.itertuples(index=False))
    print(md)
    (d / "gazebo_summary.md").write_text("# Gazebo evaluation (drl_inference_node.py)\n\n" + md + "\n")


if __name__ == "__main__":
    main()

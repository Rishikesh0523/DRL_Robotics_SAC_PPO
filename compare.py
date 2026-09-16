#!/usr/bin/env python3
"""Aggregate training runs + evaluations into comparison tables and plots.

    python compare.py --runs runs --evals results --out results

Produces
    results/learning_curves.png     eval success rate & return vs env steps (mean +- std over seeds)
    results/training_curves.png     rolling training return / episode length / success from monitor.csv
    results/comparison_bars.png     final metrics per policy
    results/comparison.md           markdown tables (per seed and aggregated)
    results/comparison.json         machine-readable summary
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"sac": "tab:blue", "ppo": "tab:red", "pctrl": "tab:gray", "pctrl_avoid": "tab:olive", "random": "black"}


def load_runs(runs_dir: str):
    runs = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        cfg_p, ev_p, sm_p = (os.path.join(d, f) for f in ("config.json", "evaluations.npz", "train_summary.json"))
        if not (os.path.isfile(cfg_p) and os.path.isfile(ev_p)):
            continue
        with open(cfg_p) as f:
            cfg = json.load(f)
        ev = np.load(ev_p)
        run = dict(tag=os.path.basename(d), algo=cfg["algo"], seed=cfg["seed"], dir=d,
                   timesteps=ev["timesteps"], results=ev["results"], ep_lengths=ev["ep_lengths"])
        if "successes" in ev:
            run["successes"] = ev["successes"]
        mon = glob.glob(os.path.join(d, "monitor_0.monitor.csv"))
        if mon:
            run["monitor"] = pd.read_csv(mon[0], skiprows=1)
        if os.path.isfile(sm_p):
            with open(sm_p) as f:
                run["summary"] = json.load(f)
        runs.append(run)
    return runs


def plot_learning_curves(runs, out: Path):
    by_algo = defaultdict(list)
    for r in runs:
        by_algo[r["algo"]].append(r)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=120)
    for algo, rs in by_algo.items():
        # align on common timesteps (min length)
        n = min(len(r["timesteps"]) for r in rs)
        t = rs[0]["timesteps"][:n]
        ret = np.stack([r["results"][:n].mean(axis=1) for r in rs])
        ln = np.stack([r["ep_lengths"][:n].mean(axis=1) for r in rs])
        c = COLORS.get(algo, None)
        for ax, y, lab in ((axes[0], ret, "eval mean return"), (axes[2], ln, "eval episode length")):
            m, s = y.mean(0), y.std(0)
            ax.plot(t, m, color=c, label=f"{algo.upper()} (n={len(rs)})")
            ax.fill_between(t, m - s, m + s, color=c, alpha=0.2)
            ax.set_ylabel(lab)
        if all("successes" in r for r in rs):
            sr = np.stack([r["successes"][:n].mean(axis=1) for r in rs])
            m, s = sr.mean(0), sr.std(0)
            axes[1].plot(t, m, color=c, label=algo.upper())
            axes[1].fill_between(t, m - s, m + s, color=c, alpha=0.2)
    axes[1].set_ylabel("eval success rate")
    axes[1].set_ylim(-0.02, 1.02)
    for ax in axes:
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Deterministic evaluation during training (20 episodes per point, mean +- std over seeds)")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png")
    plt.close(fig)


def plot_training_curves(runs, out: Path, window: int = 100):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=120)
    for r in runs:
        if "monitor" not in r:
            continue
        m = r["monitor"]
        steps = m["l"].cumsum()
        c = COLORS.get(r["algo"])
        alpha = 0.9 if r["seed"] == 0 else 0.4
        lab = r["algo"].upper() if r["seed"] == 0 else None
        axes[0].plot(steps, m["r"].rolling(window, min_periods=1).mean(), color=c, alpha=alpha, label=lab)
        axes[1].plot(steps, m["is_success"].astype(float).rolling(window, min_periods=1).mean(), color=c, alpha=alpha)
        axes[2].plot(steps, m["collision"].astype(float).rolling(window, min_periods=1).mean(), color=c, alpha=alpha)
    for ax, t in zip(axes, ("training episode return", "training success rate", "training collision rate")):
        ax.set_title(f"{t} (rolling {window} episodes)")
        ax.set_xlabel("environment steps (of env 0)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "training_curves.png")
    plt.close(fig)


def load_evals(evals_dir: str):
    evals = []
    for p in sorted(glob.glob(os.path.join(evals_dir, "eval_*.json"))):
        with open(p) as f:
            d = json.load(f)
        s = d["summary"]
        s["file"] = os.path.basename(p)
        evals.append(s)
    return evals


def plot_bars(evals, out: Path):
    if not evals:
        return
    evals = [e for e in evals if not e.get("fixed_goal") and "DRdyn" not in e.get("file", "")]
    # keep the chart readable: baselines + shipped models (+ nominal ablation), not every seed
    keep = ("random", "pctrl", "pctrl_avoid", "ppo_nominal_seed0_best", "ppo_best", "sac_best")
    curated = [e for e in evals if e["name"] in keep]
    if len(curated) >= 3:
        evals = sorted(curated, key=lambda e: keep.index(e["name"]))
    names = [e["name"] for e in evals]
    metrics = [("success_rate", "success rate"), ("collision_rate", "collision rate"),
               ("spl", "SPL (success x path efficiency)"), ("mean_time_to_goal_s", "time to goal [s] (successes)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), dpi=120)
    for ax, (k, lab) in zip(axes, metrics):
        vals = [e.get(k, np.nan) for e in evals]
        cols = [COLORS.get(n.split("_")[0], "tab:purple") for n in names]
        ax.bar(range(len(names)), vals, color=cols)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(lab)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "comparison_bars.png")
    plt.close(fig)


def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v*100:.1f}%" if pct else f"{v:.2f}"


def write_markdown(runs, evals, out: Path):
    lines = ["# SAC vs PPO vs P-controller: 2-D arena results", ""]
    lines += ["## Training runs", "",
              "| run | algo | seed | steps | wall-clock [min] | steps/s | best-model success | best-model return |",
              "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        s = r.get("summary", {})
        b = s.get("best_model", s.get("final_model", {}))
        lines.append(f"| {r['tag']} | {r['algo'].upper()} | {r['seed']} | {s.get('steps', '-'):,} | "
                     f"{s.get('wall_clock_s', float('nan'))/60:.1f} | {s.get('steps_per_s', float('nan')):.0f} | "
                     f"{fmt(b.get('success_rate'), True)} | {fmt(b.get('mean_return'))} |")
    lines += ["", "## Evaluation (deterministic policy, random start/goal unless noted)", "",
              "*dynamics*: nominal = ideal simulator dynamics; perturbed = domain-randomised dynamics "
              "(speed loss, lateral slip, delays, jittered control period) that mimic the Gazebo robot.", "",
              "| policy | dynamics | episodes | success | collision | timeout | mean return | steps to goal | "
              "time to goal [s] | path eff. | SPL | mean abs w [rad/s] |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in sorted(evals, key=lambda e: (e["name"], "DRdyn" in e.get("file", ""), e.get("fixed_goal", False))):
        name = e["name"] + (" (fixed goal 2.5,2.5)" if e.get("fixed_goal") else "")
        dyn = "perturbed" if "DRdyn" in e.get("file", "") else "nominal"
        lines.append(f"| {name} | {dyn} | {e['episodes']} | {fmt(e['success_rate'], True)} | {fmt(e['collision_rate'], True)} | "
                     f"{fmt(e['timeout_rate'], True)} | {fmt(e['mean_return'])} | {fmt(e['mean_steps_success'])} | "
                     f"{fmt(e['mean_time_to_goal_s'])} | {fmt(e['path_efficiency'])} | {fmt(e['spl'])} | "
                     f"{fmt(e['mean_abs_angular_vel'])} |")
    lines += ["", "Metric definitions: *success* = within 0.30 m of the goal; *collision* = LiDAR min range < 0.20 m "
              "or body contact; *timeout* = 300 steps (36 s); *path eff.* = shortest/actual path on successful "
              "episodes; *SPL* = success weighted by path length (0 on failure); one step = 0.12 s.", ""]
    with open(out / "comparison.md", "w") as f:
        f.write("\n".join(lines))
    with open(out / "comparison.json", "w") as f:
        json.dump({"runs": [{k: v for k, v in r.items() if k in ("tag", "algo", "seed", "summary")} for r in runs],
                   "evals": evals}, f, indent=1, default=float)
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--evals", default="results")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(a.runs)
    evals = load_evals(a.evals)
    if runs:
        plot_learning_curves(runs, out)
        plot_training_curves(runs, out)
    plot_bars(evals, out)
    write_markdown(runs, evals, out)


if __name__ == "__main__":
    main()

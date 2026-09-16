#!/usr/bin/env python3
"""Evaluate a trained SB3 model (or the P-controller baseline) in the 2-D arena.

Examples
--------
    python evaluate.py --model runs/sac_seed0/best_model.zip --episodes 200
    python evaluate.py --policy pctrl --episodes 200
    python evaluate.py --model models/sac_best.zip --fixed-goal --gif results/sac_demo.gif

Writes ``<out>.json`` (per-episode rows + summary) and ``<out>_traj.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from nav2d import Nav2DEnv, DEFAULT_CONFIG, PController
from nav2d.render import plot_trajectories, render_frame, save_gif


def load_policy(args):
    if args.policy == "pctrl":
        return PController(avoid=False), "pctrl"
    if args.policy == "pctrl_avoid":
        return PController(avoid=True), "pctrl_avoid"
    if args.policy == "random":
        class _R:
            def predict(self, obs, deterministic=True):
                return np.random.uniform(-1, 1, size=3).astype(np.float32), None
        return _R(), "random"
    from stable_baselines3 import PPO, SAC
    path = args.model
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    # runs/<tag>/best_model.zip -> <tag> ; models/sac_best.zip -> sac_best
    name = Path(path).stem if parent in ("models", "", ".") else parent
    algo = args.algo or ("sac" if "sac" in path.lower() else "ppo")
    model = (SAC if algo == "sac" else PPO).load(path, device="cpu")
    return model, name


def run(args):
    cfg = dict(DEFAULT_CONFIG)
    if args.fixed_goal:
        cfg["random_goal"] = False
    if args.fixed_start:
        cfg["random_start"] = False
    if args.dr:
        cfg["dr"] = True
    if args.env_cfg:
        cfg.update(json.loads(args.env_cfg))
    env = Nav2DEnv(cfg)
    policy, name = load_policy(args)

    rows, trajs, frames = [], [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done, ret, ang_sum, act_delta = False, 0.0, 0.0, 0.0
        prev_a = np.zeros(3)
        record_gif = args.gif and ep < args.gif_episodes
        if record_gif:
            frames.append(render_frame(env, dpi=60))
        while not done:
            a, _ = policy.predict(obs, deterministic=not args.stochastic)
            a = np.asarray(a, dtype=np.float32).reshape(-1)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            ang_sum += abs(float(obs[14]))
            act_delta += float(np.linalg.norm(a - prev_a))
            prev_a = a
            done = term or trunc
            if record_gif and (info["steps"] % 2 == 0 or done):
                frames.append(render_frame(env, dpi=60))
        outcome = "success" if info["is_success"] else ("collision" if info["collision"] else "timeout")
        # SPL-style path efficiency: shortest possible path / actual path, 0 on failure
        shortest = max(info["start_dist"] - cfg["goal_tolerance"], 1e-6)
        eff = min(1.0, shortest / max(info["path_length"], 1e-6)) if info["is_success"] else 0.0
        rows.append(dict(episode=ep, outcome=outcome, success=info["is_success"], collision=info["collision"],
                         timeout=info["timeout"], steps=info["steps"], ret=ret, path_length=info["path_length"],
                         start_dist=info["start_dist"], final_dist=info["dist_to_goal"], path_efficiency=eff,
                         mean_abs_w=ang_sum / info["steps"], mean_action_change=act_delta / info["steps"]))
        trajs.append(dict(traj=np.asarray(env.trajectory), goal=env.goal.copy(), outcome=outcome))

    R = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k != "outcome"}
    succ = R["success"].astype(bool)
    summary = dict(
        name=name, episodes=args.episodes, fixed_goal=bool(args.fixed_goal), fixed_start=bool(args.fixed_start),
        success_rate=float(succ.mean()), collision_rate=float(R["collision"].mean()),
        timeout_rate=float(R["timeout"].mean()), mean_return=float(R["ret"].mean()), std_return=float(R["ret"].std()),
        mean_steps=float(R["steps"].mean()),
        mean_steps_success=float(R["steps"][succ].mean()) if succ.any() else float("nan"),
        mean_time_to_goal_s=float(R["steps"][succ].mean() * cfg["dt"]) if succ.any() else float("nan"),
        path_efficiency=float(R["path_efficiency"][succ].mean()) if succ.any() else 0.0,
        spl=float(np.mean(R["path_efficiency"])),  # success weighted by path length (Anderson et al.)
        mean_abs_angular_vel=float(R["mean_abs_w"].mean()), mean_action_change=float(R["mean_action_change"].mean()),
        mean_final_dist=float(R["final_dist"].mean()),
    )
    print(f"== {name}  ({args.episodes} episodes{', fixed goal' if args.fixed_goal else ''})")
    for k in ("success_rate", "collision_rate", "timeout_rate", "mean_return", "mean_steps_success", "spl",
              "path_efficiency", "mean_abs_angular_vel"):
        print(f"   {k:24s} {summary[k]:.3f}")

    out = Path(args.out or f"results/eval_{name}{'_fixedgoal' if args.fixed_goal else ''}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".json", "w") as f:
        json.dump({"summary": summary, "episodes": rows}, f, indent=1, default=float)
    plot_trajectories(trajs, str(out) + "_traj.png",
                      title=f"{name}: success {summary['success_rate']*100:.0f}%  "
                            f"collision {summary['collision_rate']*100:.0f}%  (n={args.episodes})")
    if args.gif and frames:
        save_gif(frames, args.gif, fps=8)
        print(f"   gif -> {args.gif}")
    print(f"   saved {out}.json / _traj.png")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="path to SB3 .zip")
    p.add_argument("--algo", choices=["sac", "ppo"], default=None)
    p.add_argument("--policy", choices=["model", "pctrl", "pctrl_avoid", "random"], default="model")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--fixed-goal", action="store_true")
    p.add_argument("--fixed-start", action="store_true")
    p.add_argument("--dr", action="store_true", help="evaluate under domain randomisation (perturbed dynamics)")
    p.add_argument("--env-cfg", default=None, help="JSON overriding env config")
    p.add_argument("--out", default=None, help="output prefix (no extension)")
    p.add_argument("--gif", default=None, help="write an animated GIF of the first episodes")
    p.add_argument("--gif-episodes", type=int, default=3)
    args = p.parse_args()
    if args.policy == "model" and not args.model:
        p.error("--model is required unless --policy is pctrl/pctrl_avoid/random")
    run(args)


if __name__ == "__main__":
    main()

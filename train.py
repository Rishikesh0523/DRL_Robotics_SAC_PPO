#!/usr/bin/env python3
"""Train SAC or PPO on the 2-D navigation arena with Stable-Baselines3.

Examples
--------
    python train.py --algo sac --steps 500000 --seed 0
    python train.py --algo ppo --steps 1500000 --seed 0 --n-envs 8

Outputs (under ``runs/<tag>/``):
    best_model.zip        best deterministic eval model (EvalCallback)
    final_model.zip       model at the end of training
    checkpoints/          periodic checkpoints
    evaluations.npz       eval returns / lengths / success over training
    monitor.csv           per-episode training log (Monitor)
    tb/                   TensorBoard event files
    config.json           full env + algo config actually used
    train_summary.json    wall-clock, steps/s, final eval numbers
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from nav2d import Nav2DEnv, DEFAULT_CONFIG

ALGO_DEFAULTS = {
    "sac": dict(
        learning_rate=3e-4,
        buffer_size=300_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        learning_starts=5_000,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
    ),
    "ppo": dict(
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    ),
}


class SuccessRateCallback(BaseCallback):
    """Log rolling training success / collision / timeout rates to TensorBoard."""

    def __init__(self, window: int = 100):
        super().__init__()
        self.window = window
        self.buf: list = []

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if done:
                self.buf.append((info.get("is_success", False), info.get("collision", False),
                                 info.get("timeout", False)))
        if len(self.buf) > self.window:
            self.buf = self.buf[-self.window:]
        if self.buf and self.n_calls % 1000 == 0:
            arr = np.asarray(self.buf, dtype=float)
            self.logger.record("rollout/collision_rate", arr[:, 1].mean())
            self.logger.record("rollout/timeout_rate", arr[:, 2].mean())
        return True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["sac", "ppo"], required=True)
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=None, help="parallel envs (default: 1 sac, 8 ppo)")
    p.add_argument("--tag", default=None, help="run name (default: <algo>_seed<seed>)")
    p.add_argument("--logdir", default="runs")
    p.add_argument("--device", default="auto")
    p.add_argument("--eval-freq", type=int, default=10_000, help="env steps between evals")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    p.add_argument("--fixed-goal", action="store_true", help="train on the fixed (2.5, 2.5) goal only")
    p.add_argument("--fixed-start", action="store_true")
    p.add_argument("--dr", action="store_true",
                   help="domain randomisation: per-episode random speed efficiency, lateral slip, "
                        "acceleration limits, sensor/actuation delays, control period and LiDAR noise")
    p.add_argument("--sac-train-freq", type=int, default=1, help="SAC: env steps per gradient update")
    p.add_argument("--env-cfg", default=None, help="JSON string/file overriding env config")
    p.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    return p.parse_args()


def load_env_cfg(args) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if args.fixed_goal:
        cfg["random_goal"] = False
    if args.fixed_start:
        cfg["random_start"] = False
    if args.dr:
        cfg["dr"] = True
    if args.env_cfg:
        if os.path.isfile(args.env_cfg):
            with open(args.env_cfg) as f:
                cfg.update(json.load(f))
        else:
            cfg.update(json.loads(args.env_cfg))
    return cfg


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    n_envs = args.n_envs or (1 if args.algo == "sac" else 8)
    tag = args.tag or f"{args.algo}_seed{args.seed}"
    out = Path(args.logdir) / tag
    out.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    env_cfg = load_env_cfg(args)

    def env_fn(rank: int):
        def _init():
            env = Nav2DEnv(env_cfg)
            env.reset(seed=args.seed * 1000 + rank)
            return Monitor(env, filename=str(out / f"monitor_{rank}") if rank == 0 else None,
                           info_keywords=("is_success", "collision", "timeout", "path_length", "start_dist"))
        return _init

    train_env = DummyVecEnv([env_fn(i) for i in range(n_envs)])
    eval_env = DummyVecEnv([env_fn(999)])

    algo_kwargs = dict(ALGO_DEFAULTS[args.algo])
    if args.algo == "sac" and args.sac_train_freq != 1:
        algo_kwargs["train_freq"] = args.sac_train_freq
    Algo = SAC if args.algo == "sac" else PPO
    model = Algo("MlpPolicy", train_env, verbose=0, seed=args.seed, device=args.device,
                 tensorboard_log=str(out / "tb"), **algo_kwargs)

    with open(out / "config.json", "w") as f:
        json.dump({"algo": args.algo, "steps": args.steps, "seed": args.seed, "n_envs": n_envs,
                   "algo_kwargs": {k: (v if not isinstance(v, dict) else v) for k, v in algo_kwargs.items()},
                   "env_cfg": env_cfg}, f, indent=2, default=str)

    callbacks = [
        EvalCallback(eval_env, best_model_save_path=str(out), log_path=str(out),
                     eval_freq=max(args.eval_freq // n_envs, 1), n_eval_episodes=args.eval_episodes,
                     deterministic=True, render=False, verbose=0),
        CheckpointCallback(save_freq=max(args.checkpoint_freq // n_envs, 1),
                           save_path=str(out / "checkpoints"), name_prefix=args.algo, verbose=0),
        SuccessRateCallback(),
    ]

    print(f"[{tag}] training {args.algo.upper()} for {args.steps:,} steps on {model.device} "
          f"with {n_envs} env(s)  ->  {out}", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks, tb_log_name="run", progress_bar=False)
    elapsed = time.time() - t0
    model.save(out / "final_model")

    # ---- final deterministic evaluation of the final and best models
    def eval_model(m, n=100):
        succ, coll, tout = [], [], []

        def cb(locals_, globals_):
            if locals_["dones"][0]:
                info = locals_["infos"][0]
                succ.append(info.get("is_success", False))
                coll.append(info.get("collision", False))
                tout.append(info.get("timeout", False))

        rets, lens = evaluate_policy(m, eval_env, n_eval_episodes=n, deterministic=True,
                                     return_episode_rewards=True, callback=cb)
        return dict(success_rate=float(np.mean(succ)), collision_rate=float(np.mean(coll)),
                    timeout_rate=float(np.mean(tout)), mean_return=float(np.mean(rets)),
                    std_return=float(np.std(rets)), mean_length=float(np.mean(lens)))

    summary = {
        "tag": tag, "algo": args.algo, "seed": args.seed, "steps": args.steps, "n_envs": n_envs,
        "wall_clock_s": elapsed, "steps_per_s": args.steps / elapsed, "device": str(model.device),
        "final_model": eval_model(model),
    }
    best_path = out / "best_model.zip"
    if best_path.exists():
        summary["best_model"] = eval_model(Algo.load(best_path, device=args.device))
    with open(out / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{tag}] done in {elapsed/60:.1f} min ({summary['steps_per_s']:.0f} steps/s)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

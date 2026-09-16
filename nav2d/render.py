"""Matplotlib rendering helpers: single frames, trajectory overlays, GIFs."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .arena import Arena, DEFAULT_ARENA  # noqa: E402

OUTCOME_COLORS = {"success": "tab:green", "collision": "tab:red", "timeout": "tab:orange"}


def _draw_robot(ax, pos, yaw, radius, color="tab:blue"):
    import matplotlib.patches as patches

    ax.add_patch(patches.Circle(pos, radius, fc=color, ec="black", alpha=0.9, zorder=5))
    ax.arrow(pos[0], pos[1], radius * 1.6 * math.cos(yaw), radius * 1.6 * math.sin(yaw),
             head_width=0.08, color="black", zorder=6)


def render_frame(env, show_scan: bool = True, dpi: int = 80) -> np.ndarray:
    """Render the current env state to an RGB array."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    env.arena.draw(ax)
    if show_scan:
        ox, oy = env.cfg["lidar_offset"]
        cy, sy = math.cos(env.yaw), math.sin(env.yaw)
        origin = env.pos + np.array([ox * cy - oy * sy, ox * sy + oy * cy])
        ang = env.ray_angles[::6] + env.yaw
        rng = env._last_scan[::6]
        ax.plot(np.stack([np.full_like(rng, origin[0]), origin[0] + rng * np.cos(ang)]),
                np.stack([np.full_like(rng, origin[1]), origin[1] + rng * np.sin(ang)]),
                color="tab:cyan", lw=0.4, alpha=0.5)
    traj = np.asarray(env.trajectory)
    if len(traj) > 1:
        ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", lw=1.5)
    _draw_robot(ax, env.pos, env.yaw, env.cfg["robot_radius"])
    ax.plot(*env.goal, marker="*", color="gold", ms=18, mec="black", zorder=7)
    ax.add_patch(plt.Circle(env.goal, env.cfg["goal_tolerance"], fill=False, ls="--", ec="gold"))
    d, _ = env.goal_polar()
    ax.set_title(f"step {env.step_count}  dist {d:.2f} m")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def plot_trajectories(trajectories: Sequence[dict], path: str, title: str = "",
                      arena: Arena = DEFAULT_ARENA, max_n: int = 60) -> None:
    """Overlay episode paths.  Each item: {traj: (N,2), goal: (2,), outcome: str}."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120)
    arena.draw(ax)
    for ep in list(trajectories)[:max_n]:
        t = np.asarray(ep["traj"])
        col = OUTCOME_COLORS.get(ep["outcome"], "gray")
        ax.plot(t[:, 0], t[:, 1], color=col, lw=1.0, alpha=0.75)
        ax.plot(t[0, 0], t[0, 1], "o", color=col, ms=4)
        ax.plot(ep["goal"][0], ep["goal"][1], "*", color="gold", mec="black", ms=9)
    for k, c in OUTCOME_COLORS.items():
        ax.plot([], [], color=c, label=k)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_gif(frames: Iterable[np.ndarray], path: str, fps: int = 8) -> None:
    import imageio.v2 as imageio

    imageio.mimsave(path, list(frames), duration=1.0 / fps, loop=0)

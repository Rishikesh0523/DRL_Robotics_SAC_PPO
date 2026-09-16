"""Hand-crafted baseline policies operating on the 16-D observation."""

from __future__ import annotations

import math

import numpy as np


class PController:
    """Proportional goal-seeking controller with LiDAR speed reduction.

    Mirrors the baseline described in the project report ("speed reduction"
    near obstacles, no explicit avoidance).  Holonomic: it strafes directly
    toward the goal in the robot frame and turns to face it.
    """

    def __init__(self, kp_lin: float = 1.2, kp_ang: float = 1.5, v_max: float = 0.75,
                 w_max: float = 1.5, slow_zone: float = 0.8, avoid: bool = False,
                 k_avoid: float = 0.6):
        self.kp_lin = kp_lin
        self.kp_ang = kp_ang
        self.v_max = v_max
        self.w_max = w_max
        self.slow_zone = slow_zone
        self.avoid = avoid
        self.k_avoid = k_avoid
        n = 10
        # sector centre angles in robot frame, sector 0 = [-pi, -pi+36deg)
        self.sector_angles = -math.pi + (np.arange(n) + 0.5) * (2 * math.pi / n)

    def predict(self, obs, deterministic: bool = True):
        obs = np.asarray(obs, dtype=np.float64)
        if obs.ndim == 2:  # vec-env batch
            acts = np.stack([self._act(o) for o in obs])
            return acts.astype(np.float32), None
        return self._act(obs).astype(np.float32), None

    def _act(self, obs: np.ndarray) -> np.ndarray:
        sectors = obs[:10]
        dist, bearing = obs[10], obs[11]
        speed = min(self.v_max, self.kp_lin * dist)
        min_lidar = sectors.min()
        if min_lidar < self.slow_zone:
            speed *= max(0.25, min_lidar / self.slow_zone)
        vx = speed * math.cos(bearing)
        vy = speed * math.sin(bearing)
        if self.avoid:
            # repulsive velocity away from close sectors
            close = sectors < self.slow_zone
            if close.any():
                weights = (self.slow_zone - sectors[close]) / self.slow_zone
                rep_x = -np.sum(weights * np.cos(self.sector_angles[close]))
                rep_y = -np.sum(weights * np.sin(self.sector_angles[close]))
                vx += self.k_avoid * rep_x
                vy += self.k_avoid * rep_y
        w = float(np.clip(self.kp_ang * bearing, -self.w_max, self.w_max))
        v = math.hypot(vx, vy)
        if v > self.v_max:
            vx, vy = vx / v * self.v_max, vy / v * self.v_max
        # return a normalised action in [-1, 1]^3 (env / ROS node scale it back)
        return np.array([vx / self.v_max, vy / self.v_max, w / self.w_max])

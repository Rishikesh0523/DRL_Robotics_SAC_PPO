"""Gymnasium environment: holonomic (mecanum) robot navigating the 4x4 m arena.

Observation (16-D, identical layout to ``robot_nav_env2.py``)::

    [ lidar_sector_0 .. lidar_sector_9 ,   # min range per 36 deg sector, metres, 0..10
      dist_to_goal, bearing_to_goal,       # metres, radians in [-pi, pi]
      vx, vy, w, v_mag ]                   # robot-frame velocities

LiDAR sector 0 covers angles [-180, -144) deg (rear-right), sector 5 covers
[0, 36) deg (front-left) - the same ordering produced by binning the Gazebo
``/lidar`` LaserScan (360 rays, angle_min = -pi).

Action (3-D continuous): ``[vx, vy, w]`` with bounds
``[-0.75, -0.75, -1.5] .. [0.75, 0.75, 1.5]``.  Published unchanged as a
``geometry_msgs/Twist`` (linear.x, linear.y, angular.z) by the ROS 2 node.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .arena import Arena, DEFAULT_ARENA

DEFAULT_CONFIG: Dict[str, Any] = {
    # ---- timing
    "dt": 0.12,                # seconds per env step (robot_nav_env2 slept 0.12 s)
    "substeps": 4,             # physics sub-steps per env step
    "max_steps": 300,
    # ---- LiDAR (from lidar.gazebo)
    "n_rays": 360,
    "n_sectors": 10,
    "lidar_min": 0.10,
    "lidar_max": 10.0,
    "lidar_noise_std": 0.01,
    "lidar_offset": (-0.06, 0.0),   # sensor position in robot frame (URDF: x = -0.0606)
    # ---- robot / dynamics
    "robot_radius": 0.17,      # ~half diagonal of the 0.174 x 0.154 m wheel base
    "action_low": (-0.75, -0.75, -1.5),
    "action_high": (0.75, 0.75, 1.5),
    "lin_accel": 5.0,          # m/s^2   (MecanumDrive plugin max_acceleration = 5)
    "ang_accel": 10.0,         # rad/s^2
    # ---- observation
    "vel_obs": "command",      # "command": last commanded [vx, vy, w] ; "measured": simulated body velocity
    # ---- task
    "goal_tolerance": 0.30,
    "collision_dist": 0.20,    # episode ends if min LiDAR range < this
    "random_start": True,
    "random_goal": True,
    "fixed_start": (-2.0, -2.0),
    "fixed_goal": (2.5, 2.5),
    "random_yaw": True,
    "spawn_limit": 3.2,
    "spawn_clearance": 0.45,   # clearance required for start & goal points
    "min_start_goal_dist": 1.5,
    # ---- reward
    "r_progress": 5.0,         # * (d_prev - d)
    "r_goal": 20.0,
    "r_collision": -20.0,
    "r_step": -0.05,
    "clear_zone": 0.50,        # proximity penalty starts below this LiDAR range
    "r_clear": 0.30,           # max proximity penalty per step
    "r_align": 0.05,           # reward for velocity pointing toward the goal
    "r_smooth": 0.02,          # penalty on action change (reduces wobble)
    "r_timeout": 0.0,
    # ---- domain randomisation (sampled once per episode when dr=True).
    # Ranges were chosen from measurements on the Gazebo robot: lateral (vy) commands only
    # achieve ~50 % of the requested speed and drag the robot backwards, sensor data arrives
    # 1-2 control steps late, and the real-time factor varies.
    "dr": False,
    "dr_speed_x": (0.6, 1.0),        # achieved / commanded forward speed
    "dr_speed_y": (0.4, 1.0),        # achieved / commanded lateral speed
    "dr_speed_w": (0.7, 1.0),        # achieved / commanded yaw rate
    "dr_couple_xy": (0.0, 0.5),      # backward drift per unit lateral speed
    "dr_lin_accel": (1.5, 5.0),
    "dr_ang_accel": (3.0, 10.0),
    "dr_obs_delay": (0, 2),          # observation delay in control steps (inclusive)
    "dr_act_delay": (0, 1),          # action delay in control steps (inclusive)
    "dr_dt": (0.09, 0.15),           # effective control period (real-time-factor jitter)
    "dr_lidar_noise": (0.005, 0.03),
}


class Nav2DEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 8}

    def __init__(self, config: Optional[Dict[str, Any]] = None, arena: Arena = DEFAULT_ARENA,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = dict(DEFAULT_CONFIG)
        if config:
            self.cfg.update(config)
        self.arena = arena
        self.render_mode = render_mode

        c = self.cfg
        self.n_rays = int(c["n_rays"])
        self.n_sectors = int(c["n_sectors"])
        self.ray_angles = -math.pi + np.arange(self.n_rays) * (2 * math.pi / self.n_rays)
        self.rays_per_sector = self.n_rays // self.n_sectors

        # Physical command bounds; the agent acts in the normalised box [-1, 1]^3
        # and step() scales into these bounds (see scale_action).
        self.action_low = np.array(c["action_low"], dtype=np.float32)
        self.action_high = np.array(c["action_high"], dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        lidar_low = np.zeros(self.n_sectors, dtype=np.float32)
        lidar_high = np.full(self.n_sectors, c["lidar_max"], dtype=np.float32)
        goal_low = np.array([0.0, -math.pi], dtype=np.float32)
        goal_high = np.array([c["lidar_max"], math.pi], dtype=np.float32)
        vel_low = np.array([-2.0, -2.0, -math.pi, 0.0], dtype=np.float32)
        vel_high = np.array([2.0, 2.0, math.pi, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            np.concatenate([lidar_low, goal_low, vel_low]),
            np.concatenate([lidar_high, goal_high, vel_high]),
            dtype=np.float32,
        )

        # state
        self.pos = np.zeros(2)
        self.yaw = 0.0
        self.vel = np.zeros(3)          # robot-frame [vx, vy, w]
        self.goal = np.zeros(2)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.prev_dist = 0.0
        self.path_length = 0.0
        self.start_dist = 0.0
        self.trajectory: list = []
        self._last_scan = np.full(self.n_rays, c["lidar_max"])
        self._rng = np.random.default_rng()
        self.dyn = self._nominal_dynamics()
        self._obs_hist: list = []
        self._act_hist: list = []

    # ------------------------------------------------------- dynamics sampling
    def _nominal_dynamics(self) -> Dict[str, float]:
        c = self.cfg
        return dict(speed_x=1.0, speed_y=1.0, speed_w=1.0, couple_xy=0.0, lin_accel=c["lin_accel"],
                    ang_accel=c["ang_accel"], obs_delay=0, act_delay=0, dt=c["dt"],
                    lidar_noise=c["lidar_noise_std"])

    def _sample_dynamics(self) -> Dict[str, float]:
        if not self.cfg["dr"]:
            return self._nominal_dynamics()
        c, u = self.cfg, self._rng.uniform
        return dict(
            speed_x=u(*c["dr_speed_x"]), speed_y=u(*c["dr_speed_y"]), speed_w=u(*c["dr_speed_w"]),
            couple_xy=u(*c["dr_couple_xy"]), lin_accel=u(*c["dr_lin_accel"]), ang_accel=u(*c["dr_ang_accel"]),
            obs_delay=int(self._rng.integers(c["dr_obs_delay"][0], c["dr_obs_delay"][1] + 1)),
            act_delay=int(self._rng.integers(c["dr_act_delay"][0], c["dr_act_delay"][1] + 1)),
            dt=u(*c["dr_dt"]), lidar_noise=u(*c["dr_lidar_noise"]),
        )

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def wrap(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    def _scan(self) -> np.ndarray:
        c = self.cfg
        ox, oy = c["lidar_offset"]
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        origin = self.pos + np.array([ox * cy - oy * sy, ox * sy + oy * cy])
        ranges = self.arena.ray_cast(origin, self.ray_angles + self.yaw, c["lidar_max"])
        noise = self.dyn["lidar_noise"]
        if noise > 0:
            ranges = ranges + self._rng.normal(0.0, noise, size=ranges.shape)
        ranges = np.clip(ranges, c["lidar_min"], c["lidar_max"])
        self._last_scan = ranges
        return ranges

    def sectors_from_scan(self, ranges: np.ndarray) -> np.ndarray:
        n = self.rays_per_sector * self.n_sectors
        return ranges[:n].reshape(self.n_sectors, self.rays_per_sector).min(axis=1)

    def goal_polar(self) -> Tuple[float, float]:
        d = self.goal - self.pos
        dist = float(np.hypot(d[0], d[1]))
        bearing = self.wrap(math.atan2(d[1], d[0]) - self.yaw)
        return dist, bearing

    def _obs(self, sectors: np.ndarray) -> np.ndarray:
        dist, bearing = self.goal_polar()
        # Velocity features = the last *commanded* velocity (physical units).  Using the command
        # instead of measured odometry makes the feature identical in simulation and on the
        # Gazebo/ROS robot (no odometry frame / latency ambiguity); the original ROS env did the same.
        if self.cfg.get("vel_obs", "command") == "command":
            vx, vy, w = self.prev_action
        else:
            vx, vy, w = self.vel
        obs = np.concatenate([
            sectors,
            [min(dist, self.cfg["lidar_max"]), bearing],
            [vx, vy, w, math.hypot(vx, vy)],
        ]).astype(np.float32)
        return obs

    # -------------------------------------------------------------------- gym
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        c = self.cfg
        options = options or {}

        clearance = c["spawn_clearance"]
        if "start" in options:
            self.pos = np.asarray(options["start"], dtype=np.float64)
        elif c["random_start"]:
            self.pos = self.arena.sample_free(self._rng, clearance, c["spawn_limit"])
        else:
            self.pos = np.asarray(c["fixed_start"], dtype=np.float64)

        if "goal" in options:
            self.goal = np.asarray(options["goal"], dtype=np.float64)
        elif c["random_goal"]:
            for _ in range(200):
                g = self.arena.sample_free(self._rng, clearance, c["spawn_limit"])
                if np.linalg.norm(g - self.pos) >= c["min_start_goal_dist"]:
                    break
            self.goal = g
        else:
            self.goal = np.asarray(c["fixed_goal"], dtype=np.float64)

        if "yaw" in options:
            self.yaw = float(options["yaw"])
        elif c["random_yaw"]:
            self.yaw = float(self._rng.uniform(-math.pi, math.pi))
        else:
            self.yaw = 0.0

        self.vel[:] = 0.0
        self.prev_action[:] = 0.0
        self.step_count = 0
        self.path_length = 0.0
        self.start_dist, _ = self.goal_polar()
        self.prev_dist = self.start_dist
        self.trajectory = [self.pos.copy()]
        self.dyn = self._sample_dynamics()
        self._act_hist = [np.zeros(3, dtype=np.float32) for _ in range(self.dyn["act_delay"])]

        sectors = self.sectors_from_scan(self._scan())
        obs = self._obs(sectors)
        self._obs_hist = [obs] * (self.dyn["obs_delay"] + 1)
        return obs, self._info(sectors, False, False, False)

    def _delayed_obs(self, obs: np.ndarray) -> np.ndarray:
        """Return the observation from ``obs_delay`` steps ago (sensor latency model)."""
        self._obs_hist.append(obs)
        k = self.dyn["obs_delay"]
        if len(self._obs_hist) > k + 1:
            self._obs_hist.pop(0)
        return self._obs_hist[0]

    def _delayed_action(self, a: np.ndarray) -> np.ndarray:
        """Return the action issued ``act_delay`` steps ago (actuation latency model)."""
        if self.dyn["act_delay"] == 0:
            return a
        self._act_hist.append(a)
        return self._act_hist.pop(0)

    def _info(self, sectors, success, collision, timeout) -> Dict[str, Any]:
        dist, _ = self.goal_polar()
        return {
            "is_success": bool(success),
            "collision": bool(collision),
            "timeout": bool(timeout),
            "dist_to_goal": dist,
            "min_lidar": float(sectors.min()),
            "path_length": self.path_length,
            "start_dist": self.start_dist,
            "steps": self.step_count,
            "pos": self.pos.copy(),
            "yaw": self.yaw,
            "goal": self.goal.copy(),
        }

    def scale_action(self, action) -> np.ndarray:
        """Map a normalised action in [-1, 1]^3 to physical [vx, vy, w]."""
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return self.action_low + (a + 1.0) * 0.5 * (self.action_high - self.action_low)

    def step(self, action):
        c, d = self.cfg, self.dyn
        a = self.scale_action(action)
        self.step_count += 1

        # --- actuation: delay, per-axis efficiency and lateral->backward coupling (mecanum slip)
        a_applied = self._delayed_action(a)
        target = np.array([
            a_applied[0] * d["speed_x"] - d["couple_xy"] * abs(a_applied[1]),
            a_applied[1] * d["speed_y"],
            a_applied[2] * d["speed_w"],
        ])

        # --- dynamics: first-order velocity tracking with acceleration limits
        sub_dt = d["dt"] / c["substeps"]
        max_dv = np.array([d["lin_accel"], d["lin_accel"], d["ang_accel"]]) * sub_dt
        body_collision = False
        for _ in range(c["substeps"]):
            dv = np.clip(target - self.vel, -max_dv, max_dv)
            self.vel = self.vel + dv
            vx, vy, w = self.vel
            cy, sy = math.cos(self.yaw), math.sin(self.yaw)
            new_pos = self.pos + np.array([vx * cy - vy * sy, vx * sy + vy * cy]) * sub_dt
            if self.arena.clearance(new_pos) <= c["robot_radius"]:
                body_collision = True
                self.vel[:] = 0.0
                break
            self.path_length += float(np.linalg.norm(new_pos - self.pos))
            self.pos = new_pos
            self.yaw = self.wrap(self.yaw + w * sub_dt)
        self.trajectory.append(self.pos.copy())

        # --- sensing
        sectors = self.sectors_from_scan(self._scan())
        min_lidar = float(sectors.min())
        dist, bearing = self.goal_polar()

        success = dist < c["goal_tolerance"]
        collision = (not success) and (body_collision or min_lidar < c["collision_dist"])
        timeout = (not success) and (not collision) and self.step_count >= c["max_steps"]

        # --- reward
        reward = c["r_progress"] * (self.prev_dist - dist) + c["r_step"]
        if success:
            reward += c["r_goal"]
        elif collision:
            reward += c["r_collision"]
        else:
            if min_lidar < c["clear_zone"]:
                reward -= c["r_clear"] * (c["clear_zone"] - min_lidar) / c["clear_zone"]
            vmax = float(self.action_high[0])
            toward = (self.vel[0] * math.cos(bearing) + self.vel[1] * math.sin(bearing)) / vmax
            reward += c["r_align"] * toward
            reward -= c["r_smooth"] * float(np.linalg.norm((a - self.prev_action) / self.action_high))
        if timeout:
            reward += c["r_timeout"]

        self.prev_dist = dist
        self.prev_action = a
        obs = self._delayed_obs(self._obs(sectors))
        info = self._info(sectors, success, collision, timeout)
        info["dyn"] = d
        terminated = bool(success or collision)
        truncated = bool(timeout)
        return obs, float(reward), terminated, truncated, info

    # ----------------------------------------------------------------- render
    def render(self):
        from .render import render_frame
        return render_frame(self)

    def close(self):
        pass


def make_env(config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None,
             monitor: bool = True, log_dir: Optional[str] = None):
    """Factory returning a (Monitor-wrapped) Nav2DEnv; usable with SB3 make_vec_env."""
    env = Nav2DEnv(config)
    if seed is not None:
        env.reset(seed=seed)
    if monitor:
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env, filename=log_dir, info_keywords=("is_success", "collision", "timeout",
                                                             "path_length", "start_dist"))
    return env

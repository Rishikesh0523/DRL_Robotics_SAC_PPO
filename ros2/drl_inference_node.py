#!/usr/bin/env python3
"""ROS 2 inference / evaluation node for the trained navigation policy.

Runs a trained SB3 model (SAC or PPO, trained in the 2-D simulator) or the
P-controller baseline on the Gazebo mecanum robot and reports metrics.

Topics
    sub  /lidar   sensor_msgs/LaserScan   (360 rays, angle_min = -pi, range 0.1..10 m)
    sub  /odom    nav_msgs/Odometry       (pose + robot-frame twist)
    pub  /cmd_vel geometry_msgs/Twist     (linear.x, linear.y, angular.z)
    pub  /goal_pose geometry_msgs/PoseStamped  (for RViz)
    pub  /drl/status std_msgs/String       (human readable per-step status)

Episodes are reset by teleporting the robot with the Gazebo ``set_pose``
service (``gz service``), exactly like the evaluation described in the report.

Usage (after sourcing ROS 2 + your workspace, with Gazebo running):

    python3 drl_inference_node.py --model ../models/sac_best.zip --episodes 10
    python3 drl_inference_node.py --policy pctrl --episodes 10
    python3 drl_inference_node.py --model ../models/ppo_best.zip --goal 2.5 2.5 --start -2 -2 --episodes 1
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # make `nav2d` importable without installation
from nav2d import DEFAULT_ARENA, DEFAULT_CONFIG, PController  # noqa: E402

ACTION_LOW = np.array(DEFAULT_CONFIG["action_low"], dtype=np.float32)
ACTION_HIGH = np.array(DEFAULT_CONFIG["action_high"], dtype=np.float32)


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class DRLInferenceNode(Node):
    def __init__(self, args):
        super().__init__("drl_inference_node")
        self.args = args
        self.cfg = dict(DEFAULT_CONFIG)
        self.n_sectors = self.cfg["n_sectors"]
        self.lock = threading.Lock()

        sensor_qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(LaserScan, args.scan_topic, self._scan_cb, sensor_qos)
        if args.pose_source == "odom":
            pose_topic = args.odom_topic
        else:
            # The robot's /odom is written by two plugins (MecanumDrive dead-reckoning from the
            # spawn point + OdometryPublisher world pose) -> alternating jumps and no reaction to
            # teleports ("dual odometry" problem in the report).  Only OdometryPublisher writes
            # this topic, so it is unambiguous.  It is bridged by gazebo_demo.launch.py.
            pose_topic = f"/model/{args.model_name}/odometry_with_covariance"
        self.pose_topic = pose_topic
        self.create_subscription(Odometry, pose_topic, self._odom_cb, 20)
        self.cmd_pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.status_pub = self.create_publisher(String, "/drl/status", 10)

        self.scan = None
        self.scan_stamp = 0.0
        self.pos = np.zeros(2)
        self.yaw = 0.0
        self.vel = np.zeros(3)
        self.last_cmd = np.zeros(3)   # last commanded [vx, vy, w]; used as the velocity feature
        self.odom_stamp = 0.0
        self.odom_count = 0

        self.policy, self.policy_name = self._load_policy(args)
        self.rng = np.random.default_rng(args.seed)
        self.get_logger().info(f"policy: {self.policy_name}")

    # ------------------------------------------------------------ callbacks
    def _scan_cb(self, msg: LaserScan):
        r = np.asarray(msg.ranges, dtype=np.float64)
        r = np.nan_to_num(r, nan=self.cfg["lidar_max"], posinf=self.cfg["lidar_max"], neginf=self.cfg["lidar_max"])
        # self-hits / noise floor (report: readings < 0.08 m were self detections)
        r[r < self.args.lidar_floor] = self.cfg["lidar_max"]
        r = np.clip(r, self.cfg["lidar_min"], self.cfg["lidar_max"])
        if self.args.lidar_yaw_offset_deg:
            shift = int(round(self.args.lidar_yaw_offset_deg / 360.0 * len(r)))
            r = np.roll(r, shift)
        with self.lock:
            self.scan = r
            self.scan_stamp = time.time()

    def _odom_cb(self, msg: Odometry):
        with self.lock:
            self.pos[0] = msg.pose.pose.position.x
            self.pos[1] = msg.pose.pose.position.y
            self.yaw = yaw_from_quat(msg.pose.pose.orientation)
            t = msg.twist.twist
            self.vel[:] = (t.linear.x, t.linear.y, t.angular.z)
            self.odom_stamp = time.time()
            self.odom_count += 1

    # -------------------------------------------------------------- policy
    @staticmethod
    def _load_policy(args):
        if args.policy == "pctrl":
            return PController(avoid=False), "pctrl"
        if args.policy == "pctrl_avoid":
            return PController(avoid=True), "pctrl_avoid"
        from stable_baselines3 import PPO, SAC
        algo = args.algo or ("sac" if "sac" in os.path.basename(args.model).lower() else "ppo")
        model = (SAC if algo == "sac" else PPO).load(args.model, device="cpu")
        return model, f"{algo}:{os.path.basename(args.model)}"

    # ---------------------------------------------------------- observation
    def observation(self, goal: np.ndarray):
        with self.lock:
            scan = None if self.scan is None else self.scan.copy()
            pos, yaw = self.pos.copy(), self.yaw
            # velocity feature = last commanded velocity (same definition as the simulator)
            vel = self.last_cmd.copy() if self.args.vel_obs == "command" else self.vel.copy()
        if scan is None:
            return None, None
        per = len(scan) // self.n_sectors
        sectors = scan[: per * self.n_sectors].reshape(self.n_sectors, per).min(axis=1)
        d = goal - pos
        dist = float(np.hypot(*d))
        bearing = wrap(math.atan2(d[1], d[0]) - yaw)
        obs = np.concatenate([sectors, [min(dist, self.cfg["lidar_max"]), bearing],
                              [vel[0], vel[1], vel[2], math.hypot(vel[0], vel[1])]]).astype(np.float32)
        return obs, dict(pos=pos, yaw=yaw, dist=dist, min_lidar=float(sectors.min()), sectors=sectors)

    # ------------------------------------------------------------- actuation
    def send(self, action_norm):
        a = np.clip(np.asarray(action_norm, dtype=np.float32).reshape(-1), -1, 1)
        cmd = ACTION_LOW + (a + 1.0) * 0.5 * (ACTION_HIGH - ACTION_LOW)
        cmd *= self.args.speed_scale
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(cmd[0]), float(cmd[1]), float(cmd[2])
        self.cmd_pub.publish(msg)
        with self.lock:
            self.last_cmd = np.asarray(cmd, dtype=np.float64)
        return cmd

    def stop(self):
        with self.lock:
            self.last_cmd = np.zeros(3)
        for _ in range(3):
            self.cmd_pub.publish(Twist())
            time.sleep(0.05)

    def publish_goal(self, goal):
        m = PoseStamped()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x, m.pose.position.y = float(goal[0]), float(goal[1])
        m.pose.orientation.w = 1.0
        self.goal_pub.publish(m)

    # ------------------------------------------------------------- teleport
    def teleport(self, x: float, y: float, yaw: float) -> bool:
        """Move the robot with the Gazebo set_pose service."""
        qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
        req = (f'name: "{self.args.model_name}", position: {{x: {x}, y: {y}, z: 0.05}}, '
               f'orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}')
        cmd = ["gz", "service", "-s", f"/world/{self.args.world}/set_pose", "--reqtype", "gz.msgs.Pose",
               "--reptype", "gz.msgs.Boolean", "--timeout", "3000", "--req", req]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
            ok = "data: true" in out.stdout
            if not ok:
                self.get_logger().warn(f"set_pose reply: {out.stdout.strip()} {out.stderr.strip()}")
            return ok
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"teleport failed: {e}")
            return False

    def wait_for_pose(self, x, y, tol=0.15, timeout=4.0) -> bool:
        """Block until odometry reports the teleported pose (fixes stale-obs bug from the report)."""
        t0 = time.time()
        start_count = self.odom_count
        while time.time() - t0 < timeout:
            with self.lock:
                d = math.hypot(self.pos[0] - x, self.pos[1] - y)
                fresh = self.odom_count > start_count + 3
            if fresh and d < tol:
                return True
            time.sleep(0.05)
        return False

    # -------------------------------------------------------------- episode
    def sample_point(self, clearance=0.6):
        return DEFAULT_ARENA.sample_free(self.rng, clearance, self.cfg["spawn_limit"])

    def run_episode(self, ep: int, writer):
        a = self.args
        # choose start / goal
        goal = np.array(a.goal, dtype=float) if a.goal else self.sample_point()
        if a.start:
            start = np.array(a.start, dtype=float)
        elif a.no_teleport:
            with self.lock:
                start = self.pos.copy()
        else:
            for _ in range(100):
                start = self.sample_point()
                if np.linalg.norm(start - goal) >= self.cfg["min_start_goal_dist"]:
                    break
        yaw0 = a.start_yaw if a.start_yaw is not None else float(self.rng.uniform(-math.pi, math.pi))

        self.stop()
        if not a.no_teleport:
            # retry: the first set_pose right after Gazebo starts is sometimes ignored
            for attempt in range(4):
                ok = self.teleport(start[0], start[1], yaw0)
                if self.wait_for_pose(start[0], start[1], timeout=3.0 if ok else 1.0):
                    break
                self.get_logger().warn(f"teleport to ({start[0]:.2f},{start[1]:.2f}) not confirmed "
                                       f"(attempt {attempt + 1}/4); retrying")
                time.sleep(0.5)
            else:
                with self.lock:
                    start = self.pos.copy()
                self.get_logger().warn(f"using current pose ({start[0]:.2f},{start[1]:.2f}) as start")
            time.sleep(0.3)  # let LiDAR refresh at 10 Hz
        # discard stale scan
        with self.lock:
            self.scan = None
        t0 = time.time()
        while self.scan is None and time.time() - t0 < 3.0:
            time.sleep(0.02)

        self.publish_goal(goal)
        obs, st = self.observation(goal)
        if obs is None:
            self.get_logger().error("no sensor data; is Gazebo + bridge running?")
            return None
        start_dist = st["dist"]
        prev_pos = st["pos"].copy()
        path_len, steps, ret, outcome = 0.0, 0, 0.0, "timeout"
        prev_dist = start_dist
        traj = [prev_pos.copy()]
        self.get_logger().info(f"ep {ep}: start ({start[0]:.2f},{start[1]:.2f}) -> goal ({goal[0]:.2f},{goal[1]:.2f}) "
                               f"dist {start_dist:.2f} m")
        dt = self.cfg["dt"]
        while rclpy.ok() and steps < a.max_steps:
            t_step = time.time()
            act, _ = self.policy.predict(obs, deterministic=True)
            cmd = self.send(act)
            time.sleep(max(0.0, dt - (time.time() - t_step)))
            obs, st = self.observation(goal)
            steps += 1
            path_len += float(np.linalg.norm(st["pos"] - prev_pos))
            prev_pos = st["pos"].copy()
            traj.append(prev_pos.copy())
            ret += self.cfg["r_progress"] * (prev_dist - st["dist"]) + self.cfg["r_step"]
            prev_dist = st["dist"]
            if steps % 10 == 0 or a.verbose:
                s = (f"ep {ep} step {steps:3d} pos ({st['pos'][0]:5.2f},{st['pos'][1]:5.2f}) dist {st['dist']:.2f} "
                     f"minL {st['min_lidar']:.2f} cmd ({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})")
                self.status_pub.publish(String(data=s))
                if a.verbose or steps % 30 == 0:
                    self.get_logger().info(s)
            if st["dist"] < self.cfg["goal_tolerance"]:
                outcome, ret = "success", ret + self.cfg["r_goal"]
                break
            if st["min_lidar"] < a.collision_dist:
                outcome, ret = "collision", ret + self.cfg["r_collision"]
                break
        self.stop()
        shortest = max(start_dist - self.cfg["goal_tolerance"], 1e-6)
        eff = min(1.0, shortest / max(path_len, 1e-6)) if outcome == "success" else 0.0
        row = dict(episode=ep, policy=self.policy_name, outcome=outcome, steps=steps, time_s=steps * dt,
                   ret=round(ret, 3), start_x=start[0], start_y=start[1], goal_x=goal[0], goal_y=goal[1],
                   start_dist=round(start_dist, 3), final_dist=round(st["dist"], 3), path_length=round(path_len, 3),
                   path_efficiency=round(eff, 3))
        writer.writerow(row)
        self.get_logger().info(f"ep {ep}: {outcome.upper()} in {steps} steps, final dist {st['dist']:.2f} m")
        if a.traj_dir:
            Path(a.traj_dir).mkdir(parents=True, exist_ok=True)
            np.save(os.path.join(a.traj_dir, f"ep{ep:03d}_{outcome}.npy"),
                    dict(traj=np.asarray(traj), goal=goal, outcome=outcome), allow_pickle=True)
        return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="SB3 .zip (SAC or PPO)")
    p.add_argument("--algo", choices=["sac", "ppo"], default=None)
    p.add_argument("--policy", choices=["model", "pctrl", "pctrl_avoid"], default="model")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--goal", type=float, nargs=2, default=None, help="fixed goal x y (default random)")
    p.add_argument("--start", type=float, nargs=2, default=None, help="fixed start x y (default random)")
    p.add_argument("--start-yaw", type=float, default=None)
    p.add_argument("--no-teleport", action="store_true", help="do not call gz set_pose; start where the robot is")
    p.add_argument("--world", default="env1_4x4_circles")
    p.add_argument("--model-name", default="ros_gz_sim_demos", help="Gazebo model name of the robot")
    p.add_argument("--scan-topic", default="/lidar")
    p.add_argument("--odom-topic", default="/odom")
    p.add_argument("--vel-obs", choices=["command", "measured"], default="command",
                   help="velocity feature: last commanded velocity (default, matches training) or odometry twist")
    p.add_argument("--pose-source", choices=["gz", "odom"], default="gz",
                   help="gz = ground-truth model pose from /world/<world>/dynamic_pose/info (default); "
                        "odom = the robot's /odom topic (unreliable: two plugins publish on it)")
    p.add_argument("--cmd-topic", default="/cmd_vel")
    p.add_argument("--collision-dist", type=float, default=DEFAULT_CONFIG["collision_dist"])
    p.add_argument("--lidar-floor", type=float, default=0.08)
    p.add_argument("--lidar-yaw-offset-deg", type=float, default=0.0,
                   help="rotate the scan if the sensor frame is not aligned with base_link")
    p.add_argument("--speed-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default=None, help="metrics CSV (default results_gazebo/<policy>.csv)")
    p.add_argument("--traj-dir", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.policy == "model" and not args.model:
        p.error("--model required unless --policy pctrl")

    rclpy.init()
    node = DRLInferenceNode(args)
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    node.get_logger().info("waiting for /lidar and /odom ...")
    t0 = time.time()
    while rclpy.ok() and (node.scan is None or node.odom_count == 0):
        time.sleep(0.1)
        if time.time() - t0 > 30:
            node.get_logger().error("no sensor data after 30 s. Is Gazebo running with the ros_gz bridge?")
            rclpy.shutdown()
            return 1
    node.get_logger().info(f"sensors OK (scan {len(node.scan)} rays)")

    csv_path = args.csv or f"results_gazebo/{node.policy_name.replace(':', '_').replace('.zip', '')}.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["episode", "policy", "outcome", "steps", "time_s", "ret", "start_x", "start_y", "goal_x", "goal_y",
              "start_dist", "final_dist", "path_length", "path_efficiency"]
    rows = []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        try:
            for ep in range(args.episodes):
                r = node.run_episode(ep, w)
                f.flush()
                if r is None:
                    break
                rows.append(r)
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    node.stop()
    if rows:
        n = len(rows)
        succ = [r for r in rows if r["outcome"] == "success"]
        print("\n========== GAZEBO RESULTS ==========")
        print(f"policy        : {node.policy_name}")
        print(f"episodes      : {n}")
        print(f"success rate  : {len(succ)/n*100:.1f} %")
        print(f"collision rate: {sum(r['outcome']=='collision' for r in rows)/n*100:.1f} %")
        print(f"timeout rate  : {sum(r['outcome']=='timeout' for r in rows)/n*100:.1f} %")
        if succ:
            print(f"avg steps (success): {np.mean([r['steps'] for r in succ]):.1f}  "
                  f"({np.mean([r['time_s'] for r in succ]):.1f} s)")
            print(f"path efficiency    : {np.mean([r['path_efficiency'] for r in succ]):.2f}")
        print(f"csv -> {csv_path}")
        print("====================================")
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

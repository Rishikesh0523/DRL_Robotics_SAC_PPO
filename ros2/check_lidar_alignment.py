#!/usr/bin/env python3
"""Sanity check: does the Gazebo LiDAR binning match the 2-D simulator?

Teleports the robot to a few known poses, reads the real ``/lidar`` scan and
``/odom`` pose, bins into 10 sectors exactly like the inference node, and
prints them next to the sectors the 2-D simulator predicts for the same pose.
Large mismatches in *which sector* is short indicate a rotated sensor frame
(fix with ``--lidar-yaw-offset-deg`` on the inference node).

    python3 check_lidar_alignment.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rclpy  # noqa: E402

from drl_inference_node import DRLInferenceNode  # noqa: E402
from nav2d import Nav2DEnv  # noqa: E402

POSES = [(3.3, 0.0, 0.0), (0.0, 3.3, 0.0), (-3.3, 0.0, math.pi / 2), (0.0, 0.0, 0.0), (-2.0, -2.0, math.pi / 4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="env1_4x4_circles")
    p.add_argument("--model-name", default="ros_gz_sim_demos")
    p.add_argument("--lidar-yaw-offset-deg", type=float, default=0.0)
    a = p.parse_args()
    ns = argparse.Namespace(policy="pctrl", model=None, algo=None, scan_topic="/lidar", odom_topic="/odom",
                            cmd_topic="/cmd_vel", lidar_floor=0.08, lidar_yaw_offset_deg=a.lidar_yaw_offset_deg,
                            speed_scale=1.0, seed=0, world=a.world, model_name=a.model_name, pose_source="gz", interactive=False, publish_tf=False,
                            vel_obs="command")
    rclpy.init()
    node = DRLInferenceNode(ns)
    import threading
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    t0 = time.time()
    while node.scan is None or node.odom_count == 0:
        time.sleep(0.1)
        if time.time() - t0 > 30:
            print(f"no sensor data (scan={node.scan is not None}, pose msgs={node.odom_count} on {node.pose_topic})")
            rclpy.shutdown()
            return 1
    env = Nav2DEnv({"lidar_noise_std": 0.0})
    worst = 0.0
    for (x, y, yaw) in POSES:
        node.stop()
        node.teleport(x, y, yaw)
        node.wait_for_pose(x, y)
        time.sleep(0.6)
        with node.lock:
            node.scan = None
        while node.scan is None:
            time.sleep(0.02)
        obs, st = node.observation(np.array([0.0, 0.0]))
        env.reset(options={"start": (st["pos"][0], st["pos"][1]), "yaw": st["yaw"], "goal": (0.0, 0.0)})
        sim = env.sectors_from_scan(env._scan())
        gz = st["sectors"]
        err = np.abs(np.minimum(gz, 4.0) - np.minimum(sim, 4.0))
        worst = max(worst, float(err.max()))
        print(f"\npose requested ({x:+.2f},{y:+.2f},{math.degrees(yaw):+.0f}deg)  "
              f"odom ({st['pos'][0]:+.2f},{st['pos'][1]:+.2f},{math.degrees(st['yaw']):+.0f}deg)")
        print("  gazebo  :", np.array2string(gz, precision=2, max_line_width=200))
        print("  sim2d   :", np.array2string(sim, precision=2, max_line_width=200))
        print("  |diff|  :", np.array2string(err, precision=2, max_line_width=200))
    print(f"\nworst sector mismatch (capped at 4 m): {worst:.2f} m  ->  "
          f"{'OK, frames aligned' if worst < 0.5 else 'MISMATCH: check sensor orientation / offset'}")
    node.stop()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

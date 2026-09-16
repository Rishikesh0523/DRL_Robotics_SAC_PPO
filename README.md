# DRL Mobile-Robot Navigation: standalone trainer + Gazebo demo

This is the updated version of [Rishikesh0523/DRL_Robotics](https://github.com/Rishikesh0523/DRL_Robotics).
That repository holds the ROS 2 / Gazebo workspace (robot model, world, first SAC experiments); this one adds
the fast 2-D trainer, domain randomisation, the SAC-vs-PPO comparison and the Gazebo deployment node.

Self-contained package that trains **SAC** and **PPO** navigation policies for the mecanum
robot of `DRL_Robotics` *without* ROS 2 or Gazebo (a fast 2-D simulator that reproduces the
4 x 4 m arena, the 360-ray LiDAR and the 16-D observation of `robot_nav_env2.py`), compares
them against a P-controller baseline, and then runs the trained policies on the real Gazebo
robot through a ROS 2 inference node.

```
drl_nav_standalone/
  nav2d/                 2-D simulator (Gymnasium env), arena geometry, ray casting, P-controller, plots
  train.py               train SAC or PPO (Stable-Baselines3) with TensorBoard + eval logging
  evaluate.py            evaluate a model / baseline: success, collision, SPL, path efficiency, GIF
  compare.py             aggregate runs -> learning curves, bar charts, comparison.md
  models/                shipped policies: sac_best.zip, ppo_best.zip (+ ppo_nominal_best.zip ablation)
  results/               2-D results: comparison.md, learning_curves.png, trajectories, GIFs
  results_gazebo/        Gazebo results measured on the cloud VM (CSV + summary + trajectory plots)
  runs/                  training logs of the shipped runs (evaluations.npz, monitor.csv, config.json)
  ros2/
    gazebo_demo.launch.py   world + robot + bridge (gui:=true|false)
    drl_inference_node.py   runs a policy on the Gazebo robot, teleport resets, metrics CSV
    check_lidar_alignment.py  verifies Gazebo LiDAR binning == simulator binning
    summarize_gazebo_results.py
  scripts/
    demo_gazebo.sh          one-command demo (launch Gazebo + run policy)
    run_all_training.sh     reproduce the whole experiment
    evaluate_all.sh         re-evaluate everything in 2-D and rebuild the report
    make_release.sh         build the tarball
```

---

## 1. Quick start on the VirtualBox machine (Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic)

```bash
# 0. unpack next to your ROS workspace
tar xzf drl_nav_standalone_*.tgz && cd drl_nav_standalone

# 1. Python deps (system python is fine; ROS 2 needs it). Torch CPU build is enough.
pip3 install --break-system-packages -r requirements.txt
#    (if pip refuses: python3 -m venv --system-site-packages .venv && source .venv/bin/activate && pip install -r requirements.txt)

# 2. sanity check without ROS: evaluate the shipped models in the 2-D simulator (takes ~1 min)
python3 evaluate.py --model models/sac_best.zip --episodes 100
python3 evaluate.py --model models/ppo_best.zip --episodes 100
python3 evaluate.py --policy pctrl_avoid       --episodes 100
#    -> results/eval_*.json and results/eval_*_traj.png

# 3. make sure the DRL_Robotics workspace is built and sourced
source /opt/ros/jazzy/setup.bash
source ~/gz_ws/install/setup.bash          # workspace that contains src/ros_gz_sim_demos
#    (if the build fails with "urdf_tutorial not found":  sudo apt install ros-jazzy-urdf-tutorial)

# 4. Gazebo demo (GUI).  Runs 5 episodes with random start/goal, teleporting between episodes.
bash scripts/demo_gazebo.sh sac 5
bash scripts/demo_gazebo.sh ppo 5
bash scripts/demo_gazebo.sh pctrl 5
GOAL="2.5 2.5" bash scripts/demo_gazebo.sh sac 3     # fixed goal as in the report

# 5. summary table + trajectory plots of the Gazebo runs
python3 ros2/summarize_gazebo_results.py results_gazebo
```

Manual alternative to step 4 (two terminals, both with ROS 2 + workspace sourced):

```bash
ros2 launch ros2/gazebo_demo.launch.py                  # terminal 1 (add rviz:=true if you want RViz)
python3 ros2/drl_inference_node.py --model models/sac_best.zip --episodes 5 --traj-dir results_gazebo/traj_sac   # terminal 2
```

Useful flags of the inference node: `--goal X Y`, `--start X Y`, `--start-yaw RAD`, `--no-teleport`
(start from wherever the robot is), `--speed-scale 0.7`, `--verbose`.

### Interactive demo: click the start and the goal

Gazebo's own GUI has no click-to-pose tool, so the interactive mode uses RViz next to Gazebo.
RViz shows the arena, the LiDAR scan, the robot (at its true pose), the goal disc and the driven path.

```bash
bash scripts/demo_interactive.sh            # SAC (or: ppo / pctrl)
SPEED=0.7 bash scripts/demo_interactive.sh  # slower robot, recommended in VirtualBox
```

In the RViz window:
1. Click **2D Pose Estimate** in the toolbar, then click on the map and drag to set the heading.
   The robot teleports there in Gazebo.
2. Click **2D Goal Pose**, then click (and drag) where the robot should go. The episode starts immediately
   and the terminal prints the outcome.
3. Repeat as often as you like. Goals inside a wall or pillar are rejected with a warning. Ctrl+C quits.

Manual equivalent: `ros2 launch ros2/gazebo_demo.launch.py rviz:=true` in one terminal and
`python3 ros2/drl_inference_node.py --model models/sac_best.zip --interactive` in another.

---

## 2. Results

All numbers below were produced overnight on a GCP VM (n1-standard-8 + Tesla T4, Ubuntu 24.04, ROS 2 Jazzy,
Gazebo Harmonic 8.15).  Full tables: `results/comparison.md` (2-D) and `results_gazebo/gazebo_summary.md`.

### 2.1 Training (domain randomisation on, random start + goal)

| algorithm | seeds | env steps / seed | wall-clock / seed | success at end of training (2-D, 100 eps) |
|---|---|---|---|---|
| **SAC** (SB3, 2x256, 1 update / 2 steps) | 4 | 300 k | 41-63 min | 98-100 % |
| **PPO** (SB3, 2x256, 8 envs) | 3 | 2 M | 47 min | 93-99 % |

* SAC reaches ~95 % success after **~70 k** steps; PPO needs ~200 k steps for the same and stays noisier
  (see `results/learning_curves.png`).  PPO is ~9x more sample-hungry but 9x cheaper per step, so both
  need ~45-60 min of wall-clock here.
* Compare with the report: 26 k Gazebo steps took ~2.4 h at 3 FPS and ended at 60 % success (5 episodes).
  The 2-D simulator runs at ~4 700 steps/s, so 300 k steps of experience cost minutes instead of ~28 h.

### 2.2 2-D simulator evaluation (200 random start/goal episodes, deterministic policy)

| policy | dynamics | success | collision | timeout | steps to goal | path eff. | SPL |
|---|---|---|---|---|---|---|---|
| random actions | nominal | 1.0 % | 52.5 % | 46.5 % | - | - | 0.00 |
| P-controller (report baseline) | nominal | 61.5 % | 38.5 % | 0 % | 43.8 | 0.99 | 0.61 |
| P-controller + LiDAR repulsion | nominal | 97.0 % | 0 % | 3.0 % | 60.4 | 0.95 | 0.92 |
| PPO trained on nominal dynamics (ablation) | nominal | 98.5 % | 1.5 % | 0 % | 31.8 | 0.91 | 0.90 |
| PPO trained on nominal dynamics (ablation) | perturbed | 90.5 % | 9.5 % | 0 % | 50.1 | 0.74 | 0.67 |
| **PPO-DR** (`models/ppo_best.zip`) | nominal | 96.0 % | 3.5 % | 0.5 % | 35.5 | 0.87 | 0.83 |
| **PPO-DR** | perturbed | 95.0 % | 4.5 % | 0.5 % | 48.4 | 0.82 | 0.78 |
| **SAC-DR** (`models/sac_best.zip`) | nominal | 99.5 % | 0.5 % | 0 % | 40.3 | 0.85 | 0.85 |
| **SAC-DR** | perturbed | 99.0 % | 1.0 % | 0 % | 48.0 | 0.84 | 0.83 |
| SAC-DR, fixed goal (2.5, 2.5) as in the report | nominal | 100 % (50 eps) | 0 % | 0 % | 46.3 | 0.82 | 0.82 |
| PPO-DR, fixed goal (2.5, 2.5) | nominal | 96 % (50 eps) | 4 % | 0 % | 43.7 | 0.84 | 0.81 |

Across seeds: SAC-DR 99.5-100 % (4 seeds), PPO-DR 96-98.5 % (3 seeds).  One step = 0.12 s.
Plots: `results/comparison_bars.png`, `results/eval_*_traj.png`, GIFs `results/sac_demo.gif`, `results/ppo_demo.gif`.

### 2.3 Gazebo evaluation (real robot model, headless Gazebo Harmonic, 20 episodes, identical start/goal sequence)

| policy | success | collision | timeout | steps to goal | time to goal | path eff. |
|---|---|---|---|---|---|---|
| P-controller (report baseline) | 40 % | 25 % | 35 % | 137.6 | 16.5 s | 0.51 |
| P-controller + LiDAR repulsion | 55 % | 0 % | 45 % | 106.0 | 12.7 s | 0.55 |
| PPO trained on nominal dynamics (ablation) | 5 % | 75 % | 20 % | 152.0 | 18.2 s | 0.06 |
| **PPO-DR** (`models/ppo_best.zip`) | 65 % | 30 % | 5 % | 95.2 | 11.4 s | 0.44 |
| PPO-DR at 70 % speed (`--speed-scale 0.7`) | 75 % | 15 % | 10 % | 63.4 | 7.6 s | 0.71 |
| **SAC-DR** (`models/sac_best.zip`) | **90 %** | 10 % | 0 % | **44.2** | **5.3 s** | **0.69** |
| SAC-DR, fixed goal (2.5, 2.5), 10 eps | 90 % | 10 % | 0 % | 58.2 | 7.0 s | 0.61 |
| PPO-DR, fixed goal (2.5, 2.5), 10 eps | 90 % | 10 % | 0 % | 80.0 | 9.6 s | 0.45 |

Seed sweep in Gazebo (10 episodes each): SAC-DR seeds 80 / **100** / 90 / 100 %, PPO-DR seeds 60 / **70** / 20 %.
The bold seeds are shipped.  Trajectory plots: `results_gazebo/gazebo_traj_*.png`.

### 2.4 Take-aways
1. **SAC > PPO for this task.**  Same success in the 2-D simulator, but SAC transfers far better to Gazebo
   (90 % vs 65 %), reaches goals 2x faster and drives smoother paths.  PPO's paths are erratic
   (path efficiency 0.44) and it exploits the maximum speed; slowing it to 70 % recovers 10 points.
2. **Domain randomisation is what makes sim-to-Gazebo work.**  Nominal-dynamics PPO: 98.5 % in 2-D -> 5 % in
   Gazebo.  The randomised recipe: 96 % -> 65 % (PPO) and 99.5 % -> 90 % (SAC).
3. **Both DRL policies beat the report's P-controller baseline in Gazebo** (40 %); a P-controller with
   LiDAR repulsion never collides but stalls in front of pillars (45 % timeouts).
4. Remaining Gazebo failures of SAC-DR are 2 collisions when skimming a pillar at speed; a goal tolerance of
   0.3 m is met in 44 steps (5.3 s) on average from 3.8 m away.

---

## 3. What was built and why

### 3.1 The 2-D simulator (`nav2d`)
* **Arena** copied from `worlds/world.sdf`: walls at +-4 m (inner face 3.95 m), four pillars r = 0.20 m at
  (-1.0, 1.5), (1.3, 1.0), (-1.4, -0.8), (0.8, -1.6).
* **LiDAR** as in `lidar.gazebo`: 360 rays, angle_min = -pi, range 0.1-10 m, Gaussian noise; mounted
  6 cm behind the robot centre (URDF).  Binned into 10 sector minima exactly like `robot_nav_env2.py`
  (sector 0 = rear-right, sector 5 = front-left).  `ros2/check_lidar_alignment.py` teleports the Gazebo robot
  to five poses and compares: worst mismatch **0.09 m**, so scans are interchangeable.
* **Observation (16-D)**: 10 sectors [m], distance to goal [m], bearing [rad], last commanded (vx, vy, w) and
  |v|.  The velocity part uses the *command* rather than measured odometry (see 3.3).
* **Action (3-D)** in [-1, 1], scaled to vx, vy in +-0.75 m/s and w in +-1.5 rad/s, published as a Twist.
* **Dynamics**: holonomic kinematics, first-order velocity tracking with acceleration limits (MecanumDrive
  plugin: 5 m/s^2), 4 physics sub-steps per 0.12 s control step, body radius 0.17 m.
* **Episode**: random start pose and random goal (>= 1.5 m apart, >= 0.45 m clearance) unless `--fixed-goal`;
  success < 0.30 m; collision = LiDAR min < 0.20 m or body contact; timeout 300 steps.
* **Reward**: 5 x progress, +20 goal, -20 collision, -0.05 per step, proximity penalty below 0.5 m,
  small bonus for velocity toward the goal and a small action-smoothness penalty (report: wobbly paths).
* Speed: ~4 700 random steps/s single core; SAC trains at 80-160 steps/s (update-bound), PPO at 600-1 700 steps/s.

### 3.2 Domain randomisation (`--dr`)
Per episode the simulator samples: forward / lateral / yaw speed efficiency (0.6-1.0 / 0.4-1.0 / 0.7-1.0),
backward drift proportional to lateral speed (0-0.5), acceleration limits, 0-2 steps of observation delay,
0-1 step of action delay, control period 0.09-0.15 s, LiDAR noise 0.005-0.03 m.  The ranges come from
measurements on the Gazebo robot (section 3.3).  **Without it, a policy that scores 97 % in the 2-D
simulator collided in 6/6 Gazebo episodes**; with it the same training recipe transfers (section 2).

### 3.3 Findings on the Gazebo side (all handled in `ros2/`)
1. **Dual odometry.**  `/odom` is written by two plugins in `mech_mobile.gazebo`: MecanumDrive
   (dead-reckoning from the spawn point, ignores teleports) and OdometryPublisher (world pose).  Readings
   alternate between frames, exactly the "dual-odometry problem" in the report.  The node reads
   `/model/ros_gz_sim_demos/odometry_with_covariance`, which only OdometryPublisher writes and which follows
   teleports; `gazebo_demo.launch.py` bridges it.
2. **Lateral slip.**  A pure `linear.y = 0.5` command moves the robot at roughly half that speed and drags
   it backwards ~0.37 m per metre of strafing (mecanum wheel contact model).  Modelled by the randomisation.
3. **Velocity feature.**  Measured twist depends on plugin, frame and latency; the last *commanded*
   velocity is identical in simulation and on the robot, so both use it (the original `final_sac.py` did too).
4. **Stale observations after teleport.**  The node waits until the pose source confirms the new pose and a
   fresh scan arrived (fixes the "stale observations after reset" bug from the report).
5. **LiDAR floor.**  Readings < 0.08 m are treated as self-hits and set to max range, as in the report.

### 3.4 Assumptions made (you were asleep)
* Trained on **random goals** (the report used a fixed goal at (2.5, 2.5)); fixed-goal numbers are also
  reported and the demo accepts `GOAL="2.5 2.5"`.
* Goal tolerance 0.30 m and collision threshold 0.20 m (report: 0.3 m success, 0.15 m collision; 0.20 m
  is safer because the body extends ~0.17 m from the LiDAR).
* The camera is not used (LiDAR-only, like the run in the report).
* SAC uses one gradient step per two environment steps (`--sac-train-freq 2`) to fit the time budget.

---

## 4. Reproduce / retrain

```bash
bash scripts/run_all_training.sh                       # 3 seeds SAC (300k) + 3 seeds PPO (2M), DR on
DR=0 bash scripts/run_all_training.sh                  # nominal-dynamics ablation
tensorboard --logdir runs                              # live curves
bash scripts/evaluate_all.sh 200                       # 2-D evaluation of everything + results/comparison.md
python3 evaluate.py --model runs/sac_dr_seed0/best_model.zip --fixed-goal --gif results/demo.gif
```

`train.py --help` lists all knobs (`--steps`, `--seed`, `--n-envs`, `--env-cfg '{"goal_tolerance":0.5}'`, ...).

## 5. Troubleshooting
* *ModuleNotFoundError: No module named 'numpy._core...'* when loading a model: the shipped models were pickled
  with NumPy 2 -> `pip3 install --break-system-packages "numpy>=2"` (do **not** downgrade numpy).
* *"A module that was compiled using NumPy 1.x cannot be run in NumPy 2"*: Ubuntu's apt matplotlib is too old ->
  `pip3 install --break-system-packages --upgrade "matplotlib>=3.9"` (requirements.txt does this).
  cv_bridge from apt has the same issue but is not used by this package.
* *no sensor data after 30 s*: Gazebo is not running, or the bridge is missing -> use `ros2/gazebo_demo.launch.py`
  (it bridges `/lidar`, `/odom`, `/cmd_vel`, `/clock`, `/tf` and the clean odometry topic).
* *set_pose reply: ... false*: the model name differs -> `--model-name <name>` (see `gz model --list`).
* *robot spins / misses obstacles*: run `python3 ros2/check_lidar_alignment.py`; if sectors are shifted,
  pass `--lidar-yaw-offset-deg` to the node.
* *slow Gazebo in VirtualBox*: the policy runs at 0.12 s per step in wall time; if the real-time factor is
  far below 1, use `--speed-scale 0.7`.
* GUI missing in VirtualBox: `GUI=false bash scripts/demo_gazebo.sh sac 5` still produces metrics.

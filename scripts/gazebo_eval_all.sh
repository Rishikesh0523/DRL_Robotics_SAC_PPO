#!/usr/bin/env bash
# Evaluate the shipped models and baselines on the Gazebo robot (headless by default).
#   GUI=false EPISODES=20 bash scripts/gazebo_eval_all.sh
# Needs ROS 2 Jazzy + the DRL_Robotics workspace sourced (see README).
set -uo pipefail
cd "$(dirname "$0")/.."
EPISODES=${EPISODES:-20}
GUI=${GUI:-false}
SEED=${SEED:-7}
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}

pkill -f "gz sim" 2>/dev/null; pkill -f parameter_bridge 2>/dev/null; sleep 1
ros2 launch ros2/gazebo_demo.launch.py gui:=$GUI > /tmp/gazebo_eval_launch.log 2>&1 &
LAUNCH_PID=$!
trap 'kill $LAUNCH_PID 2>/dev/null; pkill -f "gz sim" 2>/dev/null || true' EXIT
for i in $(seq 1 60); do timeout 2 ros2 topic echo /lidar --once >/dev/null 2>&1 && break; sleep 1; done
mkdir -p results_gazebo

run() {  # name, args...
  local name=$1; shift
  echo "=== $name"
  timeout 3600 python3 ros2/drl_inference_node.py "$@" --episodes "$EPISODES" --seed "$SEED" \
      --csv "results_gazebo/$name.csv" --traj-dir "results_gazebo/traj_$name" 2>&1 | grep -E "ep [0-9]+:|rate|steps|effic"
}
run pctrl_avoid --policy pctrl_avoid
run pctrl       --policy pctrl
[ -f models/ppo_nominal_best.zip ] && run ppo_nominal --model models/ppo_nominal_best.zip --algo ppo
[ -f models/sac_best.zip ] && run sac_dr --model models/sac_best.zip --algo sac
[ -f models/ppo_best.zip ] && run ppo_dr --model models/ppo_best.zip --algo ppo
# fixed goal as in the report (fewer episodes)
[ -f models/sac_best.zip ] && EPISODES=10 run sac_dr_fixedgoal --model models/sac_best.zip --algo sac --goal 2.5 2.5
[ -f models/ppo_best.zip ] && EPISODES=10 run ppo_dr_fixedgoal --model models/ppo_best.zip --algo ppo --goal 2.5 2.5
python3 ros2/summarize_gazebo_results.py results_gazebo

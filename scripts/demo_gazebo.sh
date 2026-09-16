#!/usr/bin/env bash
# One-command Gazebo demo for the VirtualBox / Ubuntu 24.04 + ROS 2 Jazzy machine.
#
#   bash scripts/demo_gazebo.sh                       # SAC model, 5 random episodes, Gazebo GUI
#   bash scripts/demo_gazebo.sh ppo 10                # PPO model, 10 episodes
#   bash scripts/demo_gazebo.sh pctrl 5               # P-controller baseline
#   GUI=false bash scripts/demo_gazebo.sh sac 20      # headless (cloud / CI)
#   GOAL="2.5 2.5" bash scripts/demo_gazebo.sh sac 3  # fixed goal as in the report
#
# Prerequisites (once):  source /opt/ros/jazzy/setup.bash ; source ~/gz_ws/install/setup.bash
#   (gz_ws = the colcon workspace containing DRL_Robotics/src/ros_gz_sim_demos)
set -euo pipefail
cd "$(dirname "$0")/.."
POLICY=${1:-sac}
EPISODES=${2:-5}
GUI=${GUI:-true}
GOAL=${GOAL:-}
WS=${WS:-$HOME/gz_ws}

if ! command -v ros2 >/dev/null; then
  source /opt/ros/jazzy/setup.bash
fi
if ! ros2 pkg prefix ros_gz_sim_demos >/dev/null 2>&1; then
  if [ -f "$WS/install/setup.bash" ]; then source "$WS/install/setup.bash"; else
    echo "ros_gz_sim_demos not found. Build DRL_Robotics (colcon build) and set WS=<workspace>"; exit 1; fi
fi

case "$POLICY" in
  sac)   POLICY_ARGS="--model models/sac_best.zip" ;;
  ppo)   POLICY_ARGS="--model models/ppo_best.zip" ;;
  pctrl) POLICY_ARGS="--policy pctrl_avoid" ;;
  *)     POLICY_ARGS="--model $POLICY" ;;   # any .zip path
esac
GOAL_ARGS=""
[ -n "$GOAL" ] && GOAL_ARGS="--goal $GOAL"

echo ">> launching Gazebo (gui=$GUI) ..."
ros2 launch ros2/gazebo_demo.launch.py gui:=$GUI > /tmp/gazebo_demo_launch.log 2>&1 &
LAUNCH_PID=$!
trap 'echo ">> stopping Gazebo"; kill $LAUNCH_PID 2>/dev/null; pkill -f "gz sim" 2>/dev/null || true' EXIT

echo ">> waiting for /lidar ..."
for i in $(seq 1 60); do
  if timeout 2 ros2 topic echo /lidar --once >/dev/null 2>&1; then break; fi
  sleep 1
done

mkdir -p results_gazebo
echo ">> running policy: $POLICY  ($EPISODES episodes)"
python3 ros2/drl_inference_node.py $POLICY_ARGS $GOAL_ARGS --episodes "$EPISODES" \
    --csv "results_gazebo/${POLICY//\//_}.csv" --traj-dir "results_gazebo/traj_${POLICY//\//_}"
echo ">> done. metrics: results_gazebo/${POLICY//\//_}.csv"

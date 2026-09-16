#!/usr/bin/env bash
# Interactive demo: Gazebo + RViz; click the start and goal in RViz.
#
#   bash scripts/demo_interactive.sh            # SAC policy
#   bash scripts/demo_interactive.sh ppo        # PPO policy
#   SPEED=0.7 bash scripts/demo_interactive.sh  # slower robot (helps in VirtualBox)
#
# In the RViz window:
#   1. "2D Pose Estimate" (toolbar)  -> click + drag on the map: robot teleports there (drag sets heading)
#   2. "2D Goal Pose"                -> click + drag: goal is set and the episode starts immediately
#   Repeat 1-2 as often as you like. Ctrl+C in this terminal quits everything.
set -euo pipefail
cd "$(dirname "$0")/.."
POLICY=${1:-sac}
SPEED=${SPEED:-1.0}
WS=${WS:-$HOME/ros2_ws}

if ! command -v ros2 >/dev/null; then source /opt/ros/jazzy/setup.bash; fi
if ! ros2 pkg prefix ros_gz_sim_demos >/dev/null 2>&1; then
  if [ -f "$WS/install/setup.bash" ]; then source "$WS/install/setup.bash"; else
    echo "ros_gz_sim_demos not found. Build DRL_Robotics (colcon build) and set WS=<workspace>"; exit 1; fi
fi

case "$POLICY" in
  sac)   POLICY_ARGS="--model models/sac_best.zip" ;;
  ppo)   POLICY_ARGS="--model models/ppo_best.zip" ;;
  pctrl) POLICY_ARGS="--policy pctrl_avoid" ;;
  *)     POLICY_ARGS="--model $POLICY" ;;
esac

echo ">> launching Gazebo + RViz ..."
ros2 launch ros2/gazebo_demo.launch.py rviz:=true > /tmp/gazebo_demo_launch.log 2>&1 &
LAUNCH_PID=$!
trap 'echo ">> stopping"; kill $LAUNCH_PID 2>/dev/null; pkill -f "gz sim" 2>/dev/null || true; pkill -f rviz2 2>/dev/null || true' EXIT

echo ">> waiting for /lidar ..."
for i in $(seq 1 60); do
  if timeout 2 ros2 topic echo /lidar --once >/dev/null 2>&1; then break; fi
  sleep 1
done

mkdir -p results_gazebo
python3 ros2/drl_inference_node.py $POLICY_ARGS --interactive --speed-scale "$SPEED" \
    --csv "results_gazebo/interactive_${POLICY//\//_}.csv"

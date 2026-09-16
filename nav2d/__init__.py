"""nav2d: lightweight 2-D navigation simulator mirroring the Gazebo DRL arena.

The observation layout, action layout, arena geometry and LiDAR spec follow
`DRL_Robotics/src/drl_navigation/scripts/robot_nav_env2.py` and
`DRL_Robotics/src/ros_gz_sim_demos/worlds/world.sdf`, so policies trained here
can be dropped into the ROS 2 inference node unchanged.
"""

from .arena import Arena, DEFAULT_ARENA
from .env import Nav2DEnv, DEFAULT_CONFIG, make_env
from .policies import PController

__all__ = [
    "Arena",
    "DEFAULT_ARENA",
    "Nav2DEnv",
    "DEFAULT_CONFIG",
    "make_env",
    "PController",
]

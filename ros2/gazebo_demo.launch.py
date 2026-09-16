#!/usr/bin/env python3
"""Launch the 4x4 arena world + mecanum robot + ROS-Gazebo bridge for the DRL demo.

    ros2 launch gazebo_demo.launch.py                 # GUI
    ros2 launch gazebo_demo.launch.py gui:=false      # headless server (for CI / cloud)
    ros2 launch gazebo_demo.launch.py rviz:=true

Requires the ``ros_gz_sim_demos`` package from DRL_Robotics to be built and sourced
(it provides the world SDF, robot URDF and meshes).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_demo = get_package_share_directory("ros_gz_sim_demos")
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    world_path = os.path.join(pkg_demo, "worlds", "world.sdf")
    urdf_path = PathJoinSubstitution([FindPackageShare("ros_gz_sim_demos"), "urdf/robot/mech_mobile.urdf.xacro"])

    gui = LaunchConfiguration("gui")
    rviz = LaunchConfiguration("rviz")
    x0 = LaunchConfiguration("x")
    y0 = LaunchConfiguration("y")
    world_name = "env1_4x4_circles"  # <world name=...> in world.sdf

    args = [
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("x", default_value="-2.0"),
        DeclareLaunchArgument("y", default_value="-2.0"),
    ]

    # make sure Gazebo finds the package meshes (file://$(find ...) in the URDF)
    resource_env = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.pathsep.join(filter(None, [os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
                                      os.path.dirname(pkg_demo)])),
    )

    gz_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": f"-r -v 1 {world_path}", "on_exit_shutdown": "true"}.items(),
        condition=IfCondition(gui),
    )
    gz_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": f"-s -r -v 1 --headless-rendering {world_path}",
                          "on_exit_shutdown": "true"}.items(),
        condition=UnlessCondition(gui),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": Command([FindExecutable(name="xacro"), " ", urdf_path]),
                     "use_sim_time": True}],
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-topic", "robot_description", "-name", "ros_gz_sim_demos",
                   "-x", x0, "-y", y0, "-z", "0.05", "-Y", "0.0"],
        output="screen",
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model",
            # Clean pose source: only the OdometryPublisher plugin writes this topic (world-frame
            # pose that follows teleports).  /odom itself is written by BOTH the MecanumDrive
            # plugin (dead-reckoning from the spawn point) and OdometryPublisher -> the
            # "dual odometry" jumps described in the report.
            "/model/ros_gz_sim_demos/odometry_with_covariance@nav_msgs/msg/Odometry[gz.msgs.OdometryWithCovariance",
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    map_to_odom = Node(package="tf2_ros", executable="static_transform_publisher", name="map_to_odom",
                       arguments=["0", "0", "0", "0", "0", "0", "map", "odom"])
    lidar_frame_fix = Node(package="tf2_ros", executable="static_transform_publisher", name="lidar_frame_fix",
                           arguments=["0", "0", "0", "0", "0", "0", "lidar_sensor_link",
                                      "ros_gz_sim_demos/base_link/lidar_sensor"])

    rviz_node = Node(package="rviz2", executable="rviz2", parameters=[{"use_sim_time": True}],
                     condition=IfCondition(rviz))

    return LaunchDescription(args + [resource_env, gz_gui, gz_headless, robot_state_publisher, spawn, bridge,
                                     map_to_odom, lidar_frame_fix, rviz_node])

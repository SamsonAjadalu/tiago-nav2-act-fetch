# ROS 2 workspace (Nav2, MoveIt2 helpers, RViz)

**ROS 2 distro:** Humble. Source-only; run `colcon build` locally (`build/`, `install/`, `log/` are gitignored).

## Packages

| Package | Role |
|---------|------|
| `tiago_nav` | Nav2 bringup (`bringup.launch.py`), maps, `nav2_params.yaml`, AMCL defaults |
| `tiago_moveit_config` | MoveIt 2 demo launch (optional; used with Isaac / sim clock) |
| `tiago_rviz2` | Colored RGB–depth point cloud (`rgb_pointcloud.py` → `gemini2/rgbpoints`) |

Install system dependencies (from workspace root):

```bash
sudo apt update
rosdep install --from-paths src --ignore-src -r -y
```

## Build

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Nav2 bringup (namespaced `tiago`)

Example (tune `initial_pose_*` and `map` for your scene):

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=68   # match Isaac / bridge if used

ros2 launch tiago_nav bringup.launch.py \
  namespace:=tiago \
  use_sim_time:=true \
  use_clock_fallback:=true \
  clock_source_topic:=/tiago/clock
```

Defaults for map (`maps/tiago_house.yaml`) and AMCL `initial_pose_*` are in `launch/bringup.launch.py`. Override `map:=...` or `initial_pose_x:=...` as needed. For Isaac Sim, align `initial_pose_*` with the robot in the map frame.

## RGB point cloud (Gemini2)

```bash
ros2 launch tiago_rviz2 rgb_pointcloud.py
```

In RViz2 add **PointCloud2** on topic `gemini2/rgbpoints`.

## MoveIt demo (optional)

```bash
ros2 launch tiago_moveit_config demo.launch.py \
  start_move_group:=false use_rviz:=true mirror_joint_states:=false \
  start_robot_state_publisher:=false start_odom_static_tf:=false \
  use_sim_time:=true relay_sim_clock:=true
```

Adjust arguments to match your robot state publisher / sim setup.

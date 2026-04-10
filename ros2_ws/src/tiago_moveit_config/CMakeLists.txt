cmake_minimum_required(VERSION 3.8)
project(tiago_moveit_config)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/moveit_py_smoke_test.py
  scripts/tiago_moveit_py_grasp_near.py
  scripts/tiago_clock_relay.py
  scripts/fake_joint_state_publisher.py
  scripts/isaac_joint_trajectory_bridge.py
  scripts/isaac_topics_stub.py
  scripts/run_moveit_demo.sh
  scripts/run_tiago_moveit_py_isaac.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

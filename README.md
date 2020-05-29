# escher_openrave_cpp
OpenRAVE C++ plugin for Escher Robot to run contact planner and experiment for the paper: "Robust Humanoid Contact Planning with Learned Zero- and One-Step Capturability Prediction" in RA-L 2020.

`escher_cpp_motion_planner_interface.py` is the main testing script used to run experiments. It will call the corresponding functions in `EscherMotionPlanning.cpp` through OpenRAVE cpp interface with configurations shown in `escher_openrave_cpp_wrapper.py`. The contact planner is described in `ContactSpacePlanning.cpp`.

# Dependency

* [OpenRAVE 0.9.0](https://github.com/rdiankov/openrave/tree/v0.9.0)
* ROS Melodic/Kinetic/Indigo
* [frugally-deep](https://github.com/Dobiasd/frugally-deep)
* Tensorflow 1.40
* [comps](https://github.com/UM-ARM-Lab/comps)
* [SL](https://github.com/UM-ARM-Lab/SL_and_momopt)

# Setup

1. Install ROS, OpenRAVE, Tensorflow (with C++ API), and frugally-deep
2. Create a catkin workspace, and put `escher_openrave_cpp`, comps and SL under the same catkin workspace.
3. `catkin_make` to build the code.

# Usage

* `surface_source`: The source of the environment in planning. The user can create new environment by adding new options in `update_environment` function in `environment_handler_2.py`.<br/>
  `capture_test_env_3`: Narrow flat corridor environment.<br/>
  `capture_test_env_4`: One-wall rubble environment.<br/>
  `capture_test_env_5`: Oil platform environment.<br/>
  `load_from_data`: Load environment object file stored using `pickle` from path specified by `environment_path` parameter.

* `environment_path`: The folder which contains the stored environment object file.
* `start_env_id` and `end_env_id`: The first and last environment object file id loaded in the process.


Example Usage:
```
python escher_cpp_motion_planner_interface.py surface_source capture_test_env_4 start_env_id 0 end_env_id 0
```

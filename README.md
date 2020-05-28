# escher_openrave_cpp
OpenRAVE C++ plugin for Escher Robot

`escher_cpp_motion_planner_interface.py` is the main testing script used to run experiments. It will call the corresponding functions in `EscherMotionPlanning.cpp` through OpenRAVE cpp interface with configurations shown in `escher_openrave_cpp_wrapper.py`. The contact planner is described in `ContactSpacePlanning.cpp`.

Dependency: OpenRAVE 0.9.0, ROS Melodic/Kinetic/Indigo, frugally-deep, tensorflow 1.40, cbirrt and SL.

cmake_minimum_required(VERSION 2.8.3)
project(escher_openrave_cpp)

set(CMAKE_CXX_FLAGS "-std=c++0x -O3 -fopenmp ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS Eigen roslib urdf srdfdom openrave_catkin momentumopt momentumopt_athena visualization_tools_athena floating_base_tools_athena floating_base_tools_sl_athena generalik)

find_package(Boost REQUIRED COMPONENTS regex filesystem system)
find_package(OpenRAVE REQUIRED)
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  message(STATUS "OPENMP FOUND")
  set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})
  set(OpenMP_LIBRARIES gomp)
endif()

catkin_package()

catkin_python_setup()

include_directories(
    include
    ${Eigen_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${OpenRAVE_INCLUDE_DIRS}
)
link_directories(
    ${Boost_LIBRARY_DIRS}
    ${catkin_LIBRARY_DIRS}
    ${OpenRAVE_LIBRARY_DIRS}
)

openrave_plugin("${PROJECT_NAME}_plugin"
    src/Utilities.cpp
    src/TrimeshSurface.cpp
    src/PointGrid.cpp
    src/ContactPoint.cpp
    src/ContactRegion.cpp
    src/SurfaceContactPointGrid.cpp
    src/MapGrid.cpp
    src/ContactState.cpp
    src/EscherMotionPlanning.cpp
    src/Drawing.cpp
    src/Boundary.cpp
    src/RobotProperties.cpp
    src/ContactSpacePlanning.cpp
    src/OptimizationInterface.cpp
    src/GeneralIKInterface.cpp
    src/GIWC.cpp
)

target_link_libraries("${PROJECT_NAME}_plugin"
    qhull
    newmat
    cddgmp
    gmp
    ${catkin_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OpenMP_LIBRARIES}
)

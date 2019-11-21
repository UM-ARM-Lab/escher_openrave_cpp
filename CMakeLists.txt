cmake_minimum_required(VERSION 2.8.3)
project(escher_openrave_cpp)

set(CMAKE_CXX_FLAGS "-std=c++1y -O3 ${CMAKE_CXX_FLAGS}")
# set(CMAKE_CXX_FLAGS "-std=c++1y -O3 -fopenmp ${CMAKE_CXX_FLAGS}")
# set(CMAKE_CXX_FLAGS "-std=c++0x -O3 -fopenmp ${CMAKE_CXX_FLAGS}")

# find_package(catkin REQUIRED COMPONENTS roslib urdf srdfdom openrave_catkin momentumopt momentumopt_athena visualization_tools_athena floating_base_tools_athena floating_base_tools_sl_athena generalik yaml_cpp tensorflow_ros_cpp)
find_package(catkin REQUIRED COMPONENTS roslib urdf srdfdom openrave_catkin momentumopt momentumopt_athena visualization_tools_athena floating_base_tools_athena floating_base_tools_sl_athena generalik yaml_cpp)

find_package(Boost REQUIRED COMPONENTS regex filesystem system)
find_package(OpenRAVE REQUIRED)
find_package(OpenMP REQUIRED)

SET( Tensorflow_LIBRARY_DIRS "/home/linyuchi/tensorflow/bazel-bin/tensorflow" )
# SET( Tensorflow_INCLUDE_DIRS "/home/linyuchi/tensorflow/bazel-genfiles;/home/linyuchi/tensorflow" )
SET( Tensorflow_INCLUDE_DIRS "/home/linyuchi/tensorflow/bazel-genfiles;/home/linyuchi/tensorflow;/home/linyuchi/tensorflow/bazel-tensorflow/external/protobuf_archive/src;/home/linyuchi/tensorflow/bazel-tensorflow/external/eigen_archive;/home/linyuchi/tensorflow/bazel-tensorflow/external/nsync/public;/home/linyuchi/tensorflow/bazel-tensorflow/external/com_google_absl" )

# SET( Eigen_INCLUDE_DIRS "/usr/local/include/eigen3" ) # compiled with version 3.3.5
# SET( Eigen_INCLUDE_DIRS "/home/linyuchi/tensorflow/external/eigen_archive" ) # compiled with version 3.3.90
IF( NOT Eigen_INCLUDE_DIRS )
    find_package(Eigen REQUIRED)
ENDIF()

if(OPENMP_FOUND)
  message(STATUS "OPENMP FOUND")
  set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})
  set(OpenMP_LIBRARIES gomp)
endif()

catkin_package()

catkin_python_setup()

include_directories(
    # ${tensorflow_ros_cpp_INCLUDE_DIRS}
    ${Tensorflow_INCLUDE_DIRS}
    include
)

# /home/linyuchi/tensorflow/bazel-genfiles /home/linyuchi/tensorflow /home/linyuchi/tensorflow/bazel-tensorflow/external/protobuf_archive/src/home/linyuchi/tensorflow/bazel-tensorflow/external/eigen_archive/home/linyuchi/tensorflow/bazel-tensorflow/external/nsync/public/home/linyuchi/tensorflow/bazel-tensorflow/external/com_google_absl

include_directories(
    SYSTEM
    # ${Tensorflow_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${OpenRAVE_INCLUDE_DIRS}
)
link_directories(
    ${Tensorflow_LIBRARY_DIRS}
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
    src/TorsoPathPlanning.cpp
    src/EscherMotionPlanning.cpp
    src/Drawing.cpp
    src/Boundary.cpp
    src/RobotProperties.cpp
    src/ContactSpacePlanning.cpp
    src/OptimizationInterface.cpp
    src/GeneralIKInterface.cpp
    src/GIWC.cpp
    src/NeuralNetworkInterface.cpp
)

target_link_libraries("${PROJECT_NAME}_plugin"
    qhull
    newmat
    cddgmp
    gmp
    ${Tensorflow_LIBRARY_DIRS}/libtensorflow_cc.so
    /usr/local/lib/libprotobuf.so
    ${Tensorflow_LIBRARY_DIRS}/libtensorflow_framework.so.1
    # ${tensorflow_ros_cpp_LIBRARIES}
    ${catkin_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OpenMP_LIBRARIES}
)

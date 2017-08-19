cmake_minimum_required(VERSION 2.8.3)

find_package(catkin REQUIRED roslib urdf srdfdom openrave_catkin)

find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(OpenRAVE REQUIRED)

catkin_package()
catkin_python_setup()

include_directories(
    include
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
    src/SurfaceContactPointGrid.cpp
    src/EscherMotionPlanning.cpp    
)

target_link_libraries("${PROJECT_NAME}_plugin"
    ${catkin_LIBRARIES}
    ${Boost_LIBRARIES}
)


#if(CATKIN_ENABLE_TESTING)
#    catkin_add_nosetests(tests)
#endif(CATKIN_ENABLE_TESTING)


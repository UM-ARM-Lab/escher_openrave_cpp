#ifndef UTILITIES_HPP
#define UTILITIES_HPP

// STD
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cassert>
#include <unistd.h>
#include <sstream>
#include <chrono>

// Eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

// Boost
#include <boost/bind.hpp>

// OpenRAVE
#include <rave/rave.h>
#include <openrave/planningutils.h>


typedef Eigen::Matrix4f TransformationMatrix;
typedef Eigen::Matrix3f RotationMatrix;
typedef Eigen::Vector2f Translation2D;
typedef Eigen::Vector3f Translation3D;

typedef std::array<int,2> GridIndices2D;
typedef std::array<float,2> GridPositions2D;
typedef std::array<int,3> GridIndices3D;
typedef std::array<float,3> GridPositions3D;

const float RAD2DEG = 180.0/M_PI;
const float DEG2RAD = M_PI/180.0;

const float FOOT_HEIGHT = 0.25;
const float FOOT_WIDTH = 0.135;
const float FOOT_RADIUS = hypot(FOOT_HEIGHT/2.0,FOOT_WIDTH/2.0);
const float HAND_HEIGHT = 0.20;
const float HAND_WIDTH = 0.14;
const float HAND_RADIUS = hypot(HAND_HEIGHT/2.0,HAND_WIDTH/2.0);

const float MIN_ARM_LENGTH = 0.3;
const float MAX_ARM_LENGTH = 0.7;

const Translation3D GLOBAL_NEGATIVE_Z = Translation3D(0,0,-1);

const float MU = 0.5;
const float MAX_ANGULAR_DEVIATION = atan(MU) * DEG2RAD;

const float SURFACE_CONTACT_POINT_RESOLUTION = 0.05; // meters

const int TORSO_GRID_MIN_THETA = -180;
const int TORSO_GRID_ANGULAR_RESOLUTION = 30;

const float SHOULDER_W = 0.6;
const float SHOULDER_Z = 1.45;

enum ContactManipulator
{
	L_LEG,
	R_LEG,
	L_ARM,
	R_ARM
};

enum ContactType
{
	FOOT,
	HAND
};

enum TrimeshType
{
	GROUND,
	OTHERS
};

const std::vector<ContactManipulator> ARM_MANIPULATORS = {ContactManipulator::L_ARM,ContactManipulator::R_ARM};
const std::vector<ContactManipulator> LEG_MANIPULATORS = {ContactManipulator::L_LEG,ContactManipulator::R_LEG};

struct RPY_tf
{
	float x; // meters
	float y; // meters
	float z; // meters
	float roll; // degrees
	float pitch; // degrees
	float yaw; // degrees
};

// euclidean distance btwn two points in a 2D coordinate system
float euclideanDistance2D(const Translation2D& q, const Translation2D& p);

// euclidean distance btwn two points in a 3D coordinate system
float euclideanDistance3D(const Translation3D& q, const Translation3D& p);

TransformationMatrix constructTransformationMatrix(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22, float m23);

RotationMatrix RPYToSO3(const RPY_tf& e); // TODO: need to change to the more robust version

TransformationMatrix XYZRPYToSE3(const RPY_tf& e);


Translation2D gridPositions2DToTranslation2D(GridPositions2D positions);

GridPositions2D translation2DToGridPositions2D(Translation2D positions);

bool isValidPosition(Translation3D p);

bool isValidPosition(Translation2D p);

#include "Structure.hpp"
#include "ContactPoint.hpp"
#include "PointGrid.hpp"
#include "SurfaceContactPointGrid.hpp"
#include "GroundContactPointGrid.hpp"
#include "TrimeshSurface.hpp"
#include "EscherMotionPlanning.hpp"


#endif
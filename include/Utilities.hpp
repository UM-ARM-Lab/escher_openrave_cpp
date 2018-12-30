#ifndef UTILITIES_HPP
#define UTILITIES_HPP

// STD
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <set>
#include <unistd.h>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// Boost
#include <boost/bind.hpp>

// OpenRAVE
#include <rave/rave.h>
#include <openrave/planningutils.h>

// NEWMAT
#include <newmat/newmatap.h>
#include <newmat/newmatio.h>

// frugally deep
#include <fdeep/fdeep.hpp>

// cddlib
#define GMPRATIONAL 1
#include <cdd/setoper.h>
#include <cdd/cdd.h>

typedef Eigen::Matrix4f          TransformationMatrix;
typedef Eigen::Matrix3f          RotationMatrix;
typedef Eigen::Vector2f          Translation2D;
typedef Eigen::Vector3f          Translation3D;
typedef Eigen::Vector2f          Vector2D;
typedef Eigen::Vector3f          Vector3D;
typedef Eigen::Quaternion<float> Quaternion;

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
const float MAX_ARM_LENGTH = 0.65; // to make it more conservative

const Translation3D GLOBAL_NEGATIVE_Z = Translation3D(0,0,-1);

const float MU = 0.5;
const float MAX_ANGULAR_DEVIATION = atan(MU) * RAD2DEG;
const float STEP_TRANSITION_TIME = 0.5;
const float SUPPORT_PHASE_TIME = 0.5;

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
    R_ARM,
    MANIP_NUM
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

enum PlanningHeuristicsType
{
    EUCLIDEAN,
    DIJKSTRA
};

enum ExploreState
{
    OPEN,
    EXPLORED,
    CLOSED,
    REOPEN
};

enum TerrainType
{
    SOLID,
    GAP,
    OBSTACLE
};

enum BranchingMethod
{
    CONTACT_PROJECTION,
    CONTACT_OPTIMIZATION
};

enum ContactTransitionCode
{
    FEET_ONLY_MOVE_FOOT,                // 0
    FEET_ONLY_ADD_HAND,                 // 1
    FEET_AND_ONE_HAND_MOVE_INNER_FOOT,  // 2
    FEET_AND_ONE_HAND_MOVE_OUTER_FOOT,  // 3
    FEET_AND_ONE_HAND_BREAK_HAND,       // 4
    FEET_AND_ONE_HAND_MOVE_HAND,        // 5
    FEET_AND_ONE_HAND_ADD_HAND,         // 6
    FEET_AND_TWO_HANDS_MOVE_FOOT,        // 7
    FEET_AND_TWO_HANDS_BREAK_HAND,       // 8
    FEET_AND_TWO_HANDS_MOVE_HAND         // 9
};

enum ZeroStepCaptureCode
{
    ONE_FOOT,                   // 0
    TWO_FEET,                   // 1
    ONE_FOOT_AND_INNER_HAND,    // 2
    ONE_FOOT_AND_OUTER_HAND,    // 3
    ONE_FOOT_AND_TWO_HANDS,     // 4
    FEET_AND_ONE_HAND           // 5
};

enum OneStepCaptureCode
{
    ONE_FOOT_ADD_FOOT,                  // 0
    ONE_FOOT_ADD_INNER_HAND,            // 1
    ONE_FOOT_ADD_OUTER_HAND,            // 2
    TWO_FEET_ADD_HAND,                  // 3
    ONE_FOOT_AND_INNER_HAND_ADD_FOOT,   // 4
    ONE_FOOT_AND_INNER_HAND_ADD_HAND,   // 5
    ONE_FOOT_AND_OUTER_HAND_ADD_FOOT,   // 6
    ONE_FOOT_AND_OUTER_HAND_ADD_HAND,   // 7
    ONE_FOOT_AND_TWO_HANDS_ADD_FOOT,    // 8
    TWO_FEET_AND_ONE_HAND_ADD_HAND      // 9
};

enum DynOptApplication
{
    CONTACT_TRANSITION_DYNOPT,
    ZERO_STEP_CAPTURABILITY_DYNOPT,
    ONE_STEP_CAPTURABILITY_DYNOPT
};

enum PlanningApplication
{
    COLLECT_DATA,
    PLAN_IN_ENV
};

enum BranchingManipMode
{
    FEET_CONTACTS,
    HAND_CONTACTS,
    BREAKING_HAND_CONTACTS,
    ALL
};

struct EnumClassHash
{
    template <typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};

struct EigenVectorHash
{
    template <typename T>
    std::size_t operator()(T t) const
    {
        std::size_t hash_number = 0;
        for(int i = 0; i < t.rows(); i++)
        {
            hash_number ^= hash<float>()(t[i]) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
        }

        return hash_number;
    }
};

struct pointer_less
{
    template <typename T>
    bool operator()(const T& lhs, const T& rhs) const
    {
        return *lhs < *rhs;
    }
};

const std::vector<ContactManipulator> ALL_MANIPULATORS = {ContactManipulator::L_LEG, ContactManipulator::R_LEG, ContactManipulator::L_ARM, ContactManipulator::R_ARM};
const std::vector<ContactManipulator> ARM_MANIPULATORS = {ContactManipulator::L_ARM, ContactManipulator::R_ARM};
const std::vector<ContactManipulator> LEG_MANIPULATORS = {ContactManipulator::L_LEG, ContactManipulator::R_LEG};

class RPYTF
{
    public:
        RPYTF(){};
        RPYTF(std::array<float,6> xyzrpy)
        {
            x_ = round(xyzrpy[0] * 1000.0) / 1000.0;
            y_ = round(xyzrpy[1] * 1000.0) / 1000.0;
            z_ = round(xyzrpy[2] * 1000.0) / 1000.0;
            roll_ = round(xyzrpy[3] * 10.0) / 10.0;
            pitch_ = round(xyzrpy[4] * 10.0) / 10.0;
            yaw_ = round(xyzrpy[5] * 10.0) / 10.0;
        }

        RPYTF(float _x, float _y, float _z, float _roll, float _pitch, float _yaw)
        {
            x_ = round(_x * 1000.0) / 1000.0;
            y_ = round(_y * 1000.0) / 1000.0;
            z_ = round(_z * 1000.0) / 1000.0;
            roll_ = round(_roll * 10.0) / 10.0;
            pitch_ = round(_pitch * 10.0) / 10.0;
            yaw_ = round(_yaw * 10.0) / 10.0;
        }

        inline bool operator==(const RPYTF& other) const{ return ((this->x_ == other.x_) && (this->y_ == other.y_) && (this->z_ == other.z_) &&
                                                            (this->roll_ == other.roll_) && (this->pitch_ == other.pitch_) && (this->yaw_ == other.yaw_));}
        inline bool operator!=(const RPYTF& other) const{ return ((this->x_ != other.x_) || (this->y_ != other.y_) || (this->z_ != other.z_) ||
                                                            (this->roll_ != other.roll_) || (this->pitch_ != other.pitch_) || (this->yaw_ != other.yaw_));}

        inline std::array<float,6> getXYZRPY() const {return std::array<float,6>({x_, y_, z_, roll_, pitch_, yaw_});}
        inline Translation3D getXYZ() const {return Translation3D(x_, y_, z_);}
        inline Vector3D getRPY() const {return Vector3D(roll_, pitch_, yaw_);}

        inline void printPose() const {std::cout << x_ << " " << y_ << " " << z_ << " " << roll_ << " " << pitch_ << " " << yaw_ << std::endl;}

        float x_; // meters
        float y_; // meters
        float z_; // meters
        float roll_; // degrees
        float pitch_; // degrees
        float yaw_; // degrees

        OpenRAVE::RaveTransformMatrix<OpenRAVE::dReal> GetRaveTransformMatrix() const
        {
            OpenRAVE::RaveVector<OpenRAVE::dReal> x_axis_angle(roll_ * DEG2RAD, 0, 0);
            OpenRAVE::RaveVector<OpenRAVE::dReal> y_axis_angle(0, pitch_ * DEG2RAD, 0);
            OpenRAVE::RaveVector<OpenRAVE::dReal> z_axis_angle(0, 0, yaw_ * DEG2RAD);

            OpenRAVE::RaveVector<OpenRAVE::dReal> translation(x_, y_, z_);

            OpenRAVE::RaveTransformMatrix<OpenRAVE::dReal> transform_matrix = OpenRAVE::geometry::matrixFromAxisAngle(x_axis_angle) *
                                                                              OpenRAVE::geometry::matrixFromAxisAngle(y_axis_angle) *
                                                                              OpenRAVE::geometry::matrixFromAxisAngle(z_axis_angle);
            transform_matrix.trans = translation;

            return transform_matrix;
        }

        OpenRAVE::Transform GetRaveTransform() const
        {
            OpenRAVE::RaveVector<OpenRAVE::dReal> x_axis_angle(roll_ * DEG2RAD, 0, 0);
            OpenRAVE::RaveVector<OpenRAVE::dReal> y_axis_angle(0, pitch_ * DEG2RAD, 0);
            OpenRAVE::RaveVector<OpenRAVE::dReal> z_axis_angle(0, 0, yaw_ * DEG2RAD);

            OpenRAVE::RaveVector<OpenRAVE::dReal> translation(x_, y_, z_);

            OpenRAVE::Transform transform = OpenRAVE::Transform(OpenRAVE::geometry::quatFromAxisAngle(x_axis_angle), OpenRAVE::RaveVector<OpenRAVE::dReal>(0,0,0)) *
                                            OpenRAVE::Transform(OpenRAVE::geometry::quatFromAxisAngle(y_axis_angle), OpenRAVE::RaveVector<OpenRAVE::dReal>(0,0,0)) *
                                            OpenRAVE::Transform(OpenRAVE::geometry::quatFromAxisAngle(z_axis_angle), OpenRAVE::RaveVector<OpenRAVE::dReal>(0,0,0));
            transform.trans = translation;

            return transform;
        }
};

namespace std
{
    template <>
    class hash<RPYTF>{
        public:
            size_t operator()(const RPYTF &tf) const
            {
                size_t hash_number = 0;

                hash_number ^= hash<float>()(tf.x_) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(tf.y_) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(tf.z_) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(tf.roll_) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(tf.pitch_) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(tf.yaw_) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);

                return hash_number;
            }
    };
}

// Distance
float euclideanDistance2D(const Translation2D& q, const Translation2D& p); // euclidean distance btwn two points in a 2D coordinate system
float euclideanDistance3D(const Translation3D& q, const Translation3D& p); // euclidean distance btwn two points in a 3D coordinate system

float orientationDistance(const Vector3D& q, const Vector3D& p);
float orientationDistance(const RPYTF& q, const RPYTF& p);
float orientationDistance(const Quaternion& q, const Quaternion& p);

// Transformation Matrix
TransformationMatrix constructTransformationMatrix(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22, float m23);
TransformationMatrix inverseTransformationMatrix(TransformationMatrix T);
RotationMatrix RPYToSO3(const RPYTF& e);
TransformationMatrix XYZRPYToSE3(const RPYTF& e);
RPYTF SE3ToXYZRPY(const TransformationMatrix& T);
RPYTF SO3ToRPY(const RotationMatrix& R);
Quaternion RPYToQuaternion(const RPYTF& e);
// Quaternion SO3ToQuaternion(const );
// RotationMatrix QauternionToSO3();

// Manipulate the angle differences
float getFirstTerminalAngle(float angle);
float getAngleDifference(float angle1, float angle2);
float getAngleMean(float angle1, float angle2);

// Data structure translation
Translation2D gridPositions2DToTranslation2D(GridPositions2D positions);
GridPositions2D translation2DToGridPositions2D(Translation2D positions);

// Validate data integrity
bool isValidPosition(Translation3D p);
bool isValidPosition(Translation2D p);

// Transform pose and position from OpenRAVE to SL
RPYTF transformPoseFromOpenraveToSL(RPYTF& e);
RPYTF transformPoseFromOpenraveToSL(RPYTF& e, TransformationMatrix& offset_transform);
Eigen::Vector3d rotateVectorFromOpenraveToSL(Vector3D& t);
Vector3D rotateVectorFromSLToOpenrave(Eigen::Vector3d& t);
Eigen::Vector3d transformPositionFromOpenraveToSL(Translation3D& t);
Translation3D transformPositionFromSLToOpenrave(Eigen::Vector3d& t);

// Color
std::array<float,4> HSVToRGB(std::array<float,4> hsv);

#include "GIWC.hpp"
#include "GeneralIKInterface.hpp"
#include "Drawing.hpp"
#include "RobotProperties.hpp"
#include "Structure.hpp"
#include "ContactPoint.hpp"
#include "ContactRegion.hpp"
#include "PointGrid.hpp"
#include "SurfaceContactPointGrid.hpp"
#include "GroundContactPointGrid.hpp"
#include "TrimeshSurface.hpp"
#include "MapGrid.hpp"
#include "TorsoPathPlanning.hpp"
#include "ContactState.hpp"
#include "OptimizationInterface.hpp"
#include "NeuralNetworkInterface.hpp"
#include "ContactSpacePlanning.hpp"
#include "EscherMotionPlanning.hpp"
#include "Boundary.hpp"

#endif
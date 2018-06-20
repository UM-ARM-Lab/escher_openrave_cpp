#include "Utilities.hpp"

RobotProperties::RobotProperties(OpenRAVE::RobotBasePtr _robot, std::vector<OpenRAVE::dReal> _IK_init_DOF_Values, std::vector<OpenRAVE::dReal> _default_DOF_Values, 
                                 float _foot_h, float _foot_w, float _hand_h, float _hand_w, float _robot_z, float _top_z, float _shoulder_z, float _shoulder_w):
    name_(_robot->GetName()),
    default_DOF_Values_(_default_DOF_Values),
    IK_init_DOF_Values_(_IK_init_DOF_Values),
    foot_h_(_foot_h),
    foot_w_(_foot_w),
    foot_radius_(std::sqrt(_foot_h*_foot_h + _foot_w*_foot_w)),
    hand_h_(_hand_h),
    hand_w_(_hand_w),
    hand_radius_(std::sqrt(_hand_h*_hand_h + _hand_w*_hand_w)),
    robot_z_(_robot_z),
    top_z_(_top_z),
    shoulder_z_(_shoulder_z),
    shoulder_w_(_shoulder_w)
{
    for(auto &joint : _robot->GetJoints())
    {
        DOFName_index_map_.insert(std::make_pair(joint->GetName(), joint->GetJointIndex()));
    }

    int active_dof_index = 0;
    for(auto &dof_index : _robot->GetActiveDOFIndices())
    {
        ActiveDOFName_index_map_.insert(std::make_pair(_robot->GetJointFromDOFIndex(dof_index)->GetName(), active_dof_index));
        active_dof_index++;
    }

    _robot->GetActiveDOFLimits(lower_joint_limits_, higher_joint_limits_);
}
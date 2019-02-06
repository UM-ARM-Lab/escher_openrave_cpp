#include "Utilities.hpp"

RobotProperties::RobotProperties(OpenRAVE::RobotBasePtr _robot, std::vector<OpenRAVE::dReal> _IK_init_DOF_Values, std::vector<OpenRAVE::dReal> _default_DOF_Values,
                                 float _foot_h, float _foot_w, float _hand_h, float _hand_w, float _robot_z, float _top_z,
                                 float _shoulder_z, float _shoulder_w, float _max_arm_length, float _min_arm_length, float _max_stride):
    name_(_robot->GetName()),
    IK_init_DOF_Values_(_IK_init_DOF_Values),
    default_DOF_Values_(_default_DOF_Values),
    foot_h_(_foot_h),
    foot_w_(_foot_w),
    foot_radius_(std::sqrt(_foot_h*_foot_h + _foot_w*_foot_w)/2.0),
    hand_h_(_hand_h),
    hand_w_(_hand_w),
    hand_radius_(std::sqrt(_hand_h*_hand_h + _hand_w*_hand_w)/2.0),
    robot_z_(_robot_z),
    top_z_(_top_z),
    shoulder_z_(_shoulder_z),
    shoulder_w_(_shoulder_w),
    max_arm_length_(_max_arm_length),
    min_arm_length_(_min_arm_length),
    max_stride_(_max_stride)
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

    manipulator_name_map_.insert(std::make_pair(ContactManipulator::L_LEG, "l_leg"));
    manipulator_name_map_.insert(std::make_pair(ContactManipulator::R_LEG, "r_leg"));
    manipulator_name_map_.insert(std::make_pair(ContactManipulator::L_ARM, "l_arm"));
    manipulator_name_map_.insert(std::make_pair(ContactManipulator::R_ARM, "r_arm"));

    // construct the DOF name - SL index map
    // SL follows the sequence of left leg, right leg, left arm, right arm , and others. (at least for Athena)
    // here we only do the mapping for the 4 manipulators
    const std::vector<ContactManipulator> SL_MANIPULATOR_SEQ = {ContactManipulator::L_ARM, ContactManipulator::R_ARM, ContactManipulator::L_LEG, ContactManipulator::R_LEG};
    int SL_index_counter = 0;
    for(auto & manip : SL_MANIPULATOR_SEQ)
    {
        auto manipulator = _robot->GetManipulator(manipulator_name_map_[manip]);

        for(auto & rave_index : manipulator->GetArmIndices())
        {
            DOFindex_SLindex_map_[rave_index] = SL_index_counter;
            SL_index_counter++;
        }
    }

}

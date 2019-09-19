#include "Utilities.hpp"

RobotProperties::RobotProperties(OpenRAVE::RobotBasePtr _robot, std::vector<OpenRAVE::dReal> _IK_init_DOF_Values, std::vector<OpenRAVE::dReal> _default_DOF_Values,
                                 float _foot_h, float _foot_w, float _hand_h, float _hand_w, float _robot_z, float _top_z,
                                 float _shoulder_z, float _shoulder_w, float _max_arm_length, float _min_arm_length, float _max_stride, float _mass):
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
    max_stride_(_max_stride),
    mass_(_mass)
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

    TransformationMatrix lf_offset_transform, rf_offset_transform, lh_offset_transform, rh_offset_transform;

    if(name_ == "athena")
    {
        // construct the DOF name - SL index map
        // SL follows the sequence of left arm, right arm, left leg, right leg, torso joints, and floating base joints. (at least for Athena)
        // here we only do the mapping for the 4 manipulators
        const std::vector<ContactManipulator> SL_MANIPULATOR_SEQ = {ContactManipulator::L_ARM, ContactManipulator::R_ARM, ContactManipulator::L_LEG, ContactManipulator::R_LEG};
        int SL_index_counter = 0;
        for(auto & manip : SL_MANIPULATOR_SEQ)
        {
            auto manipulator = _robot->GetManipulator(manipulator_name_map_[manip]);

            for(auto & rave_index : manipulator->GetArmIndices())
            {
                DOFindex_SLindex_map_[rave_index] = SL_index_counter++;
            }
        }

        // torso dofs
        DOFindex_SLindex_map_[_robot->GetJoint("B_TR")->GetJointIndex()] = SL_index_counter++;
        DOFindex_SLindex_map_[_robot->GetJoint("B_TAA")->GetJointIndex()] = SL_index_counter++;

        // // L_THR, L_THF, L_IF, L_MF, L_RF, R_THR, R_THF, R_IF, R_MF, R_RF joints (seems to be the finger joints?)
        // SL_index_counter += 10;

        // // floating base dofs
        // DOFindex_SLindex_map_[_robot->GetJoint("x_prismatic_joint")->GetJointIndex()] = SL_index_counter++;
        // DOFindex_SLindex_map_[_robot->GetJoint("y_prismatic_joint")->GetJointIndex()] = SL_index_counter++;
        // DOFindex_SLindex_map_[_robot->GetJoint("z_prismatic_joint")->GetJointIndex()] = SL_index_counter++;
        // DOFindex_SLindex_map_[_robot->GetJoint("roll_revolute_joint")->GetJointIndex()] = SL_index_counter++;
        // DOFindex_SLindex_map_[_robot->GetJoint("pitch_revolute_joint")->GetJointIndex()] = SL_index_counter++;
        // DOFindex_SLindex_map_[_robot->GetJoint("yaw_revolute_joint")->GetJointIndex()] = SL_index_counter++;

        lf_offset_transform << 1, 0, 0, 0,
                               0, 1, 0, 0,
                               0, 0, 1, 0,
                               0, 0, 0, 1;
        ee_offset_transform_to_dynopt_[ContactManipulator::L_LEG] = lf_offset_transform;

        rf_offset_transform << 1, 0, 0, 0,
                               0, 1, 0, 0,
                               0, 0, 1, 0,
                               0, 0, 0, 1;
        ee_offset_transform_to_dynopt_[ContactManipulator::R_LEG] = rf_offset_transform;

        lh_offset_transform << 0, 0, 1, 0,
                               1, 0, 0, 0,
                               0, 1, 0, 0,
                               0, 0, 0, 1;
        ee_offset_transform_to_dynopt_[ContactManipulator::L_ARM] = lh_offset_transform;

        rh_offset_transform <<  0,  0, 1, 0,
                               -1,  0, 0, 0,
                                0, -1, 0, 0,
                                0,  0, 0, 1;
        ee_offset_transform_to_dynopt_[ContactManipulator::R_ARM] = rh_offset_transform;
    }
    else if(name_ == "hermes_full")
    {
        // construct the DOF name - SL index map
        // SL follows the sequence of left arm, right arm, left leg, right leg, torso joints, and floating base joints. (at least for Athena)
        // here we only do the mapping for the 4 manipulators
        const std::vector<ContactManipulator> SL_MANIPULATOR_SEQ = {ContactManipulator::L_ARM, ContactManipulator::R_ARM, ContactManipulator::L_LEG, ContactManipulator::R_LEG};
        int SL_index_counter = 0;
        for(auto & manip : SL_MANIPULATOR_SEQ)
        {
            auto manipulator = _robot->GetManipulator(manipulator_name_map_[manip]);

            for(auto & rave_index : manipulator->GetArmIndices())
            {
                DOFindex_SLindex_map_[rave_index] = SL_index_counter++;
            }
        }

        // torso dofs
        DOFindex_SLindex_map_[_robot->GetJoint("B_TR")->GetJointIndex()] = SL_index_counter++;
        DOFindex_SLindex_map_[_robot->GetJoint("B_TAA")->GetJointIndex()] = SL_index_counter++;
        DOFindex_SLindex_map_[_robot->GetJoint("B_TFE")->GetJointIndex()] = SL_index_counter++;

        // SFE and SAA joint index are reversed
        std::swap(DOFindex_SLindex_map_[_robot->GetJoint("L_SFE")->GetJointIndex()], DOFindex_SLindex_map_[_robot->GetJoint("L_SAA")->GetJointIndex()]);
        std::swap(DOFindex_SLindex_map_[_robot->GetJoint("R_SFE")->GetJointIndex()], DOFindex_SLindex_map_[_robot->GetJoint("R_SAA")->GetJointIndex()]);

        lf_offset_transform << 1, 0, 0, 0,
                               0, 1, 0, 0,
                               0, 0, 1, 0,
                               0, 0, 0, 1;
        ee_offset_transform_to_dynopt_[ContactManipulator::L_LEG] = lf_offset_transform;

        rf_offset_transform << 1, 0, 0, 0,
                               0, 1, 0, 0,
                               0, 0, 1, 0,
                               0, 0, 0, 1;
        ee_offset_transform_to_dynopt_[ContactManipulator::R_LEG] = rf_offset_transform;

        lh_offset_transform << 0,  0, -1, 0,
                               1,  0,  0, 0,
                               0, -1,  0, 0,
                               0,  0,  0, 1;
        ee_offset_transform_to_dynopt_[ContactManipulator::L_ARM] = lh_offset_transform;

        rh_offset_transform <<  0,  0, -1, 0,
                               -1,  0,  0, 0,
                                0,  1,  0, 0,
                                0,  0,  0, 1;
        ee_offset_transform_to_dynopt_[ContactManipulator::R_ARM] = rh_offset_transform;
    }
    else
    {
        RAVELOG_WARN("%s robot's DOF index to SL index mapping is not defined.\n", name_);
        getchar();
    }


}

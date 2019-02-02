#ifndef ROBOT_PROPERTIES_H
#define ROBOT_PROPERTIES_H

// #include "Utilities.hpp"

class RobotProperties
{
    public:
        RobotProperties(OpenRAVE::RobotBasePtr _robot, std::vector<OpenRAVE::dReal> _IK_init_DOF_Values, std::vector<OpenRAVE::dReal> _default_DOF_Values,
                        float _foot_h, float _foot_w, float _hand_h, float _hand_w, float _robot_z, float _top_z,
                        float _shoulder_z, float _shoulder_w, float _max_arm_length, float _min_arm_length, float _max_stride);

        const std::string name_;

        std::map<std::string, int> DOFName_index_map_;
        std::map<std::string, int> ActiveDOFName_index_map_;

        std::vector<OpenRAVE::dReal> lower_joint_limits_;
        std::vector<OpenRAVE::dReal> higher_joint_limits_;

        std::map<ContactManipulator, std::string> manipulator_name_map_;

        std::map<std::string, int> DOFName_SLindex_map_;

        const std::vector<OpenRAVE::dReal> IK_init_DOF_Values_; // the OriginalDOFValues
        const std::vector<OpenRAVE::dReal> default_DOF_Values_; // the GazeboOriginalDOFValues

        const float foot_h_;
        const float foot_w_;
        const float hand_h_;
        const float hand_w_;
        const float foot_radius_;
        const float hand_radius_;

        const float robot_z_;
        const float top_z_;
        const float shoulder_z_;
        const float shoulder_w_;

        const float max_arm_length_;
        const float min_arm_length_;

        const float max_stride_;

    private:

};

#endif
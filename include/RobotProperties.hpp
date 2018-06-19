#ifndef ROBOT_PROPERTIES_H
#define ROBOT_PROPERTIES_H

// #include "Utilities.hpp"

class RobotProperties
{
    public:
        RobotProperties::RobotProperties(OpenRAVE::RobotBasePtr _robot);

        const string name_;
        
        std::map<string,int> DOFName_index_map_;
        std::map<string,int> ActiveDOFName_index_map_;

        std::map<string,int> lower_joint_limits_;
        std::map<string,int> higher_joint_limits_;

        const std::vector<dReal> IK_init_DOF_Values_; // the OriginalDOFValues
        const std::vector<dReal> default_DOF_Values_; // the GazeboOriginalDOFValues
        
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

    private:

};

#endif
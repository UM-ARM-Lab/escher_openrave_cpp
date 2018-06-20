#include "Utilities.hpp"

ContactSpacePlanning::ContactSpacePlanning(bool _goal_as_exact_poses, 
                                           RobotProperties _robot_properties,
                                           std::vector< std::array<float,3> > _foot_transition_model, 
                                           std::vector< std::array<float,2> > _hand_transition_model, 
                                           PlanningHeuristicsType _heuristics_type,
                                           std::vector< std::shared_ptr<TrimeshSurface> > _structures,
                                           std::map<int, std::shared_ptr<TrimeshSurface> > _structures_dict):
goal_as_exact_poses_(_goal_as_exact_poses),
robot_properties_(_robot_properties),
foot_transition_model_(_foot_transition_model),
hand_transition_model_(_hand_transition_model),
heuristics_type_(_heuristics_type),
structures_(_structures),
structures_dict_(_structures_dict)
{
    for(auto & structure : structures_)
    {
        if(structure->getType() == TrimeshType::GROUND)
        {
            foot_structures_.push_back(structure);
        }
        else if(structure->getType() == TrimeshType::OTHERS)
        {
            hand_structures_.push_back(structure);
        }
    }
}

std::vector< std::shared_ptr<ContactState> > ContactSpacePlanning::ANAStarPlanning(std::shared_ptr<ContactState> initial_state, std::array<float,3> goal, float goal_radius, float time_limit)
{
    // initialize parameters
    G_ = 9999.0;
    E_ = G_;
    goal_ = goal;
    goal_radius_ = goal_radius;
    time_limit_ = time_limit;

    // clear the heap and state list, and add initial state in the list
    while(!open_heap_.empty())
    {
        open_heap_.pop();
    }
    contact_states_vector_.clear();

    open_heap_.push(initial_state);
    contact_states_vector_.push_back(initial_state);

    auto time_before_ANA_start_planning = std::chrono::high_resolution_clock::now();
    
    // main exploration loop
    bool goal_reached = false;
    bool over_time_limit = false;
    while(!open_heap_.empty())
    {
        while(!open_heap_.empty())
        {
            // get the state in the top of the heap
            std::shared_ptr<ContactState> current_state = open_heap_.top();
            open_heap_.pop();

            if(current_state->explore_state_ == ExploreState::OPEN || current_state->explore_state_ == ExploreState::REOPEN)
            {
                // Collision Checking if needed

                // Kinematics feasibility check (distance filtering and IK solver)
                if(!kinematicFeasibilityCheck(current_state))
                {
                    current_state->explore_state_ = ExploreState::CLOSED;
                    continue;                    
                }

                // Dynamics feasbility check (dynamics should be already checked when creating the state)


                // branch
                branchingSearchTree(current_state);


                current_state->explore_state_ = ExploreState::EXPLORED;
            }

        }

        if(goal_reached || over_time_limit)
        {
            break;
        }

        // modify the contact_states_vector_ after one solution has been found
    }









    // retrace the paths from the final states
    std::vector< std::shared_ptr<ContactState> > contact_state_path;

    if(goal_reached)
    {

    }

    return contact_state_path;
}

bool ContactSpacePlanning::kinematicFeasibilityCheck(std::shared_ptr<ContactState> current_state)
{
    // both feet should be in contact
    bool left_foot_in_contact = current_state->stances_array_[0]->ee_contact_status_[ContactManipulator::L_LEG];
    bool right_foot_in_contact = current_state->stances_array_[1]->ee_contact_status_[ContactManipulator::R_LEG];
    bool left_hand_in_contact = current_state->stances_array_[0]->ee_contact_status_[ContactManipulator::L_ARM];
    bool right_hand_in_contact = current_state->stances_array_[1]->ee_contact_status_[ContactManipulator::R_ARM];
    
    if(left_foot_in_contact && right_foot_in_contact)
    {
        return false;
    }

    // distance check for the current state
    TransformationMatrix feet_mean_transform = current_state->getFeetMeanTransform();

    
    Translation3D right_shoulder_position(0, -robot_properties_.shoulder_w_ / 2.0, robot_properties_.shoulder_z_);

    if(left_hand_in_contact)
    {
        Translation3D left_hand_position = current_state->stances_array_[0]->left_foot_pose_.getXYZ();
        Translation3D left_shoulder_position(0, robot_properties_.shoulder_w_ / 2.0, robot_properties_.shoulder_z_);
        left_shoulder_position = (feet_mean_transform * left_shoulder_position.homogeneous()).block(0,0,3,1);
        float left_hand_to_shoulder_dist = (left_hand_position - left_shoulder_position).norm();

        if(left_hand_to_shoulder_dist > robot_properties_.max_arm_length_ || left_hand_to_shoulder_dist < robot_properties_.min_arm_length_)
        {
            return false;
        }
    }

    if(right_hand_in_contact)
    {
        Translation3D right_hand_position = current_state->stances_array_[0]->right_foot_pose_.getXYZ();
        Translation3D right_shoulder_position(0, robot_properties_.shoulder_w_ / 2.0, robot_properties_.shoulder_z_);
        right_shoulder_position = (feet_mean_transform * right_shoulder_position.homogeneous()).block(0,0,3,1);
        float right_hand_to_shoulder_dist = (right_hand_position - right_shoulder_position).norm();

        if(right_hand_to_shoulder_dist > robot_properties_.max_arm_length_ || right_hand_to_shoulder_dist < robot_properties_.min_arm_length_)
        {
            return false;
        }
    }

    // IK solver check
    // call generalIK
    if(!current_state->is_root_)
    {

    }

    return true;
}

float ContactSpacePlanning::getDynamicScore()
{
    return 0.0;
}

void ContactSpacePlanning::branchingSearchTree(std::shared_ptr<ContactState> current_state)
{
    // branching foot contacts

    // branching hand contacts

}

void ContactSpacePlanning::branchingFootContacts(std::shared_ptr<ContactState> current_state, std::vector<ContactManipulator> branching_manips)
{
    std::array<float,6> l_foot_xyzrpy, r_foot_xyzrpy;
    float l_foot_x, l_foot_y, l_foot_z, l_foot_roll, l_foot_pitch, l_foot_yaw;
    float r_foot_x, r_foot_y, r_foot_z, r_foot_roll, r_foot_pitch, r_foot_yaw;

    float l_leg_horizontal_yaw = current_state->getLeftHorizontalYaw();
    float r_leg_horizontal_yaw = current_state->getRightHorizontalYaw();

    std::vector< std::tuple<RPYTF, RPYTF, ContactManipulator> > branching_feet_combination;

    std::shared_ptr<Stance> current_stance = current_state->stances_array_[0];

    // get all the possible branches of feet combinations
    for(auto & manip : branching_manips)
    {
        if(manip == ContactManipulator::L_LEG || manip == ContactManipulator::R_LEG)
        {            
            for(auto & step : foot_transition_model_)
            {
                l_foot_xyzrpy = current_stance->left_foot_pose_.getXYZRPY();
                r_foot_xyzrpy = current_stance->right_foot_pose_.getXYZRPY();
                
                if(manip == ContactManipulator::L_LEG)
                {
                    l_foot_xyzrpy[0] = r_foot_xyzrpy[0] + std::cos(r_leg_horizontal_yaw*DEG2RAD) * step[0] - std::sin(r_leg_horizontal_yaw*DEG2RAD) * step[1]; // x
                    l_foot_xyzrpy[1] = r_foot_xyzrpy[1] + std::sin(r_leg_horizontal_yaw*DEG2RAD) * step[0] + std::cos(r_leg_horizontal_yaw*DEG2RAD) * step[1]; // y
                    l_foot_xyzrpy[2] = 99.0; // z
                    l_foot_xyzrpy[3] = 0; // roll
                    l_foot_xyzrpy[4] = 0; // pitch
                    l_foot_xyzrpy[5] = r_leg_horizontal_yaw + step[2]; // yaw
                }
                else if(manip == ContactManipulator::R_LEG)
                {
                    r_foot_xyzrpy[0] = l_foot_xyzrpy[0] + std::cos(l_leg_horizontal_yaw*DEG2RAD) * step[0] - std::sin(l_leg_horizontal_yaw*DEG2RAD) * (-step[1]); // x
                    r_foot_xyzrpy[1] = l_foot_xyzrpy[1] + std::sin(l_leg_horizontal_yaw*DEG2RAD) * step[0] + std::cos(l_leg_horizontal_yaw*DEG2RAD) * (-step[1]); // y
                    r_foot_xyzrpy[2] = 99.0; // z
                    r_foot_xyzrpy[3] = 0; // roll
                    r_foot_xyzrpy[4] = 0; // pitch
                    r_foot_xyzrpy[5] = l_leg_horizontal_yaw - step[2]; // yaw
                }

                branching_feet_combination.push_back(std::make_tuple(RPYTF(l_foot_xyzrpy), RPYTF(r_foot_xyzrpy), manip));
            }
        }
    }

    // given all the branching feet combinations create states
    // we can add OpenMP here
    RPYTF new_left_foot_pose, new_right_foot_pose, new_left_hand_pose, new_right_hand_pose;
    ContactManipulator move_manip;
    for(auto & step_combination : branching_feet_combination)
    {
        new_left_foot_pose = std::get<0>(step_combination);
        new_right_foot_pose = std::get<1>(step_combination);
        new_left_hand_pose = current_stance->left_hand_pose_;
        new_right_hand_pose = current_stance->right_hand_pose_;

        move_manip = std::get<2>(step_combination);
        
        // do projection to find the projected feet poses

        // construct the new state
        std::shared_ptr<Stance> new_stance(new Stance(new_left_foot_pose, new_right_foot_pose, new_left_hand_pose, new_right_hand_pose, current_stance->ee_contact_status_));
        
        ContactState new_contact_state(new_stance, current_state, move_manip, false);


        //// the following can be one function

        // filter out states whose contact is outside map grid

        // verify the state kinematic feasibility

        // verify the state dynamic feasbility and the dynamic edge cost

        // calculate the edge cost

        // calculate the heuristics

        // update the state cost and CoM

        // add the state to the state vector and/or the open heap

    }


}

void ContactSpacePlanning::branchingHandContacts()
{

}

bool ContactSpacePlanning::footProjection()
{

}

bool ContactSpacePlanning::handProjection()
{
    
}

float ContactSpacePlanning::getHeuristics(std::shared_ptr<ContactState> current_state)
{
    return 0.0;
}

float ContactSpacePlanning::getEdgeCost(std::shared_ptr<ContactState> prev_state, std::shared_ptr<ContactState> current_state)
{
    return 0.0;
}

void updateExploreStates()
{

}
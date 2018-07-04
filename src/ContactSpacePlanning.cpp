#include "Utilities.hpp"
#include <omp.h>

ContactSpacePlanning::ContactSpacePlanning(std::shared_ptr<RobotProperties> _robot_properties,
                                           std::vector< std::array<float,3> > _foot_transition_model,
                                           std::vector< std::array<float,2> > _hand_transition_model,
                                           std::vector< std::shared_ptr<TrimeshSurface> > _structures,
                                           std::map<int, std::shared_ptr<TrimeshSurface> > _structures_dict,
                                           int _num_stance_in_state,
                                           std::shared_ptr< DrawingHandler > _drawing_handler):
robot_properties_(_robot_properties),
foot_transition_model_(_foot_transition_model),
hand_transition_model_(_hand_transition_model),
structures_(_structures),
structures_dict_(_structures_dict),
num_stance_in_state_(_num_stance_in_state),
drawing_handler_(_drawing_handler)
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

    dynamics_optimizer_interface_vector_.resize(OPENMP_THREAD_NUM);

    for(int i = 0; i < OPENMP_THREAD_NUM; i++)
    {
        dynamics_optimizer_interface_vector_[i].reset(new DynOptInterface(2.0, "SL_optim_config_template/cfg_kdopt_demo.yaml"));
    }

    dynamics_optimizer_interface_.reset(new DynOptInterface(2.0, "SL_optim_config_template/cfg_kdopt_demo.yaml"));
}

std::vector< std::shared_ptr<ContactState> > ContactSpacePlanning::ANAStarPlanning(std::shared_ptr<ContactState> initial_state, std::array<float,3> goal,
                                                                                   float goal_radius, PlanningHeuristicsType heuristics_type,
                                                                                   float time_limit, bool output_first_solution, bool goal_as_exact_poses)
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
    contact_states_map_.clear();

    open_heap_.push(initial_state);
    contact_states_map_.insert(std::make_pair(std::hash<ContactState>()(*initial_state), *initial_state));

    auto time_before_ANA_start_planning = std::chrono::high_resolution_clock::now();

    // main exploration loop
    bool over_time_limit = false;
    bool goal_reached = false;
    std::shared_ptr<ContactState> goal_state;

    RAVELOG_INFO("Enter the exploration loop.\n");

    while(!open_heap_.empty())
    {
        while(!open_heap_.empty())
        {
            auto current_time = std::chrono::high_resolution_clock::now();
            if(std::chrono::duration_cast<std::chrono::milliseconds>(current_time - time_before_ANA_start_planning).count() /1000.0 > time_limit)
            {
                RAVELOG_INFO("Over time limit.\n");
                over_time_limit = true;
                break;
            }
            // get the state in the top of the heap
            std::shared_ptr<ContactState> current_state = open_heap_.top();
            open_heap_.pop();

            if(current_state->explore_state_ == ExploreState::OPEN || current_state->explore_state_ == ExploreState::REOPEN)
            {
                // Collision Checking if needed

                // Kinematic and dynamic feasibility check
                // if(!stateFeasibilityCheck(current_state))
                // {
                //     current_state->explore_state_ = ExploreState::CLOSED;
                //     continue;
                // }

                current_state->explore_state_ = ExploreState::EXPLORED;

                // update E_
                if(current_state->h_ != 0 && (G_-current_state->g_)/current_state->h_ < E_)
                {
                    E_ = (G_-current_state->g_)/current_state->h_;
                }

                // plotting the contact sequence so far
                // RAVELOG_INFO("Plot the contact sequence.\n");
                drawing_handler_->ClearHandler();

                drawing_handler_->DrawContactPath(current_state);

                // check if it reaches the goal
                // RAVELOG_INFO("Check if it reaches the goal.\n");
                if(isReachedGoal(current_state))
                {
                    G_ = current_state->g_;
                    goal_state = current_state;
                    goal_reached = true;

                    current_time = std::chrono::high_resolution_clock::now();

                    RAVELOG_INFO("Solution Found: T = %5.3f, G = %5.3f, E = %5.3f. \n", std::chrono::duration_cast<std::chrono::milliseconds>(current_time - time_before_ANA_start_planning).count() /1000.0, G_, E_);

                    // getchar();

                    if(!output_first_solution)
                    {
                        updateExploreStatesAndOpenHeap();
                    }

                    break;
                }

                // branch
                // RAVELOG_INFO("Branch the search tree.\n");
                branchingSearchTree(current_state);
                // RAVELOG_INFO("Finish one iteration.\n");
            }
        }

        if(output_first_solution || over_time_limit)
        {
            break;
        }

    }


    // retrace the paths from the final states
    std::vector< std::shared_ptr<ContactState> > contact_state_path;
    if(goal_reached)
    {
        std::shared_ptr<ContactState> final_path_state = goal_state;
        drawing_handler_->ClearHandler();
        drawing_handler_->DrawContactPath(final_path_state);
        while(true)
        {
            contact_state_path.push_back(final_path_state);

            if(final_path_state->is_root_)
            {
                break;
            }

            final_path_state = final_path_state->parent_;
        }
    }
    std::reverse(contact_state_path.begin(), contact_state_path.end());

    RAVELOG_INFO("The time limit (%5.2f seconds) has been reached. Output the current best solution. Press ENTER to proceed.\n",time_limit);


    return contact_state_path;
}

bool ContactSpacePlanning::kinematicFeasibilityCheck(std::shared_ptr<ContactState> current_state)
{
    // both feet should be in contact
    bool left_foot_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::L_LEG];
    bool right_foot_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::R_LEG];
    bool left_hand_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::L_ARM];
    bool right_hand_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::R_ARM];

    if(!left_foot_in_contact || !right_foot_in_contact)
    {
        return false;
    }

    // distance check for the current state
    TransformationMatrix feet_mean_transform = current_state->getFeetMeanTransform();

    Translation3D right_shoulder_position(0, -robot_properties_->shoulder_w_ / 2.0, robot_properties_->shoulder_z_);

    if(left_hand_in_contact)
    {
        Translation3D left_hand_position = current_state->stances_vector_[0]->left_foot_pose_.getXYZ();
        Translation3D left_shoulder_position(0, robot_properties_->shoulder_w_ / 2.0, robot_properties_->shoulder_z_);
        left_shoulder_position = (feet_mean_transform * left_shoulder_position.homogeneous()).block(0,0,3,1);
        float left_hand_to_shoulder_dist = (left_hand_position - left_shoulder_position).norm();

        if(left_hand_to_shoulder_dist > robot_properties_->max_arm_length_ || left_hand_to_shoulder_dist < robot_properties_->min_arm_length_)
        {
            return false;
        }
    }

    if(right_hand_in_contact)
    {
        Translation3D right_hand_position = current_state->stances_vector_[0]->right_foot_pose_.getXYZ();
        Translation3D right_shoulder_position(0, robot_properties_->shoulder_w_ / 2.0, robot_properties_->shoulder_z_);
        right_shoulder_position = (feet_mean_transform * right_shoulder_position.homogeneous()).block(0,0,3,1);
        float right_hand_to_shoulder_dist = (right_hand_position - right_shoulder_position).norm();

        if(right_hand_to_shoulder_dist > robot_properties_->max_arm_length_ || right_hand_to_shoulder_dist < robot_properties_->min_arm_length_)
        {
            return false;
        }
    }

    // IK solver check
    // call generalIK
    if(!current_state->is_root_)
    {
        std::shared_ptr<ContactState> prev_state = current_state->parent_;
    }

    return true;
}

bool ContactSpacePlanning::dynamicFeasibilityCheck(std::shared_ptr<ContactState> current_state, float& dynamics_cost, int index)
{
    current_state->com_(0) = (current_state->stances_vector_[0]->left_foot_pose_.x_ + current_state->stances_vector_[0]->right_foot_pose_.x_) / 2.0;
    current_state->com_(1) = (current_state->stances_vector_[0]->left_foot_pose_.y_ + current_state->stances_vector_[0]->right_foot_pose_.y_) / 2.0;
    current_state->com_(2) = (current_state->stances_vector_[0]->left_foot_pose_.z_ + current_state->stances_vector_[0]->right_foot_pose_.z_) / 2.0 + robot_properties_->robot_z_;

    // return true;

    if(!current_state->is_root_)
    {
        dynamics_cost = 0.0;
        // update the state cost and CoM
        std::vector< std::shared_ptr<ContactState> > contact_state_sequence = {current_state->parent_, current_state};

        dynamics_optimizer_interface_vector_[index]->updateContactSequence(contact_state_sequence);
        // dynamics_optimizer_interface_->updateContactSequence(contact_state_sequence);

        bool simplified_dynamically_feasible = dynamics_optimizer_interface_vector_[index]->simplifiedDynamicsOptimization(dynamics_cost);

        bool dynamically_feasible = dynamics_optimizer_interface_vector_[index]->dynamicsOptimization(dynamics_cost);
        // bool dynamically_feasible = dynamics_optimizer_interface_->dynamicsOptimization(dynamics_cost);

        if(!simplified_dynamically_feasible && dynamically_feasible)
        {
            std::cout << "weird thing happens." << std::endl;
            getchar();
        }

        std::cout << "===============================================================" << std::endl;

        if(dynamically_feasible)
        {
            // update com, com_dot of the current_state
            dynamics_optimizer_interface_vector_[index]->updateStateCoM(current_state);
            // dynamics_optimizer_interface_->updateStateCoM(current_state);
        }

        // std::cout << "Dynamically feasible: " << dynamically_feasible << std::endl;

        return dynamically_feasible;
    }
    else
    {
        return true;
    }
}

bool ContactSpacePlanning::stateFeasibilityCheck(std::shared_ptr<ContactState> current_state, float& dynamics_cost, int index)
{
    // verify the state kinematic and dynamic feasibility
    return (kinematicFeasibilityCheck(current_state) && dynamicFeasibilityCheck(current_state, dynamics_cost, index));
}

void ContactSpacePlanning::branchingSearchTree(std::shared_ptr<ContactState> current_state)
{
    std::vector<ContactManipulator> branching_manips{ContactManipulator::L_LEG, ContactManipulator::R_LEG};

    // branching foot contacts
    branchingFootContacts(current_state, branching_manips);

    // branching hand contacts
    branchingHandContacts(current_state, branching_manips);

}

void ContactSpacePlanning::branchingFootContacts(std::shared_ptr<ContactState> current_state, std::vector<ContactManipulator> branching_manips)
{
    // RAVELOG_INFO("Start branching foot contact.\n");

    std::array<float,6> l_foot_xyzrpy, r_foot_xyzrpy;
    float l_foot_x, l_foot_y, l_foot_z, l_foot_roll, l_foot_pitch, l_foot_yaw;
    float r_foot_x, r_foot_y, r_foot_z, r_foot_roll, r_foot_pitch, r_foot_yaw;

    float l_leg_horizontal_yaw = current_state->getLeftHorizontalYaw();
    float r_leg_horizontal_yaw = current_state->getRightHorizontalYaw();

    std::vector< std::tuple<RPYTF, RPYTF, ContactManipulator> > branching_feet_combination;

    std::shared_ptr<Stance> current_stance = current_state->stances_vector_[0];

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

    std::vector< std::tuple<bool, std::shared_ptr<ContactState>, float> > state_feasibility_check_result(branching_feet_combination.size());
    RPYTF new_left_hand_pose = current_stance->left_hand_pose_;
    RPYTF new_right_hand_pose = current_stance->right_hand_pose_;
    const std::array<bool,ContactManipulator::MANIP_NUM> new_ee_contact_status = current_stance->ee_contact_status_;

    // RAVELOG_INFO("Total thread number: %d.\n",OPENMP_THREAD_NUM);

    #pragma omp parallel num_threads(OPENMP_THREAD_NUM) shared (branching_feet_combination, state_feasibility_check_result)
    {
        #pragma omp for schedule(static)
        for(int i = 0; i < branching_feet_combination.size(); i++)
        {
            RPYTF new_left_foot_pose, new_right_foot_pose, new_left_hand_pose, new_right_hand_pose;
            ContactManipulator move_manip;

            auto step_combination = branching_feet_combination[i];

            new_left_foot_pose = std::get<0>(step_combination);
            new_right_foot_pose = std::get<1>(step_combination);

            move_manip = std::get<2>(step_combination);

            bool projection_is_successful;

            // RAVELOG_INFO("foot projection.\n");

            // do projection to find the projected feet poses
            if(move_manip == ContactManipulator::L_LEG)
            {
                projection_is_successful = footProjection(new_left_foot_pose);
            }
            else if(move_manip == ContactManipulator::R_LEG)
            {
                projection_is_successful = footProjection(new_right_foot_pose);
            }

            if(!projection_is_successful)
            {
                continue;
            }

            // RAVELOG_INFO("construct state.\n");

            // construct the new state
            std::shared_ptr<Stance> new_stance(new Stance(new_left_foot_pose, new_right_foot_pose, new_left_hand_pose, new_right_hand_pose, new_ee_contact_status));

            std::shared_ptr<ContactState> new_contact_state(new ContactState(new_stance, current_state, move_manip, 1));

            // RAVELOG_INFO("state feasibility check.\n");

            float dynamics_cost = 0.0;
            if(stateFeasibilityCheck(new_contact_state, dynamics_cost, i%OPENMP_THREAD_NUM))
            {
                state_feasibility_check_result[i] = std::make_tuple(true, new_contact_state, dynamics_cost);
            }
            else
            {
                state_feasibility_check_result[i] = std::make_tuple(false, new_contact_state, dynamics_cost);
            }
        }
    }

    // RAVELOG_INFO("Num branching passing checks: %d.\n",state_feasibility_check_result.size());

    for(auto & check_result: state_feasibility_check_result)
    {
        bool pass_state_feasibility_check = std::get<0>(check_result);
        std::shared_ptr<ContactState> new_contact_state = std::get<1>(check_result);
        float dynamics_cost = std::get<2>(check_result);

        if(pass_state_feasibility_check)
        {
            insertState(new_contact_state, dynamics_cost);
        }

    }

}

void ContactSpacePlanning::branchingHandContacts(std::shared_ptr<ContactState> current_state, std::vector<ContactManipulator> branching_manips)
{

}

bool ContactSpacePlanning::footProjection(RPYTF& projection_pose)
{
    // can use map grid to predict if it is safelt inside a surface, and what are the surfaces to check, skip that for now

    // filter out states whose contact is outside map grid, skip it for now

    const Translation3D projection_origin = projection_pose.getXYZ();
    const Translation3D projection_ray(0, 0, -1);
    float contact_projection_yaw = projection_pose.yaw_;

    TransformationMatrix highest_projection_transformation_matrix;

    bool found_projection_surface = false;

    std::shared_ptr<TrimeshSurface> projected_surface;

    for(auto & structure : foot_structures_)
    {
        TransformationMatrix tmp_projection_transformation_matrix;
        bool valid_contact = false;
        tmp_projection_transformation_matrix = structure->projection(projection_origin, projection_ray, contact_projection_yaw, ContactType::FOOT, robot_properties_, valid_contact);

        if(valid_contact)
        {
            if(!found_projection_surface || tmp_projection_transformation_matrix(2,3) > highest_projection_transformation_matrix(2,3))
            {
                highest_projection_transformation_matrix = tmp_projection_transformation_matrix;
                projected_surface = structure;
            }

            found_projection_surface = true;
        }
    }

    if(found_projection_surface)
    {
        projection_pose = SE3ToXYZRPY(highest_projection_transformation_matrix);
    }

    return found_projection_surface;
}

bool ContactSpacePlanning::handProjection()
{

}

void ContactSpacePlanning::insertState(std::shared_ptr<ContactState> current_state, float dynamics_cost)
{
    std::shared_ptr<ContactState> prev_state = current_state->parent_;
    // calculate the edge cost and the cost to come
    current_state->g_ += getEdgeCost(prev_state, current_state, dynamics_cost);

    // calculate the heuristics (cost to go)
    current_state->h_ = getHeuristics(current_state);

    // find if there already exists this state
    std::size_t current_state_hash = std::hash<ContactState>()(*current_state);
    std::unordered_map<std::size_t, ContactState>::iterator contact_state_iterator = contact_states_map_.find(current_state_hash);

    // add the state to the state vector and/or the open heap
    if (contact_state_iterator == contact_states_map_.end()) // the state is not in the set
    {
        // RAVELOG_INFO("New state.\n");
        if(current_state->getF() < G_)
        {
            if(current_state->h_ != 0)
            {
                current_state->priority_value_ = (G_ - current_state->g_) / current_state->h_;
            }
            else
            {
                current_state->priority_value_ = (G_ - current_state->g_) / 0.00001;
            }

            auto current_state_iterator_insert_sucess_pair = contact_states_map_.insert(std::make_pair(current_state_hash, *current_state));
            auto current_state_ptr_to_the_map = std::make_shared<ContactState>(current_state_iterator_insert_sucess_pair.first->second);
            open_heap_.push(current_state_ptr_to_the_map);
        }

    }
    else
    {
        ContactState existing_state = contact_state_iterator->second;

        if(existing_state.explore_state_ != ExploreState::CLOSED && current_state->getF() < existing_state.getF())
        {
            if(existing_state.explore_state_ == ExploreState::EXPLORED)
            {
                existing_state.explore_state_ = ExploreState::REOPEN;
            }

            existing_state.g_ = current_state->g_;
            existing_state.parent_ = current_state->parent_;
            existing_state.prev_move_manip_ = current_state->prev_move_manip_;

            if(existing_state.getF() < G_)
            {
                if(existing_state.h_ != 0)
                {
                    existing_state.priority_value_ = (G_ - existing_state.g_) / existing_state.h_;
                }
                else
                {
                    existing_state.priority_value_ = (G_ - existing_state.g_) / 0.00001;
                }
                auto existing_state_ptr_to_the_set = std::make_shared<ContactState>(existing_state);
                open_heap_.push(existing_state_ptr_to_the_set);
            }
        }
    }

}

float ContactSpacePlanning::getHeuristics(std::shared_ptr<ContactState> current_state)
{
    float euclidean_distance_to_goal = std::sqrt(std::pow(current_state->com_(0) - goal_[0],2) + std::pow(current_state->com_(1) - goal_[1],2));
    float step_cost_to_goal = euclidean_distance_to_goal / robot_properties_->max_stride_;

    return (euclidean_distance_to_goal + step_cost_to_goal);
}

float ContactSpacePlanning::getEdgeCost(std::shared_ptr<ContactState> prev_state, std::shared_ptr<ContactState> current_state, float dynamics_cost)
{
    float traveling_distance_cost = std::sqrt(std::pow(current_state->com_(0) - prev_state->com_(0), 2) + std::pow(current_state->com_(1) - prev_state->com_(1), 2));
    float orientation_cost = 0.1 * fabs(current_state->getFeetMeanHorizontalYaw() - prev_state->getFeetMeanHorizontalYaw());
    float step_cost = step_cost_weight_;
    dynamics_cost = dynamics_cost_weight_ * dynamics_cost;

    return (traveling_distance_cost + orientation_cost + step_cost + dynamics_cost);
}

void ContactSpacePlanning::updateExploreStatesAndOpenHeap()
{
    while(!open_heap_.empty())
    {
        open_heap_.pop();
    }

    for(auto & contact_state_hash_pair : contact_states_map_)
    {
        std::shared_ptr<ContactState> contact_state = std::make_shared<ContactState>(contact_state_hash_pair.second);

        if(contact_state->explore_state_ == ExploreState::OPEN || contact_state->explore_state_ == ExploreState::REOPEN || contact_state->explore_state_ == ExploreState::EXPLORED)
        {
            if(contact_state->getF() >= G_)
            {
                contact_state->explore_state_ = ExploreState::CLOSED;
            }
            else
            {
                if(contact_state->h_ != 0)
                {
                    contact_state->priority_value_ = (G_ - contact_state->g_) / contact_state->h_;
                }
                else
                {
                    contact_state->priority_value_ = (G_ - contact_state->g_) / 0.00001;
                }

                if(contact_state->explore_state_ == ExploreState::OPEN || contact_state->explore_state_ == ExploreState::REOPEN)
                {
                    open_heap_.push(contact_state);
                }
            }
        }
    }
}

bool ContactSpacePlanning::isReachedGoal(std::shared_ptr<ContactState> current_state)
{
    return std::sqrt(std::pow(goal_[0]-current_state->com_(0), 2) + std::pow(goal_[1]-current_state->com_(1), 2)) <= goal_radius_;
}
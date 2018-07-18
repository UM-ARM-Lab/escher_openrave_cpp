#include "Utilities.hpp"
#include <omp.h>

ContactSpacePlanning::ContactSpacePlanning(std::shared_ptr<RobotProperties> _robot_properties,
                                           std::vector< std::array<float,3> > _foot_transition_model,
                                           std::vector< std::array<float,2> > _hand_transition_model,
                                           std::vector< std::shared_ptr<TrimeshSurface> > _structures,
                                           std::map<int, std::shared_ptr<TrimeshSurface> > _structures_dict,
                                           std::shared_ptr<MapGrid> _map_grid,
                                           int _num_stance_in_state,
                                           int _thread_num,
                                           std::shared_ptr< DrawingHandler > _drawing_handler,
                                           int _planning_id):
robot_properties_(_robot_properties),
foot_transition_model_(_foot_transition_model),
hand_transition_model_(_hand_transition_model),
structures_(_structures),
structures_dict_(_structures_dict),
map_grid_(_map_grid),
num_stance_in_state_(_num_stance_in_state),
thread_num_(_thread_num),
drawing_handler_(_drawing_handler),
planning_id_(_planning_id)
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

    dynamics_optimizer_interface_vector_.resize(2 * std::max(foot_transition_model_.size(), hand_transition_model_.size()));

    for(int i = 0; i < dynamics_optimizer_interface_vector_.size(); i++)
    {
        dynamics_optimizer_interface_vector_[i] = std::make_shared<OptimizationInterface>(STEP_TRANSITION_TIME, "SL_optim_config_template/cfg_kdopt_demo.yaml");
    }
}

std::vector< std::shared_ptr<ContactState> > ContactSpacePlanning::ANAStarPlanning(std::shared_ptr<ContactState> initial_state, std::array<float,3> goal,
                                                                                   float goal_radius, PlanningHeuristicsType heuristics_type,
                                                                                   BranchingMethod branching_method,
                                                                                   float time_limit, bool output_first_solution, bool goal_as_exact_poses)
{
    // // define robot initial state
	// momentumopt::DynamicsState ini_state;
	// ini_state.fillInitialRobotState(cfg_file);

	// // create instances of momentum optimizers
	// momentumopt::PlannerSetting planner_setting;
	// momentumopt::DynamicsOptimizer dyn_optimizer;
	// momentumopt::KinematicsOptimizer kin_optimizer;
	// momentumopt::ContactPlanFromFile cnt_optimizer;

	// // initialize optimizers
	// planner_setting.initialize(cfg_file);
	// KinematicsInterfaceSl kin_interface(planner_setting.get(momentumopt::PlannerDoubleParam_Frequency));

	// cnt_optimizer.initialize(planner_setting, &kin_interface);
	// dyn_optimizer.initialize(planner_setting, &kin_interface);
	// kin_optimizer.initialize(planner_setting, &kin_interface);

	// // optimize motion
	// cnt_optimizer.optimize(ini_state, dyn_optimizer.dynamicsSequence());
	// dyn_optimizer.optimize(ini_state, kin_optimizer.kinematicsSequence(), false);
	// for (int iter_id=0; iter_id<planner_setting.get(momentumopt::PlannerIntParam_NumKinDynIterations); iter_id++) {
	//   kin_optimizer.optimize(ini_state, dyn_optimizer.dynamicsSequence(), iter_id>0);
	//   dyn_optimizer.optimize(ini_state, kin_optimizer.kinematicsSequence(), true);
	// }

    // Construct a map to map from edge to dynamics sequence

    // query this map to get the set of dynamics sequcne

    // concatenate the dynamics sequence

    // call the kinematics optimizer


    // initialize parameters
    G_ = 9999.0;
    E_ = G_;
    goal_ = goal;
    goal_radius_ = goal_radius;
    time_limit_ = time_limit;

    heuristics_type_ = heuristics_type;

    // Initialize the random number sampler
    std::mt19937_64 rng;
    // initialize the random number generator with time-dependent seed
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);
    // initialize a uniform distribution between 0 and 1
    std::uniform_real_distribution<double> double_unif(0, 1);

    // clear the heap and state list, and add initial state in the list
    while(!open_heap_.empty())
    {
        open_heap_.pop();
    }
    contact_states_map_.clear();

    open_heap_.push(initial_state);
    contact_states_map_.insert(std::make_pair(std::hash<ContactState>()(*initial_state), initial_state));

    auto time_before_ANA_start_planning = std::chrono::high_resolution_clock::now();

    // main exploration loop
    bool over_time_limit = false;
    bool goal_reached = false;
    std::shared_ptr<ContactState> goal_state;

    RAVELOG_INFO("Enter the exploration loop.\n");

    std::vector< std::tuple<int,float,float,float,int> > planning_result; // planning time, path cost, dynamics cost, step num

    while(!open_heap_.empty())
    {
        while(!open_heap_.empty())
        {
            auto current_time = std::chrono::high_resolution_clock::now();
            if(std::chrono::duration_cast<std::chrono::milliseconds>(current_time - time_before_ANA_start_planning).count() / 1000.0 > time_limit)
            {
                RAVELOG_INFO("Over time limit.\n");
                over_time_limit = true;
                break;
            }
            // get the state in the top of the heap
            std::shared_ptr<ContactState> current_state;
            if(double_unif(rng) >= epsilon_) // explore the top of the heap
            {
                current_state = open_heap_.top();
                open_heap_.pop();
            }
            else // randomly explore (* uniform random in the heap, not uniform random over the search space)
            {
                std::uniform_int_distribution<> int_unif(0, contact_states_map_.size()-1);
                auto random_it = std::next(std::begin(contact_states_map_), int_unif(rng));
                current_state = random_it->second;
            }

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

                // std::cout << current_state->explore_state_ << std::endl;
                // std::size_t current_state_hash = std::hash<ContactState>()(*current_state);
                // std::unordered_map<std::size_t, std::shared_ptr<ContactState> >::iterator contact_state_iterator = contact_states_map_.find(current_state_hash);
                // current_state->stances_vector_[0]->left_foot_pose_.printPose();
                // current_state->stances_vector_[0]->right_foot_pose_.printPose();
                // current_state->stances_vector_[0]->left_hand_pose_.printPose();
                // current_state->stances_vector_[0]->right_hand_pose_.printPose();
                // std::cout << current_state->stances_vector_[0]->ee_contact_status_[0] << " "
                //           << current_state->stances_vector_[0]->ee_contact_status_[1] << " "
                //           << current_state->stances_vector_[0]->ee_contact_status_[2] << " "
                //           << current_state->stances_vector_[0]->ee_contact_status_[3] << std::endl;
                // std::cout << "hash: " << current_state_hash << std::endl;
                // std::cout << "hash: " << std::hash<ContactState>()(*current_state) << std::endl;
                // std::cout << current_state->stances_vector_.size() << std::endl;

                // if (contact_state_iterator == contact_states_map_.end()) // the state is not in the set
                // {
                //     std::cout << "wait, what?" << std::endl;
                // }

                // std::cout << "===============" << std::endl;

                // contact_state_iterator->second->stances_vector_[0]->left_foot_pose_.printPose();
                // contact_state_iterator->second->stances_vector_[0]->right_foot_pose_.printPose();
                // contact_state_iterator->second->stances_vector_[0]->left_hand_pose_.printPose();
                // contact_state_iterator->second->stances_vector_[0]->right_hand_pose_.printPose();

                // std::cout << contact_state_iterator->second->explore_state_ << std::endl;
                // getchar();


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

                    int step_count = 0;
                    std::shared_ptr<ContactState> path_state = goal_state;
                    while(true)
                    {
                        if(path_state->is_root_)
                        {
                            break;
                        }

                        path_state = path_state->parent_;
                        step_count++;
                    }

                    current_time = std::chrono::high_resolution_clock::now();

                    float planning_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - time_before_ANA_start_planning).count() /1000.0;

                    RAVELOG_INFO("Solution Found: T = %5.3f, G = %5.3f, E = %5.3f, DynCost: %5.3f, # of Steps: %d. \n", planning_time, G_, E_, current_state->accumulated_dynamics_cost_, step_count);

                    planning_result.push_back(std::make_tuple(planning_id_, planning_time, G_, current_state->accumulated_dynamics_cost_, step_count));

                    // getchar();

                    if(!output_first_solution)
                    {
                        updateExploreStatesAndOpenHeap();
                    }

                    break;
                }

                // branch
                branchingSearchTree(current_state, branching_method);
            }
        }

        if(output_first_solution || over_time_limit)
        {
            break;
        }

    }

    // store the planning result
    std::ofstream planning_result_fstream("contact_planning_result_weight_0_3.txt",std::ofstream::app);
    for(auto intermediate_result : planning_result)
    {
        planning_result_fstream << std::get<0>(intermediate_result) << " "
                                << std::get<1>(intermediate_result) << " "
                                << std::get<2>(intermediate_result) << " "
                                << std::get<3>(intermediate_result) << " "
                                << std::get<4>(intermediate_result) << " ";
        planning_result_fstream << std::endl;
    }


    // retrace the paths from the final states
    std::vector< std::shared_ptr<ContactState> > contact_state_path;
    if(goal_reached)
    {
        if(over_time_limit)
        {
            RAVELOG_INFO("The time limit (%5.2f seconds) has been reached. Output the current best solution.\n",time_limit);
        }
        else if(!over_time_limit && !output_first_solution)
        {
            RAVELOG_WARN("Exhausted the search tree. Output the optimal solution.\n");
        }
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

        std::reverse(contact_state_path.begin(), contact_state_path.end());

        kinematicsVerification(contact_state_path);
    }
    else if(!over_time_limit)
    {
        RAVELOG_WARN("Exhausted the search tree. No solution found.\n");
    }
    else if(over_time_limit)
    {
        RAVELOG_ERROR("The time limit (%5.2f seconds) has been reached. No solution found.\n",time_limit);
    }

    drawing_handler_->ClearHandler();

    return contact_state_path;
}

void ContactSpacePlanning::kinematicsVerification(std::vector< std::shared_ptr<ContactState> > contact_state_path)
{

    std::cout << "Verify the kinematic feasibility of the resulting contact sequence." << std::endl;

    std::vector<momentumopt::DynamicsSequence> dynamics_sequence_vector;

    std::array<int,4> num_contacts = {1,1,0,0};

    for(auto & contact_state : contact_state_path)
    {
        if(!contact_state->is_root_)
        {
            dynamics_sequence_vector.push_back(contact_state->parent_edge_dynamics_sequence_);
            num_contacts[contact_state->prev_move_manip_]++;
        }
    }

    std::cout << "num contacts: (lf,rf) " << num_contacts[0] << " " << num_contacts[1] << std::endl;

    std::shared_ptr<OptimizationInterface> kinematics_optimization_interface =
    std::make_shared<OptimizationInterface>(STEP_TRANSITION_TIME, "SL_optim_config_template/cfg_kdopt_demo.yaml");

    kinematics_optimization_interface->updateContactSequence(contact_state_path);
    kinematics_optimization_interface->dynamicsSequenceConcatenation(dynamics_sequence_vector);

    // kinematics optimization
    std::cout << "Start the Kinematics Optimization." << std::endl;
    kinematics_optimization_interface->simplifiedKinematicsOptimization();
    std::cout << "Finished the Kinematics Optimization." << std::endl;
}

bool ContactSpacePlanning::kinematicsFeasibilityCheck(std::shared_ptr<ContactState> current_state)
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

bool ContactSpacePlanning::dynamicsFeasibilityCheck(std::shared_ptr<ContactState> current_state, float& dynamics_cost, int index)
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

        bool dynamically_feasible = dynamics_optimizer_interface_vector_[index]->simplifiedDynamicsOptimization(dynamics_cost);
        // bool dynamically_feasible = dynamics_optimizer_interface_vector_[index]->dynamicsOptimization(dynamics_cost);

        // bool simplified_dynamically_feasible = dynamics_optimizer_interface_vector_[index]->simplifiedDynamicsOptimization(dynamics_cost);
        // bool dynamically_feasible = dynamics_optimizer_interface_vector_[index]->dynamicsOptimization(dynamics_cost);
        // if(!simplified_dynamically_feasible && dynamically_feasible)
        // {
        //     std::cout << "weird thing happens." << std::endl;
        //     getchar();
        // }
        // std::cout << "===============================================================" << std::endl;

        if(dynamically_feasible)
        {
            // update com, com_dot, and parent edge dynamics sequence of the current_state
            dynamics_optimizer_interface_vector_[index]->updateStateCoM(current_state);
            dynamics_optimizer_interface_vector_[index]->recordEdgeDynamicsSequence(current_state);
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
    return (kinematicsFeasibilityCheck(current_state) && dynamicsFeasibilityCheck(current_state, dynamics_cost, index));
}

void ContactSpacePlanning::branchingSearchTree(std::shared_ptr<ContactState> current_state, BranchingMethod branching_method)
{
    std::vector<ContactManipulator> branching_manips;

    if(current_state->prev_move_manip_ != ContactManipulator::L_LEG)
    {
        branching_manips.push_back(ContactManipulator::L_LEG);
    }

    if(current_state->prev_move_manip_ != ContactManipulator::R_LEG)
    {
        branching_manips.push_back(ContactManipulator::R_LEG);
    }

    if(branching_method == BranchingMethod::CONTACT_PROJECTION)
    {
        // branching foot contacts
        branchingFootContacts(current_state, branching_manips);

        // branching hand contacts
        branchingHandContacts(current_state, branching_manips);
    }
    else if(branching_method == BranchingMethod::CONTACT_OPTIMIZATION)
    {

    }
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

    // RAVELOG_INFO("Total thread number: %d.\n",thread_num_);

    #pragma omp parallel num_threads(thread_num_) shared (branching_feet_combination, state_feasibility_check_result)
    {
        #pragma omp for schedule(static)
        for(int i = 0; i < branching_feet_combination.size(); i++)
        {
            RPYTF new_left_foot_pose, new_right_foot_pose;
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

            if(projection_is_successful)
            {
                // RAVELOG_INFO("construct state.\n");

                // construct the new state
                std::shared_ptr<Stance> new_stance(new Stance(new_left_foot_pose, new_right_foot_pose, new_left_hand_pose, new_right_hand_pose, new_ee_contact_status));

                std::shared_ptr<ContactState> new_contact_state(new ContactState(new_stance, current_state, move_manip, 1));

                // RAVELOG_INFO("state feasibility check.\n");

                float dynamics_cost = 0.0;
                if(stateFeasibilityCheck(new_contact_state, dynamics_cost, i))
                {
                    state_feasibility_check_result[i] = std::make_tuple(true, new_contact_state, dynamics_cost);
                }
                else
                {
                    state_feasibility_check_result[i] = std::make_tuple(false, new_contact_state, dynamics_cost);
                }
            }
            else
            {
                state_feasibility_check_result[i] = std::make_tuple(false, current_state, 0.0);
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
    std::unordered_map<std::size_t, std::shared_ptr<ContactState> >::iterator contact_state_iterator = contact_states_map_.find(current_state_hash);

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

            // std::cout << "======new branch======" << std::endl;
            // current_state->stances_vector_[0]->left_foot_pose_.printPose();
            // current_state->stances_vector_[0]->right_foot_pose_.printPose();
            // current_state->stances_vector_[0]->left_hand_pose_.printPose();
            // current_state->stances_vector_[0]->right_hand_pose_.printPose();
            // std::cout << current_state->stances_vector_[0]->ee_contact_status_[0] << " "
            //               << current_state->stances_vector_[0]->ee_contact_status_[1] << " "
            //               << current_state->stances_vector_[0]->ee_contact_status_[2] << " "
            //               << current_state->stances_vector_[0]->ee_contact_status_[3] << std::endl;
            // std::cout << current_state->stances_vector_.size() << std::endl;
            // std::cout << "hash: " << current_state_hash << std::endl;
            // getchar();

            current_state->accumulated_dynamics_cost_ = prev_state->accumulated_dynamics_cost_ + dynamics_cost;

            contact_states_map_.insert(std::make_pair(current_state_hash, current_state));
            open_heap_.push(current_state);
        }

    }
    else
    {
        std::shared_ptr<ContactState> existing_state = contact_state_iterator->second;

        // std::cout << "===============================" << std::endl;
        // std::cout << "existing state: F: " << existing_state->getF() << ", explore state: " << existing_state->explore_state_ << std::endl;
        // std::cout << "new state: F: " << current_state->getF() << ", explore state: " << current_state->explore_state_ << std::endl;
        // getchar();

        if(existing_state->explore_state_ != ExploreState::CLOSED && current_state->getF() < existing_state->getF())
        {
            if(existing_state->explore_state_ == ExploreState::EXPLORED)
            {
                existing_state->explore_state_ = ExploreState::REOPEN;
            }

            existing_state->g_ = current_state->g_;
            existing_state->parent_ = current_state->parent_;
            existing_state->prev_move_manip_ = current_state->prev_move_manip_;

            existing_state->accumulated_dynamics_cost_ = prev_state->accumulated_dynamics_cost_ + dynamics_cost;

            if(existing_state->getF() < G_)
            {
                if(existing_state->h_ != 0)
                {
                    existing_state->priority_value_ = (G_ - existing_state->g_) / existing_state->h_;
                }
                else
                {
                    existing_state->priority_value_ = (G_ - existing_state->g_) / 0.00001;
                }
                open_heap_.push(existing_state);
            }
        }
    }

}

float ContactSpacePlanning::getHeuristics(std::shared_ptr<ContactState> current_state)
{
    if(heuristics_type_ == PlanningHeuristicsType::EUCLIDEAN)
    {
        // float euclidean_distance_to_goal = std::sqrt(std::pow(current_state->com_(0) - goal_[0],2) + std::pow(current_state->com_(1) - goal_[1],2));
        float euclidean_distance_to_goal = std::sqrt(std::pow(current_state->mean_feet_position_[0] - goal_[0],2) +
                                                     std::pow(current_state->mean_feet_position_[1] - goal_[1],2));
        float step_cost_to_goal = step_cost_weight_ * (euclidean_distance_to_goal / robot_properties_->max_stride_);

        return (euclidean_distance_to_goal + step_cost_to_goal);
    }
    else if(heuristics_type_ == PlanningHeuristicsType::DIJKSTRA)
    {
        GridIndices3D cell_indices = map_grid_->positionsToIndices({current_state->mean_feet_position_[0], current_state->mean_feet_position_[1], current_state->getFeetMeanHorizontalYaw()});
        MapCell3D cell = map_grid_->cell_3D_list_[cell_indices[0]][cell_indices[1]][cell_indices[2]];

        if(cell.terrain_type_ == TerrainType::SOLID)
        {
            return cell.g_;
        }
        else
        {
            return 9999.0;
        }
    }
    else
    {
        RAVELOG_ERROR("Unknown heuristics type.\n");
    }
}

float ContactSpacePlanning::getEdgeCost(std::shared_ptr<ContactState> prev_state, std::shared_ptr<ContactState> current_state, float dynamics_cost)
{
    // float traveling_distance_cost = std::sqrt(std::pow(current_state->com_(0) - prev_state->com_(0), 2) + std::pow(current_state->com_(1) - prev_state->com_(1), 2));
    float traveling_distance_cost = std::sqrt(std::pow(current_state->mean_feet_position_[0] - prev_state->mean_feet_position_[0], 2) +
                                              std::pow(current_state->mean_feet_position_[1] - prev_state->mean_feet_position_[1], 2));
    float orientation_cost = 0.01 * fabs(current_state->getFeetMeanHorizontalYaw() - prev_state->getFeetMeanHorizontalYaw());
    float step_cost = step_cost_weight_;
    dynamics_cost = dynamics_cost_weight_ * dynamics_cost;

    return (traveling_distance_cost + orientation_cost + step_cost + dynamics_cost);
    // return (traveling_distance_cost + orientation_cost + step_cost);
    // return dynamics_cost;
}

void ContactSpacePlanning::updateExploreStatesAndOpenHeap()
{
    while(!open_heap_.empty())
    {
        open_heap_.pop();
    }

    for(auto & contact_state_hash_pair : contact_states_map_)
    {
        std::shared_ptr<ContactState> contact_state = contact_state_hash_pair.second;

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
    // return std::sqrt(std::pow(goal_[0]-current_state->com_(0), 2) + std::pow(goal_[1]-current_state->com_(1), 2)) <= goal_radius_;
    return std::sqrt(std::pow(goal_[0]-current_state->mean_feet_position_[0], 2) + std::pow(goal_[1]-current_state->mean_feet_position_[1], 2)) <= goal_radius_;
}
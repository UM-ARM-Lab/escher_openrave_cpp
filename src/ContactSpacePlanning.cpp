#include "Utilities.hpp"
// #include <omp.h>

ContactSpacePlanning::ContactSpacePlanning(std::shared_ptr<RobotProperties> _robot_properties,
                                           std::vector< std::array<float,3> > _foot_transition_model,
                                           std::vector< std::array<float,2> > _hand_transition_model,
                                           std::vector< std::shared_ptr<TrimeshSurface> > _structures,
                                           std::map<int, std::shared_ptr<TrimeshSurface> > _structures_dict,
                                           std::shared_ptr<MapGrid> _map_grid,
                                           std::shared_ptr<GeneralIKInterface> _general_ik_interface,
                                           int _num_stance_in_state,
                                           int _thread_num,
                                           std::shared_ptr< DrawingHandler > _drawing_handler,
                                           int _planning_id,
                                           bool _use_dynamics_planning,
                                           std::vector<std::pair<Vector3D, float> > _disturbance_samples,
                                           PlanningApplication _planning_application,
                                           bool _check_zero_step_capturability,
                                           bool _check_one_step_capturability,
                                           bool _check_contact_transition_feasibility):
robot_properties_(_robot_properties),
foot_transition_model_(_foot_transition_model),
hand_transition_model_(_hand_transition_model),
structures_(_structures),
structures_dict_(_structures_dict),
map_grid_(_map_grid),
num_stance_in_state_(_num_stance_in_state),
thread_num_(_thread_num),
drawing_handler_(_drawing_handler),
planning_id_(_planning_id),
use_dynamics_planning_(_use_dynamics_planning),
general_ik_interface_(_general_ik_interface),
disturbance_samples_(_disturbance_samples),
planning_application_(_planning_application),
check_zero_step_capturability_(_check_zero_step_capturability),
check_one_step_capturability_(_check_one_step_capturability),
check_contact_transition_feasibility_(_check_contact_transition_feasibility)
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

    RAVELOG_INFO("Initialize neural network interface.\n");
    // neural_network_interface_vector_.resize(2 * std::max(foot_transition_model_.size(), hand_transition_model_.size()));
    neural_network_interface_vector_.resize(1);

    for(unsigned int i = 0; i < neural_network_interface_vector_.size(); i++)
    {
        neural_network_interface_vector_[i] = std::make_shared<NeuralNetworkInterface>("../data/dynopt_result/objective_regression_nn_models/",
                                                                                       "../data/dynopt_result/feasibility_classification_nn_models/",
                                                                                       "../data/dynopt_result/zero_step_capture_feasibility_classification_nn_models/",
                                                                                       "../data/dynopt_result/one_step_capture_feasibility_classification_nn_models/");
    }

    RAVELOG_INFO("Initialize dynamics optimizer interface.\n");
    dynamics_optimizer_interface_vector_.resize(2 * (foot_transition_model_.size() + hand_transition_model_.size()));

    for(unsigned int i = 0; i < dynamics_optimizer_interface_vector_.size(); i++)
    {
        dynamics_optimizer_interface_vector_[i] = std::make_shared<OptimizationInterface>(STEP_TRANSITION_TIME, SUPPORT_PHASE_TIME, "../data/SL_optim_config_template/cfg_kdopt_demo.yaml");
    }

    one_step_capture_dynamics_optimizer_interface_vector_.resize(2 * (foot_transition_model_.size() + hand_transition_model_.size()));

    for(unsigned int i = 0; i < one_step_capture_dynamics_optimizer_interface_vector_.size(); i++)
    {
        one_step_capture_dynamics_optimizer_interface_vector_[i] = std::make_shared<OptimizationInterface>(0.5, 2.0, "../data/SL_optim_config_template/cfg_kdopt_demo_one_step_capturability.yaml",
                                                                                                           DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT);
    }

    zero_step_capture_dynamics_optimizer_interface_vector_.resize(2 * (foot_transition_model_.size() + hand_transition_model_.size()));

    for(unsigned int i = 0; i < zero_step_capture_dynamics_optimizer_interface_vector_.size(); i++)
    {
        zero_step_capture_dynamics_optimizer_interface_vector_[i] = std::make_shared<OptimizationInterface>(0.0, 2.0, "../data/SL_optim_config_template/cfg_kdopt_demo_one_step_capturability.yaml",
                                                                                                           DynOptApplication::ZERO_STEP_CAPTURABILITY_DYNOPT);
    }

    if(!use_dynamics_planning_)
    {
        general_ik_interface_vector_.resize(2 * std::max(foot_transition_model_.size(), hand_transition_model_.size()));

        for(int i = 0; i < general_ik_interface_vector_.size(); i++)
        {
            // OpenRAVE::EnvironmentBasePtr cloned_env = general_ik_interface_->env_->CloneSelf(OpenRAVE::Clone_Bodies);
            // general_ik_interface_vector_[i] = std::make_shared<GeneralIKInterface>(cloned_env, cloned_env->GetRobot(general_ik_interface_->robot_->GetName()));
            general_ik_interface_vector_[i] = general_ik_interface_;
        }
    }

    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng_.seed(ss);
}

std::vector< std::shared_ptr<ContactState> > ContactSpacePlanning::ANAStarPlanning(std::shared_ptr<ContactState> initial_state, std::array<float,3> goal,
                                                                                   float goal_radius, PlanningHeuristicsType heuristics_type,
                                                                                   BranchingMethod branching_method,
                                                                                   float time_limit, float epsilon,
                                                                                   bool output_first_solution, bool goal_as_exact_poses,
                                                                                   bool use_learned_dynamics_model, bool enforce_stop_in_the_end)
{
    RAVELOG_INFO("Start ANA* planning.\n");

    if(!feetReprojection(initial_state))
    {
        RAVELOG_INFO("The initial state is not feasible. Abort contact planning.\n");
        std::vector< std::shared_ptr<ContactState> > empty_path;
        return empty_path;
    }

    // initialize parameters
    G_ = 999999.0;
    E_ = G_;
    goal_ = goal;
    goal_radius_ = goal_radius;
    time_limit_ = time_limit;
    epsilon_ = epsilon;
    heuristics_type_ = heuristics_type;
    use_learned_dynamics_model_ = use_learned_dynamics_model;
    enforce_stop_in_the_end_ = enforce_stop_in_the_end;

    // initialize the random number generator with time-dependent seed
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng_.seed(ss);
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

    int drawing_counter = 0;

    std::vector< std::tuple<int,float,float,float,int> > planning_result; // planning time, path cost, dynamics cost, step num

    std::vector< std::vector<std::shared_ptr<ContactState> > > all_solution_contact_paths;
    std::vector<float> all_solution_planning_times;

    {
        OpenRAVE::EnvironmentMutex::scoped_lock lockenv(general_ik_interface_->env_->GetMutex());

        while(!open_heap_.empty())
        {
            while(!open_heap_.empty())
            {
                // auto time_start = std::chrono::high_resolution_clock::now();
                // std::cout << "*" << std::endl;

                auto current_time = std::chrono::high_resolution_clock::now();
                if(std::chrono::duration_cast<std::chrono::milliseconds>(current_time - time_before_ANA_start_planning).count() / 1000.0 > time_limit)
                {
                    RAVELOG_INFO("Over time limit.\n");
                    over_time_limit = true;
                    break;
                }
                // get the state in the top of the heap
                std::shared_ptr<ContactState> current_state;
                if(double_unif(rng_) >= epsilon_) // explore the top of the heap
                {
                    current_state = open_heap_.top();
                    open_heap_.pop();
                }
                else // randomly explore (* uniform random in the heap, not uniform random over the search space)
                {
                    std::uniform_int_distribution<> int_unif(0, contact_states_map_.size()-1);
                    auto random_it = std::next(std::begin(contact_states_map_), int_unif(rng_));
                    current_state = random_it->second;
                }

                // current_state = open_heap_.top();
                // open_heap_.pop();

                if(current_state->explore_state_ == ExploreState::OPEN || current_state->explore_state_ == ExploreState::REOPEN)
                {
                    // Collision Checking if needed

                    // Kinematic and dynamic feasibility check
                    float dummy_dynamics_cost = 0;
                    int dummy_index = 0;
                    if(!use_dynamics_planning_ && !stateFeasibilityCheck(current_state, dummy_dynamics_cost, dummy_index))
                    {
                        // getchar();
                        current_state->explore_state_ = ExploreState::CLOSED;
                        continue;
                    }

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
                    // drawing_handler_->ClearHandler();
                    // drawing_handler_->DrawContactPath(current_state);
                    if(drawing_counter == 10)
                    {
                        drawing_handler_->ClearHandler();
                        drawing_handler_->DrawContactPath(current_state);
                        drawing_counter = 0;
                    }

                    drawing_counter++;

                    // check if it reaches the goal
                    // RAVELOG_INFO("Check if it reaches the goal.\n");
                    if(isReachedGoal(current_state))
                    {
                        G_ = current_state->g_;
                        goal_state = current_state;
                        goal_reached = true;

                        int step_count = 0;
                        std::shared_ptr<ContactState> path_state = goal_state;

                        std::vector<std::shared_ptr<ContactState> > solution_contact_path;

                        while(true)
                        {
                            solution_contact_path.push_back(std::make_shared<ContactState>(*path_state));
                            if(path_state->is_root_)
                            {
                                break;
                            }

                            path_state = path_state->parent_;
                            step_count++;
                        }

                        std::reverse(solution_contact_path.begin(), solution_contact_path.end());
                        for(unsigned int i = 0; i < solution_contact_path.size(); i++)
                        {
                            if(!solution_contact_path[i]->is_root_)
                            {
                                solution_contact_path[i]->parent_ = solution_contact_path[i-1];
                            }
                        }
                        all_solution_contact_paths.push_back(solution_contact_path);

                        current_time = std::chrono::high_resolution_clock::now();

                        float planning_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - time_before_ANA_start_planning).count() /1000.0;

                        all_solution_planning_times.push_back(planning_time);

                        RAVELOG_INFO("Solution Found: T = %5.3f, G = %5.3f, E = %5.3f, DynCost: %5.3f, # of Steps: %d. \n", planning_time, G_, E_, current_state->accumulated_dynamics_cost_, step_count);

                        // getchar();

                        planning_result.push_back(std::make_tuple(planning_id_, planning_time, G_, current_state->accumulated_dynamics_cost_, step_count));

                        // getchar();

                        if(!output_first_solution)
                        {
                            updateExploreStatesAndOpenHeap();
                        }

                        break;
                    }

                    // getchar();
                    // std::cout << "current: g: " << current_state->g_ << ", h: " << current_state->h_ << ", priority: " << current_state->priority_value_ << std::endl;
                    // std::cout << "CoM: " << current_state->com_[0] << " " << current_state->com_[1] << " " << current_state->com_[2] << " " << std::endl;

                    // branch
                    // auto time_before_branching = std::chrono::high_resolution_clock::now();

                    branchingSearchTree(current_state, branching_method);

                    // auto time_after_branching = std::chrono::high_resolution_clock::now();
                    // std::cout << "verification time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_before_branching - time_start).count()/1000.0 << " ms" << std::endl;
                    // std::cout << "branching time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_after_branching - time_before_branching).count()/1000.0 << " ms" << std::endl;

                }
            }

            if(output_first_solution || over_time_limit)
            {
                break;
            }

        }

        general_ik_interface_->robot_->SetDOFValues(robot_properties_->IK_init_DOF_Values_);
    }

    // // store the planning result
    // std::ofstream planning_result_fstream("contact_planning_result_weight_3_0_test_env_5_2_epsilon_0_optimization.txt", std::ofstream::app);
    // for(auto intermediate_result : planning_result)
    // {
    //     planning_result_fstream << std::get<0>(intermediate_result) << " "
    //                             << std::get<1>(intermediate_result) << " "
    //                             << std::get<2>(intermediate_result) << " "
    //                             << std::get<3>(intermediate_result) << " "
    //                             << std::get<4>(intermediate_result) << " ";
    //     planning_result_fstream << std::endl;
    // }

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

        std::ofstream planning_dynamics_cost_fstream("contact_planning_test_results.txt", std::ofstream::app);

        float total_learned_dynamics_cost = contact_state_path[contact_state_path.size()-1]->accumulated_dynamics_cost_;
        float total_dynamics_cost_segment_by_segment = 0;
        float total_dynamics_cost = 0;
        // if(use_dynamics_planning_)
        // {
        //     kinematicsVerification_StateOnly(contact_state_path);
        //     std::cout << "Predicted CoM trajectory(Red)." << std::endl;
        //     // getchar();
        // }

        // total_dynamics_cost_segment_by_segment = fillDynamicsSequenceSegmentBySegment(contact_state_path);
        // std::cout << "Piecewise Optimization CoM trajectory(Green)." << std::endl;
        // getchar();

        // total_dynamics_cost = fillDynamicsSequence(contact_state_path);
        // std::cout << "Whole Contact Sequence Optimization CoM trajectory(Blue)." << std::endl;
        // getchar();

        float final_plan_planning_time = 0.0;
        int num_contact_sequence_tried = 0;

        auto time_before_dynopt = std::chrono::high_resolution_clock::now();

        for(int i = all_solution_contact_paths.size()-1; i >= 0; i--)
        {
            final_plan_planning_time = all_solution_planning_times[i];

            drawing_handler_->ClearHandler();
            drawing_handler_->DrawContactPath(all_solution_contact_paths[i][all_solution_contact_paths[i].size()-1]);
            std::cout << "Solution Path " << i << ": " << std::endl;

            total_learned_dynamics_cost = all_solution_contact_paths[i][all_solution_contact_paths[i].size()-1]->accumulated_dynamics_cost_;

            if(use_dynamics_planning_)
            {
                kinematicsVerification_StateOnly(all_solution_contact_paths[i]);
                std::cout << "Predicted CoM trajectory(Red)." << std::endl;
                // getchar();
            }

            // total_dynamics_cost_segment_by_segment = fillDynamicsSequenceSegmentBySegment(all_solution_contact_paths[i]);
            // std::cout << "Piecewise Optimization CoM trajectory(Green)." << std::endl;

            total_dynamics_cost = fillDynamicsSequence(all_solution_contact_paths[i]);
            std::cout << "Whole Contact Sequence Optimization CoM trajectory(Blue)." << std::endl;

            if(total_dynamics_cost != 99999.0)
            {
                getchar();
                num_contact_sequence_tried = all_solution_contact_paths.size() - i;

                std::ofstream planning_contact_list_fstream("contact_list.txt", std::ofstream::out);

                for(auto & state : all_solution_contact_paths[i])
                {
                    if(!state->is_root_)
                    {
                        int move_manip = int(state->prev_move_manip_);
                        planning_contact_list_fstream << move_manip << " "
                                                      << state->stances_vector_[0]->ee_contact_poses_[move_manip].x_ << " "
                                                      << state->stances_vector_[0]->ee_contact_poses_[move_manip].y_ << " "
                                                      << state->stances_vector_[0]->ee_contact_poses_[move_manip].z_ << " "
                                                      << state->stances_vector_[0]->ee_contact_poses_[move_manip].roll_ << " "
                                                      << state->stances_vector_[0]->ee_contact_poses_[move_manip].pitch_ << " "
                                                      << state->stances_vector_[0]->ee_contact_poses_[move_manip].yaw_ << " "
                                                      << state->com_[0] << " " << state->com_[1] << " " << state->com_[2] << std::endl;
                    }
                }

                break;
            }
        }

        auto time_after_dynopt = std::chrono::high_resolution_clock::now();

        std::cout << planning_id_ << " " << total_learned_dynamics_cost << " "
                                            << total_dynamics_cost_segment_by_segment << " "
                                            << total_dynamics_cost << " "
                                            << final_plan_planning_time << std::endl;

        planning_dynamics_cost_fstream << planning_id_ << " " << total_learned_dynamics_cost << " "
                                                                << total_dynamics_cost_segment_by_segment << " "
                                                                << total_dynamics_cost << " "
                                                                << final_plan_planning_time << " "
                                                                << num_contact_sequence_tried << " "
                                                                << std::chrono::duration_cast<std::chrono::microseconds>(time_after_dynopt - time_before_dynopt).count()/1000000.0;


        if(total_dynamics_cost != 99999.0)
        {
            planning_dynamics_cost_fstream << " ";
            planning_dynamics_cost_fstream << dynamics_optimizer_interface_vector_[0]->mean_min_force_dist_to_boundary_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_min_cop_dist_to_boundary_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_max_force_angle_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_max_lateral_force_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_mean_force_dist_to_boundary_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_mean_cop_dist_to_boundary_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_mean_force_angle_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_mean_lateral_force_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_force_rms_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_max_torque_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_mean_torque_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_lmom_x_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_lmom_y_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_lmom_z_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_lmom_norm_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_lmom_rate_x_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_lmom_rate_y_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_lmom_rate_z_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_lmom_rate_norm_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_amom_x_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_amom_y_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_amom_z_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_amom_norm_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_amom_rate_x_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_amom_rate_y_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_amom_rate_z_ << " "
                                           << dynamics_optimizer_interface_vector_[0]->mean_amom_rate_norm_;
        }

        planning_dynamics_cost_fstream << std::endl;

        // getchar();

        // for(int i = all_solution_contact_paths.size()-1; i >= 0; i--)
        // {
        //     std::cout << "Solution Path " << i << ": " << std::endl;
        //     verifyContactSequenceDynamicsFeasibilityPrediction(all_solution_contact_paths[i]);

        //     // total_dynamics_cost_segment_by_segment = fillDynamicsSequenceSegmentBySegment(all_solution_contact_paths[i]);
        //     // total_dynamics_cost = fillDynamicsSequence(all_solution_contact_paths[i]);

        //     // std::cout << planning_id_ << " " << all_solution_contact_paths[i][all_solution_contact_paths[i].size()-1]->accumulated_dynamics_cost_
        //     //                           << " " << total_dynamics_cost_segment_by_segment
        //     //                           << " " << total_dynamics_cost << std::endl;
        // }
    }
    else if(!over_time_limit)
    {
        RAVELOG_WARN("Exhausted the search tree. No solution found.\n");
    }
    else if(over_time_limit)
    {
        RAVELOG_ERROR("The time limit (%5.2f seconds) has been reached. No solution found.\n",time_limit);
    }

    // getchar();

    drawing_handler_->ClearHandler();

    return contact_state_path;
}

void ContactSpacePlanning::verifyContactSequenceDynamicsFeasibilityPrediction(std::vector< std::shared_ptr<ContactState> > contact_state_path)
{
    // get the dynamics score of the whole sequence, the com and com dot will be updated by this function
    float total_dynamics_cost = 0;
    int num_dynamically_feasible_segment = 0;

    for(auto & contact_state : contact_state_path)
    {
        if(!contact_state->is_root_)
        {
            float dynamics_cost = 0.0;
            bool dynamically_feasible;

            // update the state cost and CoM
            std::vector< std::shared_ptr<ContactState> > contact_state_sequence = {contact_state->parent_, contact_state};
            dynamics_optimizer_interface_vector_[0]->updateContactSequence(contact_state_sequence);
            dynamically_feasible = dynamics_optimizer_interface_vector_[0]->dynamicsOptimization(dynamics_cost);

            if(dynamically_feasible)
            {
                // update com, com_dot, and parent edge dynamics sequence of the current_state
                total_dynamics_cost += dynamics_cost;
                std::cout << "1 ";
                num_dynamically_feasible_segment++;
            }
            else
            {
                // RAVELOG_ERROR("The resulting dynamics sequence is not dynamically feasible.");
                std::cout << "0 ";
                // getchar();
            }

            dynamics_optimizer_interface_vector_[0]->storeDynamicsOptimizationResult(contact_state, dynamics_cost, dynamically_feasible, planning_id_);
        }
    }

    std::cout << std::endl << num_dynamically_feasible_segment << "/" <<  contact_state_path.size()-1 << " segments are feasible." << std::endl;

}

float ContactSpacePlanning::fillDynamicsSequenceSegmentBySegment(std::vector< std::shared_ptr<ContactState> > contact_state_path)
{
    // get the dynamics score of the whole sequence, the com and com dot will be updated by this function
    float total_dynamics_cost = 0;
    for(auto & contact_state : contact_state_path)
    {
        if(!contact_state->is_root_)
        {
            float dynamics_cost = 0.0;
            bool dynamically_feasible;

            // update the state cost and CoM
            std::vector< std::shared_ptr<ContactState> > contact_state_sequence = {contact_state->parent_, contact_state};
            dynamics_optimizer_interface_vector_[0]->updateContactSequence(contact_state_sequence);
            dynamically_feasible = dynamics_optimizer_interface_vector_[0]->dynamicsOptimization(dynamics_cost);

            if(dynamically_feasible)
            {
                // update com, com_dot, and parent edge dynamics sequence of the current_state
                dynamics_optimizer_interface_vector_[0]->updateStateCoM(contact_state);
                dynamics_optimizer_interface_vector_[0]->recordEdgeDynamicsSequence(contact_state);
                total_dynamics_cost += dynamics_cost;
            }
            else
            {
                RAVELOG_ERROR("The resulting dynamics sequence is not dynamically feasible.");
                // getchar();
                return 99999.0;
            }

            drawing_handler_->DrawLineSegment(contact_state->parent_->com_, contact_state->com_, {0,1,0,1});
        }

        drawing_handler_->DrawLocation(contact_state->com_, Vector3D(0,1,0));
        drawing_handler_->DrawArrow(contact_state->com_, contact_state->com_dot_, Vector3D(0,1,0));
    }

    std::cout << "Total Dynamics Cost: " << total_dynamics_cost << std::endl;

    return total_dynamics_cost;
}

float ContactSpacePlanning::fillDynamicsSequence(std::vector< std::shared_ptr<ContactState> > contact_state_path)
{
    // get the dynamics score of the whole sequence, the com and com dot will be updated by this function
    float total_dynamics_cost = 0;
    bool dynamically_feasible;

    // update the state cost and CoM
    dynamics_optimizer_interface_vector_[0]->updateContactSequence(contact_state_path);
    dynamically_feasible = dynamics_optimizer_interface_vector_[0]->dynamicsOptimization(total_dynamics_cost);

    if(!dynamically_feasible)
    {
        RAVELOG_ERROR("The resulting dynamics sequence is not dynamically feasible.");
        // getchar();
        return 99999.0;
    }

    dynamics_optimizer_interface_vector_[0]->drawCoMTrajectory(drawing_handler_, Vector3D(0,0,1));

    std::cout << "Total Dynamics Cost: " << total_dynamics_cost << std::endl;

    return total_dynamics_cost;
}

void ContactSpacePlanning::kinematicsVerification_StateOnly(std::vector< std::shared_ptr<ContactState> > contact_state_path)
{
    std::cout << "Verify the kinematic feasibility of the resulting contact sequence." << std::endl;

    // std::array<int,4> num_contacts = {1,1,0,0};

    for(auto & contact_state : contact_state_path)
    {
        if(!contact_state->is_root_)
        {
            // std::cout << "dynamics cost: "<< contact_state->accumulated_dynamics_cost_ - contact_state->parent_->accumulated_dynamics_cost_ << std::endl;
            drawing_handler_->DrawLineSegment(contact_state->parent_->com_, contact_state->com_, {1,0,0,1});
        }
        // std::cout << "com: " << contact_state->com_[0] << " " << contact_state->com_[1] << " " << contact_state->com_[2] << std::endl;
        // std::cout << "com_dot: " << contact_state->com_dot_[0] << " " << contact_state->com_dot_[1] << " " << contact_state->com_dot_[2] << std::endl;
        drawing_handler_->DrawLocation(contact_state->com_, Vector3D(1,0,0));
        drawing_handler_->DrawArrow(contact_state->com_, contact_state->com_dot_, Vector3D(1,0,0));
    }

    // // reachability check
    // general_ik_interface_->balanceMode() = OpenRAVE::BalanceMode::BALANCE_NONE;
    // general_ik_interface_->returnClosest() = true;
    // general_ik_interface_->exactCoM() = true;
    // general_ik_interface_->noRotation() = false;
    // general_ik_interface_->executeMotion() = true;
    // general_ik_interface_->robot_->SetDOFValues(robot_properties_->IK_init_DOF_Values_);
    // general_ik_interface_->robot_->GetActiveDOFValues(general_ik_interface_->q0());
    // std::pair<bool,std::vector<OpenRAVE::dReal> > ik_result;
    // int state_id = 1;
    // for(auto & contact_state : contact_state_path)
    // {
    //     general_ik_interface_->resetContactStateRelatedParameters();

    //     // get the pose of contacting end-effectors
    //     for(auto & manip : ALL_MANIPULATORS)
    //     {
    //         if(contact_state->stances_vector_[0]->ee_contact_status_[manip])
    //         {
    //             std::cout << robot_properties_->manipulator_name_map_[manip] << ": "
    //                       << contact_state->stances_vector_[0]->ee_contact_poses_[manip].x_ << " "
    //                       << contact_state->stances_vector_[0]->ee_contact_poses_[manip].y_ << " "
    //                       << contact_state->stances_vector_[0]->ee_contact_poses_[manip].z_ << std::endl;

    //             general_ik_interface_->addNewManipPose(robot_properties_->manipulator_name_map_[manip], contact_state->stances_vector_[0]->ee_contact_poses_[manip].GetRaveTransform());
    //             general_ik_interface_->addNewContactManip(robot_properties_->manipulator_name_map_[manip], MU);
    //         }
    //     }

    //     // get the CoM and transform it from SL frame to OpenRAVE frame
    //     Translation3D com = contact_state->com_;
    //     general_ik_interface_->CenterOfMass()[0] = com[0];
    //     general_ik_interface_->CenterOfMass()[1] = com[1];
    //     general_ik_interface_->CenterOfMass()[2] = com[2];
    //     ik_result = general_ik_interface_->solve();
    //     general_ik_interface_->q0() = ik_result.second;

    //     std::cout << "com: " << com[0] << ", " << com[1] << ", " << com[2] << std::endl;
    //     std::cout << "result: " << ik_result.first << std::endl;

    //     // getchar();
    // }
}

void ContactSpacePlanning::kinematicsVerification(std::vector< std::shared_ptr<ContactState> > contact_state_path)
{

    std::cout << "Verify the kinematic feasibility of the resulting contact sequence." << std::endl;

    std::vector<momentumopt::DynamicsSequence> dynamics_sequence_vector;

    // std::array<int,4> num_contacts = {1,1,0,0};

    for(auto & contact_state : contact_state_path)
    {
        if(!contact_state->is_root_)
        {
            std::cout << "dynamics cost: "<< contact_state->accumulated_dynamics_cost_ - contact_state->parent_->accumulated_dynamics_cost_ << std::endl;
            dynamics_sequence_vector.push_back(contact_state->parent_edge_dynamics_sequence_);
            // num_contacts[contact_state->prev_move_manip_]++;
            drawing_handler_->DrawLineSegment(contact_state->parent_->com_, contact_state->com_, {1,0,0,1});
        }
        std::cout << "com: " << contact_state->com_[0] << " " << contact_state->com_[1] << " " << contact_state->com_[2] << std::endl;
        std::cout << "com_dot: " << contact_state->com_dot_[0] << " " << contact_state->com_dot_[1] << " " << contact_state->com_dot_[2] << std::endl;
        drawing_handler_->DrawLocation(contact_state->com_, Vector3D(1,0,0));
        drawing_handler_->DrawArrow(contact_state->com_, contact_state->com_dot_, Vector3D(1,0,0));
    }

    /*********************************FAILED ATTEMPT TO USE KINEMATICS SOLVER BASED ON SL*********************************/
    // std::shared_ptr<OptimizationInterface> kinematics_optimization_interface =
    // std::make_shared<OptimizationInterface>(STEP_TRANSITION_TIME, "SL_optim_config_template/cfg_kdopt_demo.yaml");

    // kinematics_optimization_interface->updateContactSequence(contact_state_path);
    // kinematics_optimization_interface->dynamicsSequenceConcatenation(dynamics_sequence_vector);

    // // // kinematics optimization
    // // std::cout << "Start the Kinematics Optimization." << std::endl;
    // // kinematics_optimization_interface->simplifiedKinematicsOptimization();
    // // std::cout << "Finished the Kinematics Optimization." << std::endl;
    /*********************************FAILED ATTEMPT TO USE KINEMATICS SOLVER BASED ON SL*********************************/

    // reachability check
    general_ik_interface_->balanceMode() = OpenRAVE::BalanceMode::BALANCE_NONE;
    general_ik_interface_->returnClosest() = true;
    general_ik_interface_->exactCoM() = true;
    general_ik_interface_->noRotation() = false;
    general_ik_interface_->executeMotion() = true;
    general_ik_interface_->robot_->SetDOFValues(robot_properties_->IK_init_DOF_Values_);
    general_ik_interface_->robot_->GetActiveDOFValues(general_ik_interface_->q0());
    std::pair<bool,std::vector<OpenRAVE::dReal> > ik_result;
    int state_id = 1;
    for(auto & dynamics_sequence : dynamics_sequence_vector)
    {
        std::shared_ptr<ContactState> current_state = contact_state_path[state_id];
        std::shared_ptr<ContactState> prev_state = current_state->parent_;
        ContactManipulator moving_manipulator = current_state->prev_move_manip_;

        int dynamics_state_id = 0;
        for(auto & dynamics_state : dynamics_sequence.dynamicsSequence())
        {
            general_ik_interface_->resetContactStateRelatedParameters();
            // get the pose of contacting end-effectors
            std::cout << "moving manipulator: " << robot_properties_->manipulator_name_map_[moving_manipulator] << std::endl;
            for(auto & manip : ALL_MANIPULATORS)
            {
                if(manip != moving_manipulator && current_state->stances_vector_[0]->ee_contact_status_[manip])
                {
                    std::cout << robot_properties_->manipulator_name_map_[manip] << ": " << current_state->stances_vector_[0]->ee_contact_poses_[manip].x_ << " "
                              << current_state->stances_vector_[0]->ee_contact_poses_[manip].y_ << " "
                              << current_state->stances_vector_[0]->ee_contact_poses_[manip].z_ << std::endl;

                    general_ik_interface_->addNewManipPose(robot_properties_->manipulator_name_map_[manip], current_state->stances_vector_[0]->ee_contact_poses_[manip].GetRaveTransform());
                    general_ik_interface_->addNewContactManip(robot_properties_->manipulator_name_map_[manip], MU);
                }
                else if(manip == moving_manipulator)
                {
                    if(prev_state->stances_vector_[0]->ee_contact_status_[manip] && current_state->stances_vector_[0]->ee_contact_status_[manip])
                    {
                        double ratio = (1.0 * dynamics_state_id) / dynamics_sequence.dynamicsSequence().size();
                        OpenRAVE::Transform moving_foot_start = prev_state->stances_vector_[0]->ee_contact_poses_[manip].GetRaveTransform();
                        OpenRAVE::Transform moving_foot_goal = current_state->stances_vector_[0]->ee_contact_poses_[manip].GetRaveTransform();
                        OpenRAVE::Transform moving_foot_transform;
                        moving_foot_transform.rot = OpenRAVE::geometry::quatSlerp(moving_foot_start.rot, moving_foot_goal.rot, ratio);
                        moving_foot_transform.trans = moving_foot_start.trans * (1-ratio) + moving_foot_goal.trans * ratio;
                        moving_foot_transform.trans[2] += (1-2*fabs(ratio-0.5)) * 0.1;

                        general_ik_interface_->addNewManipPose(robot_properties_->manipulator_name_map_[manip], moving_foot_transform);
                    }
                }
            }

            // get the CoM and transform it from SL frame to OpenRAVE frame
            Translation3D com = transformPositionFromSLToOpenrave(dynamics_state.centerOfMass());
            general_ik_interface_->CenterOfMass()[0] = com[0];
            general_ik_interface_->CenterOfMass()[1] = com[1];
            general_ik_interface_->CenterOfMass()[2] = com[2];
            ik_result = general_ik_interface_->solve();
            general_ik_interface_->q0() = ik_result.second;

            std::cout << "com: " << com[0] << ", " << com[1] << ", " << com[2] << std::endl;
            std::cout << "result: " << ik_result.first << std::endl;

            getchar();
            dynamics_state_id++;
        }
        state_id++;
    }

}

void ContactSpacePlanning::setupStateReachabilityIK(std::shared_ptr<ContactState> current_state, std::shared_ptr<GeneralIKInterface> general_ik_interface)
{
    bool left_hand_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::L_ARM];
    bool right_hand_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::R_ARM];

    general_ik_interface->resetContactStateRelatedParameters();

    // Contact Manipulator Pose
    general_ik_interface->addNewManipPose("l_leg", current_state->stances_vector_[0]->left_foot_pose_.GetRaveTransform());
    general_ik_interface->addNewManipPose("r_leg", current_state->stances_vector_[0]->right_foot_pose_.GetRaveTransform());
    if(left_hand_in_contact)
    {
        general_ik_interface->addNewManipPose("l_arm", current_state->stances_vector_[0]->left_hand_pose_.GetRaveTransform());
    }
    if(right_hand_in_contact)
    {
        general_ik_interface->addNewManipPose("r_arm", current_state->stances_vector_[0]->right_hand_pose_.GetRaveTransform());
    }

    // Center of Mass
    std::array<OpenRAVE::dReal,3> com;
    general_ik_interface->CenterOfMass()[0] = current_state->mean_feet_position_[0];
    general_ik_interface->CenterOfMass()[1] = current_state->mean_feet_position_[1];
    general_ik_interface->CenterOfMass()[2] = current_state->mean_feet_position_[2] + robot_properties_->robot_z_;

    // Initial Configuration
    std::vector<OpenRAVE::dReal> init_config = robot_properties_->IK_init_DOF_Values_;
    init_config[robot_properties_->DOFName_index_map_["x_prismatic_joint"]] = current_state->mean_feet_position_[0];
    init_config[robot_properties_->DOFName_index_map_["y_prismatic_joint"]] = current_state->mean_feet_position_[1];
    init_config[robot_properties_->DOFName_index_map_["z_prismatic_joint"]] = current_state->mean_feet_position_[2] + 1.0;
    init_config[robot_properties_->DOFName_index_map_["yaw_revolute_joint"]] = current_state->getFeetMeanHorizontalYaw() * DEG2RAD - M_PI/2.0;
    general_ik_interface->robot_->SetDOFValues(init_config);
    general_ik_interface->robot_->GetActiveDOFValues(general_ik_interface->q0());

    general_ik_interface->balanceMode() = OpenRAVE::BalanceMode::BALANCE_NONE;
    general_ik_interface->returnClosest() = false;
    general_ik_interface->exactCoM() = false;
    general_ik_interface->noRotation() = false;
    general_ik_interface->executeMotion() = false;
}

bool ContactSpacePlanning::kinematicsFeasibilityCheck(std::shared_ptr<ContactState> current_state, int index)
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

    if(left_hand_in_contact)
    {
        Translation3D left_hand_position = current_state->stances_vector_[0]->left_hand_pose_.getXYZ();
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
        Translation3D right_hand_position = current_state->stances_vector_[0]->right_hand_pose_.getXYZ();
        Translation3D right_shoulder_position(0, -robot_properties_->shoulder_w_ / 2.0, robot_properties_->shoulder_z_);
        right_shoulder_position = (feet_mean_transform * right_shoulder_position.homogeneous()).block(0,0,3,1);
        float right_hand_to_shoulder_dist = (right_hand_position - right_shoulder_position).norm();

        if(right_hand_to_shoulder_dist > robot_properties_->max_arm_length_ || right_hand_to_shoulder_dist < robot_properties_->min_arm_length_)
        {
            return false;
        }
    }

    // cannot be the same contact poses
    if(!current_state->is_root_)
    {
        bool found_contact_difference = false;
        std::shared_ptr<ContactState> prev_state = current_state->parent_;
        for(auto & manip : ALL_MANIPULATORS)
        {
            if(prev_state->stances_vector_[0]->ee_contact_status_[manip] != current_state->stances_vector_[0]->ee_contact_status_[manip]) // the contact is made or broken
            {
                found_contact_difference = true;
                break;
            }
            else if(current_state->stances_vector_[0]->ee_contact_status_[manip]) // both previous state and current state have this contact
            {
                if(prev_state->stances_vector_[0]->ee_contact_poses_[manip] != current_state->stances_vector_[0]->ee_contact_poses_[manip])
                {
                    found_contact_difference = true;
                    break;
                }
            }
        }

        if(!found_contact_difference)
        {
            return false;
        }
    }

    // IK solver check
    // call generalIK
    if(!use_dynamics_planning_ && !current_state->is_root_)
    // if(false)
    {
        // reachability check
        std::shared_ptr<GeneralIKInterface> general_ik_interface = general_ik_interface_vector_[index];
        std::pair<bool,std::vector<OpenRAVE::dReal> > ik_result;

        setupStateReachabilityIK(current_state, general_ik_interface);

        ik_result = general_ik_interface->solve();

        if(!ik_result.first)
        {
            return false;
        }
        else
        {
            general_ik_interface->robot_->SetActiveDOFValues(ik_result.second);
            for(int axis_id = 0; axis_id < 3; axis_id++)
            {
                current_state->nominal_com_[axis_id] = general_ik_interface->robot_->GetCenterOfMass()[axis_id];
                // std::cout << current_state->nominal_com_[axis_id] << " ";
            }
            // std::cout << std::endl;
            // getchar();
        }


        // end-points IK
        ContactManipulator moving_manipulator = current_state->prev_move_manip_;

        // touching down
        general_ik_interface->q0() = ik_result.second;
        general_ik_interface->balanceMode() = OpenRAVE::BalanceMode::BALANCE_GIWC;
        general_ik_interface->reuseGIWC() = false;
        float weight = 0.0;
        general_ik_interface->CenterOfMass()[0] = 0;
        general_ik_interface->CenterOfMass()[1] = 0;
        general_ik_interface->CenterOfMass()[2] = 0;
        for(auto & manip : ALL_MANIPULATORS)
        {
            if(manip != moving_manipulator && current_state->stances_vector_[0]->ee_contact_status_[manip])
            {
                general_ik_interface->addNewContactManip(robot_properties_->manipulator_name_map_[manip], MU);

                if(manip == ContactManipulator::L_LEG || manip == ContactManipulator::R_LEG)
                {
                    general_ik_interface->CenterOfMass()[0] += current_state->stances_vector_[0]->ee_contact_poses_[manip].x_;
                    general_ik_interface->CenterOfMass()[1] += current_state->stances_vector_[0]->ee_contact_poses_[manip].y_;
                    general_ik_interface->CenterOfMass()[2] += current_state->stances_vector_[0]->ee_contact_poses_[manip].z_;
                    weight += 1;
                }
            }
        }

        // update target com
        general_ik_interface->CenterOfMass()[0] /= weight;
        general_ik_interface->CenterOfMass()[1] /= weight;
        general_ik_interface->CenterOfMass()[2] /= weight;
        general_ik_interface->CenterOfMass()[2] += robot_properties_->robot_z_;

        ik_result = general_ik_interface->solve();

        // std::cout << "l foot: " << current_state->stances_vector_[0]->ee_contact_poses_[ContactManipulator::L_LEG].x_ << " "
        //                         << current_state->stances_vector_[0]->ee_contact_poses_[ContactManipulator::L_LEG].y_ << " "
        //                         << current_state->stances_vector_[0]->ee_contact_poses_[ContactManipulator::L_LEG].z_ << std::endl;

        // std::cout << "r foot: " << current_state->stances_vector_[0]->ee_contact_poses_[ContactManipulator::R_LEG].x_ << " "
        //                         << current_state->stances_vector_[0]->ee_contact_poses_[ContactManipulator::R_LEG].y_ << " "
        //                         << current_state->stances_vector_[0]->ee_contact_poses_[ContactManipulator::R_LEG].z_ << std::endl;

        // std::cout << "com: " << general_ik_interface->robot_->GetCenterOfMass() << std::endl;

        // // std::cout << ik_result.first << std::endl;

        // getchar();

        if(!ik_result.first)
        {
            return false;
        }

        // taking off
        std::shared_ptr<ContactState> prev_state = current_state->parent_;
        setupStateReachabilityIK(prev_state, general_ik_interface);
        general_ik_interface->returnClosest() = true;
        ik_result = general_ik_interface->solve();

        general_ik_interface->q0() = ik_result.second;
        general_ik_interface->returnClosest() = false;
        general_ik_interface->balanceMode() = OpenRAVE::BalanceMode::BALANCE_GIWC;
        general_ik_interface->reuseGIWC() = true;
        weight = 0.0;
        general_ik_interface->CenterOfMass()[0] = 0;
        general_ik_interface->CenterOfMass()[1] = 0;
        general_ik_interface->CenterOfMass()[2] = 0;
        for(auto & manip : ALL_MANIPULATORS)
        {
            if(manip != moving_manipulator && prev_state->stances_vector_[0]->ee_contact_status_[manip])
            {
                general_ik_interface->addNewContactManip(robot_properties_->manipulator_name_map_[manip], MU);

                if(manip == ContactManipulator::L_LEG || manip == ContactManipulator::R_LEG)
                {
                    general_ik_interface->CenterOfMass()[0] += current_state->stances_vector_[0]->ee_contact_poses_[manip].x_;
                    general_ik_interface->CenterOfMass()[1] += current_state->stances_vector_[0]->ee_contact_poses_[manip].y_;
                    general_ik_interface->CenterOfMass()[2] += current_state->stances_vector_[0]->ee_contact_poses_[manip].z_;
                    weight += 1;
                }
            }
        }

        // update target com
        general_ik_interface->CenterOfMass()[0] /= weight;
        general_ik_interface->CenterOfMass()[1] /= weight;
        general_ik_interface->CenterOfMass()[2] /= weight;
        general_ik_interface->CenterOfMass()[2] += robot_properties_->robot_z_;

        ik_result = general_ik_interface->solve();

        // std::cout << "com: " << general_ik_interface->robot_->GetCenterOfMass() << std::endl;

        // std::cout << ik_result.first << std::endl;

        // getchar();

        if(!ik_result.first)
        {
            return false;
        }
    }

    return true;
}

bool ContactSpacePlanning::dynamicsFeasibilityCheck(std::shared_ptr<ContactState> current_state, float& dynamics_cost, int index)
{
    if(!current_state->is_root_)
    {
        dynamics_cost = 0.0;
        bool dynamically_feasible;

        if(use_learned_dynamics_model_ && !(enforce_stop_in_the_end_ && isReachedGoal(current_state)))
        {
            // float ground_truth_dynamics_cost;
            // // update the state cost and CoM
            // std::vector< std::shared_ptr<ContactState> > contact_state_sequence = {current_state->parent_, current_state};

            // dynamics_optimizer_interface_vector_[0ndex]->updateContactSequence(contact_state_sequence);

            // dynamically_feasible = dynamics_optimizer_interface_vector_[index]->dynamicsOptimization(ground_truth_dynamics_cost);

            // std::cout << "Dynamically feasible: " << dynamically_feasible << std::endl;

            // if(dynamically_feasible)
            // {
            //     std::cout << "Ground Truth Dynamics Cost: " << ground_truth_dynamics_cost;
            // }

            // auto time_before_dynamics_prediction = std::chrono::high_resolution_clock::now();
            dynamically_feasible = neural_network_interface_vector_[0]->predictContactTransitionDynamics(current_state, dynamics_cost);
            // auto time_after_dynamics_prediction = std::chrono::high_resolution_clock::now();
            // std::cout << "prediction time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_after_dynamics_prediction - time_before_dynamics_prediction).count()/1000.0 << " ms" << std::endl;

            // std::cout << ", Predicted Dynamics Cost: " << dynamics_cost;
            // if(dynamically_feasible)
            // {
            //     std::cout << ", Error: " << dynamics_cost - ground_truth_dynamics_cost;
            // }
            // std::cout << std::endl;
            // getchar();
        }
        else
        {
            // update the state cost and CoM

            dynamics_optimizer_interface_vector_[index]->step_transition_time_ = 0.5;
            dynamics_optimizer_interface_vector_[index]->support_phase_time_ = 0.5;
            std::vector< std::shared_ptr<ContactState> > contact_state_sequence = {current_state->parent_, current_state};
            dynamics_optimizer_interface_vector_[index]->updateContactSequence(contact_state_sequence);

            // float desired_speed = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); // random from 0 to 1 m/s
            // get the fake CoM translation
            // current state
            // double total_weight = 0.8;
            // Translation3D fake_next_com = 0.4 * current_state->stances_vector_[0]->left_foot_pose_.getXYZ() +
            //                               0.4 * current_state->stances_vector_[0]->right_foot_pose_.getXYZ();

            // if(current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::L_ARM])
            // {
            //     total_weight += 0.1;
            //     fake_next_com = fake_next_com +  0.1 * current_state->stances_vector_[0]->left_hand_pose_.getXYZ();
            // }

            // if(current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::R_ARM])
            // {
            //     total_weight += 0.1;
            //     fake_next_com = fake_next_com +  0.1 * current_state->stances_vector_[0]->right_hand_pose_.getXYZ();
            // }

            // fake_next_com /= total_weight;

            // Translation3D com_translation(fake_next_com[0]-current_state->parent_->com_[0],fake_next_com[1]-current_state->parent_->com_[1],0);

            // dynamics_optimizer_interface_vector_[index]->updateReferenceDynamicsSequence(com_translation, desired_speed);

            dynamically_feasible = dynamics_optimizer_interface_vector_[index]->dynamicsOptimization(dynamics_cost);

            if(dynamically_feasible)
            {
                // update com, com_dot, and parent edge dynamics sequence of the current_state
                dynamics_optimizer_interface_vector_[index]->updateStateCoM(current_state);
                dynamics_optimizer_interface_vector_[index]->recordEdgeDynamicsSequence(current_state);

                // std::cout << " " << dynamics_cost << std::endl;
                // getchar();
            }

            dynamics_optimizer_interface_vector_[index]->storeDynamicsOptimizationResult(current_state, dynamics_cost, dynamically_feasible, planning_id_);

            // dynamics_cost = std::numeric_limits<float>::max();
            // std::shared_ptr<ContactState> tmp_current_state = std::make_shared<ContactState>(*current_state);
            // std::shared_ptr<ContactState> tmp_prev_state = std::make_shared<ContactState>(*tmp_current_state->parent_);
            // tmp_current_state->parent_ = tmp_prev_state;
            // std::vector< std::shared_ptr<ContactState> > contact_state_sequence = {tmp_current_state->parent_, tmp_current_state};
            // dynamically_feasible = false;

            // for(float transition_time = 0.6; transition_time < 1.5; transition_time += 0.2)
            // {
            //     float tmp_dynamics_cost;
            //     dynamics_optimizer_interface_vector_[index]->step_transition_time_ = transition_time;
            //     dynamics_optimizer_interface_vector_[index]->support_phase_time_ = transition_time;
            //     dynamics_optimizer_interface_vector_[index]->updateContactSequence(contact_state_sequence);

            //     // bool dynamically_feasible = dynamics_optimizer_interface_vector_[index]->simplifiedDynamicsOptimization(dynamics_cost);
            //     bool tmp_dynamically_feasible = dynamics_optimizer_interface_vector_[index]->dynamicsOptimization(tmp_dynamics_cost);

            //     if(tmp_dynamically_feasible)
            //     {
            //         std::cout << dynamics_optimizer_interface_vector_[index]->step_transition_time_ << " " << dynamics_optimizer_interface_vector_[index]->support_phase_time_ << " " << tmp_dynamics_cost << std::endl;
            //         getchar();

            //         // update com, com_dot, and parent edge dynamics sequence of the current_state
            //         dynamics_optimizer_interface_vector_[index]->updateStateCoM(tmp_current_state);
            //         dynamics_optimizer_interface_vector_[index]->recordEdgeDynamicsSequence(tmp_current_state);

            //         dynamically_feasible = true;
            //         if(tmp_dynamics_cost < dynamics_cost)
            //         {
            //             dynamics_cost = tmp_dynamics_cost;
            //             dynamics_optimizer_interface_vector_[index]->updateStateCoM(current_state);
            //             dynamics_optimizer_interface_vector_[index]->recordEdgeDynamicsSequence(current_state);
            //         }
            //     }

            //     dynamics_optimizer_interface_vector_[index]->storeDynamicsOptimizationResult(tmp_current_state, tmp_dynamics_cost, tmp_dynamically_feasible, planning_id_);
            // }

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
    if(use_dynamics_planning_)
    {
        bool state_feasibility = kinematicsFeasibilityCheck(current_state, index) && dynamicsFeasibilityCheck(current_state, dynamics_cost, index);

        // bool left_hand_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::L_ARM];
        // bool right_hand_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::R_ARM];

        // for sampling contact transition mode 0 1
        // if(left_hand_in_contact || right_hand_in_contact)
        // {
        //     return false;
        // }

        // for sampling contact transition mode 2 3 4 5 6 7 8 9
        // if(!(left_hand_in_contact || right_hand_in_contact || current_state->is_root_))
        // {
        //     return false;
        // }

        // for sampling contact transition mode 2 3 4 5 6
        // if(!current_state->is_root_ && ((left_hand_in_contact && right_hand_in_contact) || (!left_hand_in_contact && !right_hand_in_contact)))
        // {
        //     return false;
        // }

        // for sampling contact transition mode 6 7 8 9
        // if(!current_state->is_root_ && !current_state->parent_->is_root_ && (!left_hand_in_contact || !right_hand_in_contact))
        // {
        //     return false;
        // }

        return state_feasibility;
    }
    else
    {
        return kinematicsFeasibilityCheck(current_state, index);
    }
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
    if(current_state->prev_move_manip_ != ContactManipulator::L_ARM)
    {
        branching_manips.push_back(ContactManipulator::L_ARM);
    }
    if(current_state->prev_move_manip_ != ContactManipulator::R_ARM)
    {
        branching_manips.push_back(ContactManipulator::R_ARM);
    }

    if(branching_method == BranchingMethod::CONTACT_PROJECTION)
    {
        // // branching foot contacts
        // branchingFootContacts(current_state, branching_manips);

        // // branching hand contacts
        // branchingHandContacts(current_state, branching_manips);

        branchingContacts(current_state, BranchingManipMode::ALL);
    }
    else if(branching_method == BranchingMethod::CONTACT_OPTIMIZATION)
    {

    }
}

void ContactSpacePlanning::branchingContacts(std::shared_ptr<ContactState> current_state, BranchingManipMode branching_mode)
{
    std::vector<ContactManipulator> branching_manips;
    std::vector< std::array<float,2> > hand_transition_model;

    if(branching_mode == BranchingManipMode::ALL)
    {
        branching_manips = ALL_MANIPULATORS;
        hand_transition_model = hand_transition_model_;
    }
    else if(branching_mode == BranchingManipMode::FEET_CONTACTS)
    {
        branching_manips = LEG_MANIPULATORS;
    }
    else if(branching_mode == BranchingManipMode::HAND_CONTACTS)
    {
        branching_manips = ARM_MANIPULATORS;
        hand_transition_model = hand_transition_model_;
    }
    else if(branching_mode == BranchingManipMode::BREAKING_HAND_CONTACTS)
    {
        branching_manips = ARM_MANIPULATORS;
        hand_transition_model.push_back({-99.0, -99.0});
    }
    else
    {
        RAVELOG_ERROR("Unknown branching mode.");
        getchar();
    }

    // remove the last branched manipulator from the branching manips list
    if(!current_state->is_root_)
    {
        branching_manips.erase(std::remove(branching_manips.begin(), branching_manips.end(), current_state->prev_move_manip_), branching_manips.end());
    }

    const float l_leg_horizontal_yaw = current_state->getLeftHorizontalYaw();
    const float r_leg_horizontal_yaw = current_state->getRightHorizontalYaw();
    const float mean_horizontal_yaw = current_state->getFeetMeanHorizontalYaw();
    const RotationMatrix robot_yaw_rotation = RPYToSO3(RPYTF(0, 0, 0, 0, 0, mean_horizontal_yaw));

    std::shared_ptr<Stance> current_stance = current_state->stances_vector_[0];
    RPYTF current_left_foot_pose = current_stance->left_foot_pose_;
    RPYTF current_right_foot_pose = current_stance->right_foot_pose_;
    RPYTF current_left_hand_pose = current_stance->left_hand_pose_;
    RPYTF current_right_hand_pose = current_stance->right_hand_pose_;
    std::array<bool,ContactManipulator::MANIP_NUM> current_ee_contact_status = current_stance->ee_contact_status_;

    float current_height = (current_stance->left_foot_pose_.z_ + current_stance->right_foot_pose_.z_) / 2.0;
    std::uniform_real_distribution<float> projection_height_unif(current_height - 0.15, current_height + 0.15);

    std::vector< std::shared_ptr<ContactState> > branching_states;
    std::vector<bool> has_branch(ContactManipulator::MANIP_NUM, false);
    // std::vector< std::vector<std::shared_ptr<ContactState> > > branching_states_by_move_manip(ContactManipulator::MANIP_NUM);

    // get all the possible branching states of the current state
    for(auto & move_manip : branching_manips)
    {
        if(move_manip == ContactManipulator::L_LEG || move_manip == ContactManipulator::R_LEG)
        {
            for(auto & step : foot_transition_model_)
            {
                std::array<float,6> l_foot_xyzrpy = current_stance->left_foot_pose_.getXYZRPY();
                std::array<float,6> r_foot_xyzrpy = current_stance->right_foot_pose_.getXYZRPY();

                if(move_manip == ContactManipulator::L_LEG)
                {
                    l_foot_xyzrpy[0] = r_foot_xyzrpy[0] + std::cos(r_leg_horizontal_yaw*DEG2RAD) * step[0] - std::sin(r_leg_horizontal_yaw*DEG2RAD) * step[1]; // x
                    l_foot_xyzrpy[1] = r_foot_xyzrpy[1] + std::sin(r_leg_horizontal_yaw*DEG2RAD) * step[0] + std::cos(r_leg_horizontal_yaw*DEG2RAD) * step[1]; // y
                    l_foot_xyzrpy[2] = 99.0; // z
                    l_foot_xyzrpy[3] = 0; // roll
                    l_foot_xyzrpy[4] = 0; // pitch
                    l_foot_xyzrpy[5] = r_leg_horizontal_yaw + step[2]; // yaw
                }
                else if(move_manip == ContactManipulator::R_LEG)
                {
                    r_foot_xyzrpy[0] = l_foot_xyzrpy[0] + std::cos(l_leg_horizontal_yaw*DEG2RAD) * step[0] - std::sin(l_leg_horizontal_yaw*DEG2RAD) * (-step[1]); // x
                    r_foot_xyzrpy[1] = l_foot_xyzrpy[1] + std::sin(l_leg_horizontal_yaw*DEG2RAD) * step[0] + std::cos(l_leg_horizontal_yaw*DEG2RAD) * (-step[1]); // y
                    r_foot_xyzrpy[2] = 99.0; // z
                    r_foot_xyzrpy[3] = 0; // roll
                    r_foot_xyzrpy[4] = 0; // pitch
                    r_foot_xyzrpy[5] = l_leg_horizontal_yaw - step[2]; // yaw
                }

                RPYTF new_left_foot_pose = RPYTF(l_foot_xyzrpy);
                RPYTF new_right_foot_pose = RPYTF(r_foot_xyzrpy);

                bool projection_is_successful;
                // RAVELOG_INFO("foot projection.\n");

                // do projection to find the projected feet poses
                if(planning_application_ == PlanningApplication::PLAN_IN_ENV)
                {
                    if(move_manip == ContactManipulator::L_LEG)
                    {
                        projection_is_successful = footProjection(move_manip, new_left_foot_pose);
                    }
                    else if(move_manip == ContactManipulator::R_LEG)
                    {
                        projection_is_successful = footProjection(move_manip, new_right_foot_pose);
                    }
                }
                else if(planning_application_ == PlanningApplication::COLLECT_DATA)
                {
                    float projection_height = projection_height_unif(rng_);
                    if(move_manip == ContactManipulator::L_LEG)
                    {
                        projection_is_successful = footPoseSampling(move_manip, new_left_foot_pose, projection_height);
                    }
                    else if(move_manip == ContactManipulator::R_LEG)
                    {
                        projection_is_successful = footPoseSampling(move_manip, new_right_foot_pose, projection_height);
                    }
                }

                if(projection_is_successful)
                {
                    // RAVELOG_INFO("construct state.\n");
                    // construct the new state
                    std::shared_ptr<Stance> new_stance = std::make_shared<Stance>(new_left_foot_pose, new_right_foot_pose, current_left_hand_pose, current_right_hand_pose, current_ee_contact_status);
                    std::shared_ptr<ContactState> new_contact_state = std::make_shared<ContactState>(new_stance, current_state, move_manip, 1, robot_properties_->robot_z_);
                    branching_states.push_back(new_contact_state);
                    has_branch[int(move_manip)] = true;
                    // branching_states_by_move_manip[move_manip].push_back(new_contact_state);
                }
            }
        }
        else if(move_manip == ContactManipulator::L_ARM || move_manip == ContactManipulator::R_ARM)
        {
            Translation3D relative_shoulder_position;
            Translation3D global_left_shoulder_position, global_right_shoulder_position;

            if(move_manip == ContactManipulator::L_ARM)
            {
                relative_shoulder_position[0] = 0;
                relative_shoulder_position[1] = robot_properties_->shoulder_w_/2.0;
                relative_shoulder_position[2] = robot_properties_->shoulder_z_;
                global_left_shoulder_position = robot_yaw_rotation * relative_shoulder_position + current_state->mean_feet_position_;
            }
            else if(move_manip == ContactManipulator::R_ARM)
            {
                relative_shoulder_position[0] = 0;
                relative_shoulder_position[1] = -robot_properties_->shoulder_w_/2.0;
                relative_shoulder_position[2] = robot_properties_->shoulder_z_;
                global_right_shoulder_position = robot_yaw_rotation * relative_shoulder_position + current_state->mean_feet_position_;
            }

            // iterate through the arm transition model
            for(auto & arm_orientation : hand_transition_model_)
            {
                std::array<bool,ContactManipulator::MANIP_NUM> new_ee_contact_status = current_ee_contact_status;
                RPYTF new_left_hand_pose = current_state->stances_vector_[0]->left_hand_pose_;
                RPYTF new_right_hand_pose = current_state->stances_vector_[0]->right_hand_pose_;
                bool projection_is_successful = false;

                if(arm_orientation[0] != -99.0) // making contact
                {
                    std::array<float,2> global_arm_orientation;

                    if(move_manip == ContactManipulator::L_ARM)
                    {
                        global_arm_orientation[0] = mean_horizontal_yaw + 90.0 - arm_orientation[0];
                        global_arm_orientation[1] = arm_orientation[1];
                    }
                    else if(move_manip == ContactManipulator::R_ARM)
                    {
                        global_arm_orientation[0] = mean_horizontal_yaw - 90.0 + arm_orientation[0];
                        global_arm_orientation[1] = arm_orientation[1];
                    }

                    new_left_hand_pose = current_stance->left_hand_pose_;
                    new_right_hand_pose = current_stance->right_hand_pose_;

                    if(planning_application_ == PlanningApplication::PLAN_IN_ENV)
                    {
                        if(move_manip == ContactManipulator::L_ARM)
                        {
                            projection_is_successful = handProjection(move_manip, global_left_shoulder_position, global_arm_orientation, new_left_hand_pose);
                        }
                        else if(move_manip == ContactManipulator::R_ARM)
                        {
                            projection_is_successful = handProjection(move_manip, global_right_shoulder_position, global_arm_orientation, new_right_hand_pose);
                        }
                    }
                    else if(planning_application_ == PlanningApplication::COLLECT_DATA)
                    {
                        if(move_manip == ContactManipulator::L_ARM)
                        {
                            projection_is_successful = handPoseSampling(move_manip, global_left_shoulder_position, global_arm_orientation, new_left_hand_pose);
                        }
                        else if(move_manip == ContactManipulator::R_ARM)
                        {
                            projection_is_successful = handPoseSampling(move_manip, global_right_shoulder_position, global_arm_orientation, new_right_hand_pose);
                        }
                    }

                    new_ee_contact_status[move_manip] = true;
                }
                else // breaking contact
                {
                    projection_is_successful = current_state->manip_in_contact(move_manip);
                    new_ee_contact_status[move_manip] = false;

                    if(move_manip == ContactManipulator::L_ARM)
                    {
                        new_left_hand_pose = RPYTF(-99.0, -99.0, -99.0, -99.0, -99.0, -99.0);
                    }
                    else if(move_manip == ContactManipulator::R_ARM)
                    {
                        new_right_hand_pose = RPYTF(-99.0, -99.0, -99.0, -99.0, -99.0, -99.0);
                    }
                }

                if(projection_is_successful)
                {
                    // construct the new state
                    std::shared_ptr<Stance> new_stance = std::make_shared<Stance>(current_left_foot_pose, current_right_foot_pose, new_left_hand_pose, new_right_hand_pose, new_ee_contact_status);
                    std::shared_ptr<ContactState> new_contact_state = std::make_shared<ContactState>(new_stance, current_state, move_manip, 1, robot_properties_->robot_z_);
                    branching_states.push_back(new_contact_state);
                    has_branch[int(move_manip)] = true;
                    // branching_states_by_move_manip[move_manip].push_back(new_contact_state);
                }
            }
        }
    }

    // take out manips which do not have a single branch, and avoid the unnecessary calculation of its disturbance cost.
    for(auto & manip : ALL_MANIPULATORS)
    {
        if(!has_branch[int(manip)])
        {
            branching_manips.erase(std::remove(branching_manips.begin(), branching_manips.end(), manip), branching_manips.end());
        }
    }

    // Find the forces rejected by each category of the moving manipulator
    std::vector< std::unordered_set<int> > failing_disturbances_by_manip(ContactManipulator::MANIP_NUM);
    std::vector<float> disturbance_costs(ContactManipulator::MANIP_NUM, 0.0);
    if(!disturbance_samples_.empty())
    {
        std::map< std::array<bool, ContactManipulator::MANIP_NUM>, std::unordered_set<int> > checked_zero_capture_state;
        for(auto & move_manip : branching_manips)
        {
            std::cout << "Branch manip: " << move_manip << std::endl;

            // get the initial state for this move_manip
            std::unordered_set<int> failing_disturbances;

            // construct the state for the floating moving end-effector
            std::array<bool,ContactManipulator::MANIP_NUM> ee_contact_status = current_state->stances_vector_[0]->ee_contact_status_;
            ee_contact_status[move_manip] = false;

            // the zero capture state has already been checked.
            if(checked_zero_capture_state.find(ee_contact_status) != checked_zero_capture_state.end())
            {
                failing_disturbances_by_manip[move_manip] = checked_zero_capture_state.find(ee_contact_status)->second;
                continue;
            }

            std::array<RPYTF, ContactManipulator::MANIP_NUM> ee_contact_poses = current_state->stances_vector_[0]->ee_contact_poses_;
            ee_contact_poses[move_manip] = RPYTF(-99.0, -99.0, -99.0, -99.0, -99.0, -99.0);

            std::shared_ptr<Stance> zero_step_capture_stance = std::make_shared<Stance>(ee_contact_poses[0], ee_contact_poses[1],
                                                                                        ee_contact_poses[2], ee_contact_poses[3],
                                                                                        ee_contact_status);

            Translation3D initial_com = current_state->com_;

            for(int disturb_id = 0; disturb_id < disturbance_samples_.size(); disturb_id++)
            {
                bool disturbance_rejected = false;
                auto disturbance = disturbance_samples_[disturb_id];
                Vector3D post_impact_com_dot = current_state->com_dot_ + disturbance.first;

                std::cout << "Disturbance ID: " << disturb_id << ", (" << disturbance.first[0] << ", " << disturbance.first[1] << ", " << disturbance.first[2] << ")" << std::endl;

                // zero step capturability
                std::shared_ptr<ContactState> zero_step_capture_contact_state = std::make_shared<ContactState>(zero_step_capture_stance, initial_com, post_impact_com_dot, 1);
                std::vector< std::shared_ptr<ContactState> > zero_step_capture_contact_state_sequence = {zero_step_capture_contact_state};

                std::cout << "Zero Step Capture Check:" << std::endl;

                if(check_zero_step_capturability_)
                {
                    bool zero_step_dynamically_feasible;
                    if(use_learned_dynamics_model_)
                    {
                        zero_step_dynamically_feasible = neural_network_interface_vector_[0]->predictZeroStepCaptureDynamics(zero_step_capture_contact_state);
                        std::cout << zero_step_dynamically_feasible << std::endl;
                    }
                    else
                    {
                        // drawing_handler_->ClearHandler();
                        // drawing_handler_->DrawContactPath(zero_step_capture_contact_state);
                        // // getchar();

                        zero_step_capture_dynamics_optimizer_interface_vector_[0]->updateContactSequence(zero_step_capture_contact_state_sequence);

                        float zero_step_dummy_dynamics_cost = 0.0;
                        zero_step_dynamically_feasible = zero_step_capture_dynamics_optimizer_interface_vector_[0]->dynamicsOptimization(zero_step_dummy_dynamics_cost);

                        // zero_step_capture_dynamics_optimizer_interface_vector_[0]->storeDynamicsOptimizationResult(zero_step_capture_contact_state, zero_step_dummy_dynamics_cost, zero_step_dynamically_feasible, planning_id_);
                    }

                    // zero_step_capture_dynamics_optimizer_interface_vector_[0]->updateContactSequence(zero_step_capture_contact_state_sequence);
                    // zero_step_capture_dynamics_optimizer_interface_vector_[0]->exportConfigFiles("../data/SL_optim_config_template/cfg_kdopt_demo_invdynkin_template.yaml",
                    //                   "/home/yuchi/amd_workspace_video/workspace/src/catkin/humanoids/humanoid_control/motion_planning/momentumopt_sl/momentumopt_athena/config/capture_test/cfg_kdopt_demo.yaml",
                    //                   "/home/yuchi/amd_workspace_video/workspace/src/catkin/humanoids/humanoid_control/motion_planning/momentumopt_sl/momentumopt_athena/config/capture_test/Objects.cf",
                    //                   robot_properties_);
                    // getchar();

                    if(zero_step_dynamically_feasible)
                    {
                        disturbance_rejected = true;
                        continue;
                    }
                }

                std::cout << "++++++++++++++++++++" << std::endl;

                // one step capturability
                if(check_one_step_capturability_)
                {
                    std::cout << "One Step Capture Check:" << std::endl;

                    for(int i = 0; i < branching_states.size(); i++)
                    {
                        std::shared_ptr<ContactState> branching_state = branching_states[i];
                        ContactManipulator capture_contact_manip = branching_state->prev_move_manip_;

                        // if(branching_state->prev_move_manip_ != move_manip)
                        // {
                        //     RAVELOG_ERROR("Moving manipulator info mismatch.\n");
                        //     getchar();
                        // }

                        if(branching_state->manip_in_contact(capture_contact_manip) &&
                           !zero_step_capture_contact_state->manip_in_contact(capture_contact_manip)) // only consider branches making a new contact
                        {
                            std::shared_ptr<ContactState> prev_contact_state = std::make_shared<ContactState>(*zero_step_capture_contact_state);

                            // construct the state for the floating moving end-effector
                            std::array<bool,ContactManipulator::MANIP_NUM> ee_contact_status = prev_contact_state->stances_vector_[0]->ee_contact_status_;
                            ee_contact_status[capture_contact_manip] = true;
                            std::array<RPYTF, ContactManipulator::MANIP_NUM> ee_contact_poses = prev_contact_state->stances_vector_[0]->ee_contact_poses_;
                            ee_contact_poses[capture_contact_manip] = branching_state->stances_vector_[0]->ee_contact_poses_[capture_contact_manip];

                            std::shared_ptr<Stance> one_step_capture_stance = std::make_shared<Stance>(ee_contact_poses[0], ee_contact_poses[1],
                                                                                                       ee_contact_poses[2], ee_contact_poses[3],
                                                                                                       ee_contact_status);

                            std::shared_ptr<ContactState> one_step_capture_contact_state = std::make_shared<ContactState>(one_step_capture_stance, prev_contact_state, capture_contact_manip, 1, robot_properties_->robot_z_);
                            std::vector< std::shared_ptr<ContactState> > one_step_capture_contact_state_sequence = {prev_contact_state, one_step_capture_contact_state};

                            // std::cout << "##########" << std::endl;
                            // std::cout << capture_contact_manip << std::endl;
                            // std::cout << one_step_capture_contact_state->prev_move_manip_ << std::endl;
                            // std::cout << "^^^^^^^^^" << std::endl;

                            bool one_step_dynamically_feasible;
                            if(use_learned_dynamics_model_)
                            {
                                one_step_dynamically_feasible = neural_network_interface_vector_[0]->predictOneStepCaptureDynamics(one_step_capture_contact_state);
                                std::cout << "Capture Manip: " << capture_contact_manip << ", " << one_step_dynamically_feasible << std::endl;
                            }
                            else
                            {
                                one_step_capture_dynamics_optimizer_interface_vector_[i]->updateContactSequence(one_step_capture_contact_state_sequence);

                                float one_step_dummy_dynamics_cost = 0.0;
                                one_step_dynamically_feasible = one_step_capture_dynamics_optimizer_interface_vector_[i]->dynamicsOptimization(one_step_dummy_dynamics_cost);

                                one_step_capture_dynamics_optimizer_interface_vector_[i]->storeDynamicsOptimizationResult(one_step_capture_contact_state, one_step_dummy_dynamics_cost, one_step_dynamically_feasible, planning_id_);

                                // drawing_handler_->ClearHandler();
                                // drawing_handler_->DrawContactPath(one_step_capture_contact_state);
                                // // getchar();
                            }

                            if(one_step_dynamically_feasible)
                            {
                                exportContactSequenceOptimizationConfigFiles(one_step_capture_dynamics_optimizer_interface_vector_[0],
                                                                             one_step_capture_contact_state_sequence,
                                                                             "../data/SL_optim_config_template/cfg_kdopt_demo_invdynkin_template.yaml",
                                                                             "/home/yuchi/amd_workspace_video/workspace/src/catkin/humanoids/humanoid_control/motion_planning/momentumopt_sl/momentumopt_athena/config/capture_test/cfg_kdopt_demo.yaml",
                                                                             "/home/yuchi/amd_workspace_video/workspace/src/catkin/humanoids/humanoid_control/motion_planning/momentumopt_sl/momentumopt_athena/config/capture_test/Objects.cf");

                                std::cout << "A one step capture has been recorded." << std::endl;
                                getchar();

                                disturbance_rejected = true;
                                break;
                            }
                        }
                    }
                }

                std::cout << "====================" << std::endl;
                // getchar();

                if(!disturbance_rejected)
                {
                    failing_disturbances.insert(disturb_id);
                }
            }

            std::cout << "Rejected Disturbance / Total Disturbance: " << disturbance_samples_.size() - failing_disturbances.size() << "/" << disturbance_samples_.size() << std::endl;
            getchar();

            checked_zero_capture_state.insert(std::make_pair(ee_contact_status, failing_disturbances));
            failing_disturbances_by_manip[move_manip] = failing_disturbances;
        }

        for(auto & manip_id : branching_manips)
        {
            for(auto disturb_id : failing_disturbances_by_manip[int(manip_id)])
            {
                disturbance_costs[int(manip_id)] += disturbance_samples_[disturb_id].second;
            }
        }
    }

    // RAVELOG_WARN("finish check capturability.\n");

    // Find the dynamics cost and capturability cost
    std::vector< std::tuple<bool, std::shared_ptr<ContactState>, float, float> > state_feasibility_check_result(branching_states.size());
    if(check_contact_transition_feasibility_)
    {
        for(int i = 0; i < branching_states.size(); i++)
        {
            std::shared_ptr<ContactState> branching_state = branching_states[i];
            float dynamics_cost = 0.0;
            float disturbance_cost = disturbance_costs[int(branching_state->prev_move_manip_)];
            if(!use_dynamics_planning_ || stateFeasibilityCheck(branching_state, dynamics_cost, i)) // we use lazy checking when not using dynamics planning
            {
                state_feasibility_check_result[i] = std::make_tuple(true, branching_state, dynamics_cost, disturbance_cost);
            }
            else
            {
                state_feasibility_check_result[i] = std::make_tuple(false, branching_state, dynamics_cost, disturbance_cost);
            }
        }
    }
    else
    {
        for(int i = 0; i < branching_states.size(); i++)
        {
            std::shared_ptr<ContactState> branching_state = branching_states[i];
            float dynamics_cost = 0.0;
            float disturbance_cost = disturbance_costs[int(branching_state->prev_move_manip_)];
            state_feasibility_check_result[i] = std::make_tuple(true, branching_states[i], dynamics_cost, disturbance_cost);
        }
    }

    for(auto & check_result : state_feasibility_check_result)
    {
        bool pass_state_feasibility_check = std::get<0>(check_result);
        std::shared_ptr<ContactState> branching_state = std::get<1>(check_result);
        float dynamics_cost = std::get<2>(check_result);
        float disturbance_cost = std::get<3>(check_result);

        if(pass_state_feasibility_check)
        {
            insertState(branching_state, dynamics_cost, disturbance_cost);
        }
    }
}


/***************************************************************************************************************/



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

    float current_height = (current_stance->left_foot_pose_.z_ + current_stance->right_foot_pose_.z_) / 2.0;
    std::uniform_real_distribution<float> projection_height_unif(current_height - 0.15, current_height + 0.15);

    // RAVELOG_INFO("Total thread number: %d.\n",thread_num_);

    // #pragma omp parallel num_threads(thread_num_) shared (branching_feet_combination, state_feasibility_check_result)
    {
        // #pragma omp for schedule(static)
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
            if(planning_application_ == PlanningApplication::PLAN_IN_ENV)
            {
                if(move_manip == ContactManipulator::L_LEG)
                {
                    projection_is_successful = footProjection(move_manip, new_left_foot_pose);
                }
                else if(move_manip == ContactManipulator::R_LEG)
                {
                    projection_is_successful = footProjection(move_manip, new_right_foot_pose);
                }
            }
            else if(planning_application_ == PlanningApplication::COLLECT_DATA)
            {
                float projection_height = projection_height_unif(rng_);
                if(move_manip == ContactManipulator::L_LEG)
                {
                    projection_is_successful = footPoseSampling(move_manip, new_left_foot_pose, projection_height);
                }
                else if(move_manip == ContactManipulator::R_LEG)
                {
                    projection_is_successful = footPoseSampling(move_manip, new_right_foot_pose, projection_height);
                }
            }

            if(projection_is_successful)
            {
                // RAVELOG_INFO("construct state.\n");

                // construct the new state
                std::shared_ptr<Stance> new_stance = std::make_shared<Stance>(new_left_foot_pose, new_right_foot_pose, new_left_hand_pose, new_right_hand_pose, new_ee_contact_status);

                std::shared_ptr<ContactState> new_contact_state = std::make_shared<ContactState>(new_stance, current_state, move_manip, 1, robot_properties_->robot_z_);

                // RAVELOG_INFO("state feasibility check.\n");

                float dynamics_cost = 0.0;
                if(!use_dynamics_planning_ || stateFeasibilityCheck(new_contact_state, dynamics_cost, i)) // we use lazy checking when not using dynamics planning
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
    const float mean_horizontal_yaw = current_state->getFeetMeanHorizontalYaw();
    const RotationMatrix robot_yaw_rotation = RPYToSO3(RPYTF(0, 0, 0, 0, 0, mean_horizontal_yaw));

    Translation3D relative_shoulder_position;
    Translation3D global_left_shoulder_position, global_right_shoulder_position;

    std::vector< std::pair<std::array<float,2>, ContactManipulator> > branching_hands_combination;

    std::shared_ptr<Stance> current_stance = current_state->stances_vector_[0];
    RPYTF new_left_foot_pose = current_stance->left_foot_pose_;
    RPYTF new_right_foot_pose = current_stance->right_foot_pose_;
    std::array<float,2> breaking_contact_indicator = {-99.0,-99.0};

    // assuming waist is fixed, which is wrong.
    for(auto & manip : branching_manips)
    {
        if(manip == ContactManipulator::L_ARM || manip == ContactManipulator::R_ARM)
        {
            if(manip == ContactManipulator::L_ARM)
            {
                relative_shoulder_position[0] = 0;
                relative_shoulder_position[1] = robot_properties_->shoulder_w_/2.0;
                relative_shoulder_position[2] = robot_properties_->shoulder_z_;
                global_left_shoulder_position = robot_yaw_rotation * relative_shoulder_position + current_state->mean_feet_position_;
            }
            else if(manip == ContactManipulator::R_ARM)
            {
                relative_shoulder_position[0] = 0;
                relative_shoulder_position[1] = -robot_properties_->shoulder_w_/2.0;
                relative_shoulder_position[2] = robot_properties_->shoulder_z_;
                global_right_shoulder_position = robot_yaw_rotation * relative_shoulder_position + current_state->mean_feet_position_;
            }

            // iterate through the arm transition model
            for(auto & arm_orientation : hand_transition_model_)
            {
                if(arm_orientation[0] != -99.0) // making contact
                {
                    std::array<float,2> global_arm_orientation;

                    if(manip == ContactManipulator::L_ARM)
                    {
                        global_arm_orientation[0] = mean_horizontal_yaw + 90.0 - arm_orientation[0];
                        global_arm_orientation[1] = arm_orientation[1];
                    }
                    else if(manip == ContactManipulator::R_ARM)
                    {
                        global_arm_orientation[0] = mean_horizontal_yaw - 90.0 + arm_orientation[0];
                        global_arm_orientation[1] = arm_orientation[1];
                    }

                    branching_hands_combination.push_back(std::make_pair(global_arm_orientation, manip));
                }
                else // breaking contact
                {
                    if(current_stance->ee_contact_status_[manip])
                    {
                        branching_hands_combination.push_back(std::make_pair(breaking_contact_indicator, manip));
                    }
                }
            }
        }
    }

    std::vector< std::tuple<bool, std::shared_ptr<ContactState>, float> > state_feasibility_check_result(branching_hands_combination.size());

    // #pragma omp parallel num_threads(thread_num_) shared (branching_hands_combination, state_feasibility_check_result)
    {
        // #pragma omp for schedule(static)
        for(int i = 0; i < branching_hands_combination.size(); i++)
        {
            auto hand_combination = branching_hands_combination[i];

            std::array<float,2> global_arm_orientation = hand_combination.first;
            ContactManipulator move_manip = hand_combination.second;

            bool projection_is_successful = false;

            std::array<bool,ContactManipulator::MANIP_NUM> new_ee_contact_status = current_stance->ee_contact_status_;
            RPYTF new_left_hand_pose = current_stance->left_hand_pose_;
            RPYTF new_right_hand_pose = current_stance->right_hand_pose_;

            if(global_arm_orientation[0] != -99.0)
            {
                if(planning_application_ == PlanningApplication::PLAN_IN_ENV)
                {
                    if(move_manip == ContactManipulator::L_ARM)
                    {
                        projection_is_successful = handProjection(move_manip, global_left_shoulder_position, global_arm_orientation, new_left_hand_pose);
                    }
                    else if(move_manip == ContactManipulator::R_ARM)
                    {
                        projection_is_successful = handProjection(move_manip, global_right_shoulder_position, global_arm_orientation, new_right_hand_pose);
                    }
                }
                else if(planning_application_ == PlanningApplication::COLLECT_DATA)
                {
                    if(move_manip == ContactManipulator::L_ARM)
                    {
                        projection_is_successful = handPoseSampling(move_manip, global_left_shoulder_position, global_arm_orientation, new_left_hand_pose);
                    }
                    else if(move_manip == ContactManipulator::R_ARM)
                    {
                        projection_is_successful = handPoseSampling(move_manip, global_right_shoulder_position, global_arm_orientation, new_right_hand_pose);
                    }
                }

                new_ee_contact_status[move_manip] = true;
            }
            else
            {
                projection_is_successful = true;
                new_ee_contact_status[move_manip] = false;

                if(move_manip == ContactManipulator::L_ARM)
                {
                    new_left_hand_pose = RPYTF(-99.0, -99.0, -99.0, -99.0, -99.0, -99.0);
                }
                else if(move_manip == ContactManipulator::R_ARM)
                {
                    new_right_hand_pose = RPYTF(-99.0, -99.0, -99.0, -99.0, -99.0, -99.0);
                }
            }

            if(projection_is_successful)
            {
                // construct the new state
                std::shared_ptr<Stance> new_stance = std::make_shared<Stance>(new_left_foot_pose, new_right_foot_pose, new_left_hand_pose, new_right_hand_pose, new_ee_contact_status);

                std::shared_ptr<ContactState> new_contact_state = std::make_shared<ContactState>(new_stance, current_state, move_manip, 1, robot_properties_->robot_z_);

                float dynamics_cost = 0.0;
                float capturability_cost = 0.0;
                if(!use_dynamics_planning_ || stateFeasibilityCheck(new_contact_state, dynamics_cost, i)) // we use lazy checking when not using dynamics planning
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

    for(auto & check_result: state_feasibility_check_result)
    {
        bool pass_state_feasibility_check = std::get<0>(check_result);
        std::shared_ptr<ContactState> new_contact_state = std::get<1>(check_result);
        float dynamics_cost = std::get<2>(check_result);

        if(pass_state_feasibility_check)
        {
            // dynamics_cost = 0.0;
            insertState(new_contact_state, dynamics_cost);
        }
    }
}

bool ContactSpacePlanning::footProjection(ContactManipulator& contact_manipulator, RPYTF& projection_pose)
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
        tmp_projection_transformation_matrix = structure->projection(projection_origin, projection_ray, contact_projection_yaw, contact_manipulator, robot_properties_, valid_contact);

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

bool ContactSpacePlanning::footPoseSampling(ContactManipulator& contact_manipulator, RPYTF& projection_pose, double height)
{
    const Translation3D projection_origin = projection_pose.getXYZ();
    const Translation3D projection_ray(0, 0, -1);
    float contact_projection_yaw = projection_pose.yaw_;

    TransformationMatrix highest_projection_transformation_matrix;

    std::uniform_real_distribution<double> double_unif(-20 * DEG2RAD, 20 * DEG2RAD);

    float normal_roll = double_unif(rng_);
    float normal_pitch = double_unif(rng_);

    float sin_roll = std::sin(normal_roll);
    float cos_roll = std::cos(normal_roll);
    float sin_pitch = std::sin(normal_pitch);
    float cos_pitch = std::cos(normal_pitch);

    std::shared_ptr<TrimeshSurface> projected_surface = std::make_shared<TrimeshSurface>(Translation3D(projection_origin[0], projection_origin[1], height),
                                                                                         Vector3D(sin_pitch, -cos_pitch*sin_roll, cos_pitch*cos_roll),
                                                                                         TrimeshType::GROUND);

    TransformationMatrix projection_transformation_matrix;
    bool valid_contact = true;
    projection_transformation_matrix = projected_surface->projection(projection_origin,
                                                                     projection_ray,
                                                                     contact_projection_yaw,
                                                                     contact_manipulator,
                                                                     robot_properties_,
                                                                     valid_contact);

    projection_pose = SE3ToXYZRPY(projection_transformation_matrix);

    return true;
}


bool ContactSpacePlanning::handProjection(ContactManipulator& contact_manipulator, Translation3D& shoulder_position, std::array<float,2>& arm_orientation, RPYTF& projection_pose)
{
    float arm_length = 9999.0;
    bool found_projection_surface = false;
    Vector3D arm_projection_ray(std::cos(arm_orientation[0] * DEG2RAD) * std::cos(arm_orientation[1] * DEG2RAD),
                                std::sin(arm_orientation[0] * DEG2RAD) * std::cos(arm_orientation[1] * DEG2RAD),
                                std::sin(arm_orientation[1] * DEG2RAD));

    std::shared_ptr<TrimeshSurface> projected_surface;
    TransformationMatrix closest_projection_transformation_matrix;

    for(auto & structure : hand_structures_)
    {
        TransformationMatrix tmp_projection_transformation_matrix;
        bool valid_contact = false;
        tmp_projection_transformation_matrix = structure->projection(shoulder_position, arm_projection_ray, 0.0, contact_manipulator, robot_properties_, valid_contact);

        if(valid_contact)
        {
            double temp_arm_legnth = (tmp_projection_transformation_matrix.block(0,3,3,1) - shoulder_position).norm();
            if(!found_projection_surface || temp_arm_legnth < arm_length)
            {
                arm_length = temp_arm_legnth;
                closest_projection_transformation_matrix = tmp_projection_transformation_matrix;
                projected_surface = structure;
            }

            found_projection_surface = true;
        }
    }

    if(found_projection_surface)
    {
        projection_pose = SE3ToXYZRPY(closest_projection_transformation_matrix);
    }

    return found_projection_surface;
}

bool ContactSpacePlanning::handPoseSampling(ContactManipulator& contact_manipulator,
                                            Translation3D& shoulder_position,
                                            std::array<float,2>& arm_orientation,
                                            RPYTF& projection_pose)
{

    // std::uniform_real_distribution<double> arm_length_unif(robot_properties_->min_arm_length_, robot_properties_->max_arm_length_);
    std::uniform_real_distribution<double> arm_length_unif(0.2, 0.5);

    float arm_length = arm_length_unif(rng_);
    Vector3D arm_projection_ray(std::cos(arm_orientation[0] * DEG2RAD) * std::cos(arm_orientation[1] * DEG2RAD),
                                std::sin(arm_orientation[0] * DEG2RAD) * std::cos(arm_orientation[1] * DEG2RAD),
                                std::sin(arm_orientation[1] * DEG2RAD));

    std::uniform_real_distribution<double> orientation_unif(-20 * DEG2RAD, 20 * DEG2RAD);

    float normal_roll, normal_pitch;

    if(contact_manipulator == ContactManipulator::L_ARM)
    {
        normal_roll = 90 * DEG2RAD + orientation_unif(rng_);
        normal_pitch = orientation_unif(rng_);
    }
    else if(contact_manipulator == ContactManipulator::R_ARM)
    {
        normal_roll = -90 * DEG2RAD + orientation_unif(rng_);
        normal_pitch = orientation_unif(rng_);
    }

    float sin_roll = std::sin(normal_roll);
    float cos_roll = std::cos(normal_roll);
    float sin_pitch = std::sin(normal_pitch);
    float cos_pitch = std::cos(normal_pitch);

    Vector3D normal(sin_pitch, -cos_pitch*sin_roll, cos_pitch*cos_roll);

    // std::cout << int(contact_manipulator) << std::endl;
    // std::cout << shoulder_position + arm_length * arm_projection_ray << std::endl;
    // std::cout << normal << std::endl;
    // getchar();

    std::shared_ptr<TrimeshSurface> projected_surface = std::make_shared<TrimeshSurface>(shoulder_position + arm_length * arm_projection_ray,
                                                                                         normal,
                                                                                         TrimeshType::OTHERS);
    TransformationMatrix projection_transformation_matrix;

    bool valid_contact = true;
    projection_transformation_matrix = projected_surface->projection(shoulder_position,
                                                                     arm_projection_ray,
                                                                     0.0,
                                                                     contact_manipulator,
                                                                     robot_properties_,
                                                                     valid_contact);

    projection_pose = SE3ToXYZRPY(projection_transformation_matrix);

    return true;

}

bool ContactSpacePlanning::feetReprojection(std::shared_ptr<ContactState> state)
{
    std::shared_ptr<Stance> current_stance = state->stances_vector_[0];
    bool has_projection = true;
    std::array<RPYTF,ContactManipulator::MANIP_NUM> reprojected_ee_contact_poses = current_stance->ee_contact_poses_;

    for(int i = 0; i < ContactManipulator::MANIP_NUM; i++)
    {
        ContactManipulator move_manip = (ContactManipulator)i;
        if(current_stance->ee_contact_status_[i] &&
           (move_manip == ContactManipulator::L_LEG || move_manip == ContactManipulator::R_LEG))
        {
            reprojected_ee_contact_poses[i].z_ = 99.0;
            has_projection = footProjection(move_manip, reprojected_ee_contact_poses[i]);

            if(!has_projection)
            {
                break;
            }
        }
    }

    if(has_projection)
    {
        std::shared_ptr<Stance> reprojected_stance = std::make_shared<Stance>(reprojected_ee_contact_poses[0],
                                                                              reprojected_ee_contact_poses[1],
                                                                              reprojected_ee_contact_poses[2],
                                                                              reprojected_ee_contact_poses[3],
                                                                              current_stance->ee_contact_status_);
        state->stances_vector_.clear();
        state->stances_vector_.push_back(reprojected_stance);
    }

    return has_projection;
}

void ContactSpacePlanning::insertState(std::shared_ptr<ContactState> current_state, float dynamics_cost, float disturbance_cost)
{
    std::shared_ptr<ContactState> prev_state = current_state->parent_;
    // calculate the edge cost and the cost to come
    current_state->g_ = prev_state->g_ + getEdgeCost(prev_state, current_state, dynamics_cost, disturbance_cost);

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
        // RAVELOG_INFO("Existing state.\n");
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
                current_state->priority_value_ = existing_state->priority_value_;
            }
        }
    }

    // std::cout << "branching: g: " << current_state->g_ << ", h: " << current_state->h_ << ", priority: " << current_state->priority_value_ << ", dynamics_cost: " << dynamics_cost << ", prev_move_manip: " << current_state->prev_move_manip_ << std::endl;
    // std::cout << "CoM: " << current_state->com_[0] << " " << current_state->com_[1] << " " << current_state->com_[2] << " " << std::endl;
    // std::cout << "L_FOOT: " << current_state->stances_vector_[0]->left_foot_pose_.x_ << " " << current_state->stances_vector_[0]->left_foot_pose_.y_ << " " << current_state->stances_vector_[0]->left_foot_pose_.y_ << " " << std::endl;
    // std::cout << "R_FOOT: " << current_state->stances_vector_[0]->right_foot_pose_.x_ << " " << current_state->stances_vector_[0]->right_foot_pose_.y_ << " " << current_state->stances_vector_[0]->right_foot_pose_.y_ << " " << std::endl;
    // std::cout << "L_HAND: " << current_state->stances_vector_[0]->left_hand_pose_.x_ << " " << current_state->stances_vector_[0]->left_hand_pose_.y_ << " " << current_state->stances_vector_[0]->left_hand_pose_.y_ << " " << std::endl;

}

float ContactSpacePlanning::getHeuristics(std::shared_ptr<ContactState> current_state)
{
    if(heuristics_type_ == PlanningHeuristicsType::EUCLIDEAN)
    {
        // float euclidean_distance_to_goal = std::sqrt(std::pow(current_state->com_(0) - goal_[0],2) + std::pow(current_state->com_(1) - goal_[1],2));
        float euclidean_distance_to_goal = fabs(current_state->max_manip_x_ - goal_[0]);
        // float euclidean_distance_to_goal = std::sqrt(std::pow(current_state->mean_feet_position_[0] - goal_[0],2) +
        //                                              std::pow(current_state->mean_feet_position_[1] - goal_[1],2));
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

float ContactSpacePlanning::getEdgeCost(std::shared_ptr<ContactState> prev_state, std::shared_ptr<ContactState> current_state, float dynamics_cost, float disturbance_cost)
{
    // float traveling_distance_cost = std::sqrt(std::pow(current_state->com_(0) - prev_state->com_(0), 2) + std::pow(current_state->com_(1) - prev_state->com_(1), 2));
    // float traveling_distance_cost = std::sqrt(std::pow(current_state->mean_feet_position_[0] - prev_state->mean_feet_position_[0], 2) +
    //                                           std::pow(current_state->mean_feet_position_[1] - prev_state->mean_feet_position_[1], 2));
    float traveling_distance_cost = current_state->max_manip_x_ - prev_state->max_manip_x_;
    float orientation_cost = 0.01 * fabs(current_state->getFeetMeanHorizontalYaw() - prev_state->getFeetMeanHorizontalYaw());
    float step_cost = step_cost_weight_;

    // if(prev_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::L_ARM] || current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::R_ARM])
    // {
    //     step_cost = 0.001 * step_cost;
    // }

    // if(current_state->prev_move_manip_ == ContactManipulator::L_LEG)
    // {
    //     step_cost = step_cost * 0.9;
    // }

    // if(prev_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::L_ARM] || current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::R_ARM])
    // {
    //     return 0.1 * (traveling_distance_cost + orientation_cost + step_cost + dynamics_cost_weight_ * dynamics_cost);
    // }

    return (traveling_distance_cost + orientation_cost + step_cost + dynamics_cost_weight_ * dynamics_cost + disturbance_cost_weight_ * disturbance_cost);
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
    // return current_state->mean_feet_position_[0] > goal_[0]-goal_radius_;
    // return current_state->max_manip_x_ > (goal_[0] - goal_radius_);
}

void ContactSpacePlanning::storeSLEnvironment() // for mixed integer implementation
{
    // std::array<float,2> foot_erosion = {0.11, 0.055};
    // std::array<float,2> hand_erosion = {0.045, 0.045};

    std::array<float,2> foot_erosion = {0.0, 0.0};
    std::array<float,2> hand_erosion = {0.0, 0.0};

    std::ofstream SL_region_list_fstream("SL_region_list_" + std::to_string(planning_id_) + ".txt", std::ofstream::out);

    SL_region_list_fstream << "num_regions: " << structures_.size() << std::endl;
    for(unsigned int i = 0; i < structures_.size(); i++)
    {
        SL_region_list_fstream << "region" << i << ": [";
        std::shared_ptr<TrimeshSurface> structure = structures_[i];
        std::vector<Translation2D> proj_vertices = structure->getProjVertices();
        for(unsigned int j = 0; j < 4; j++)
        {
            SL_region_list_fstream << "[";
            Eigen::Vector4f homogeneous_proj_vertex;

            std::array<float,2> eroded_vertex;
            for(unsigned int dim = 0; dim < 2; dim++)
            {
                if(proj_vertices[j][dim] > 0)
                {
                    eroded_vertex[dim] = float(proj_vertices[j][dim] - foot_erosion[dim]);
                    // eroded_vertex[dim] = std::max(0.0, float(proj_vertices[j][dim] - foot_erosion[dim]));
                }
                else
                {
                    eroded_vertex[dim] = float(proj_vertices[j][dim] + foot_erosion[dim]);
                    // eroded_vertex[dim] = std::min(0.0, float(proj_vertices[j][dim] + foot_erosion[dim]));
                }
            }

            homogeneous_proj_vertex << eroded_vertex[0], eroded_vertex[1], 0, 1;
            Translation3D openrave_vertex = (structure->getTransform()*homogeneous_proj_vertex).block(0,0,3,1);
            Eigen::Vector3d SL_vertex = transformPositionFromOpenraveToSL(openrave_vertex);
            SL_region_list_fstream << SL_vertex[0] << ", " << SL_vertex[1] << ", " << SL_vertex[2] << "]";

            if(j < 3)
            {
                SL_region_list_fstream << ", ";
            }
        }

        SL_region_list_fstream << "]" << std::endl;
    }

}

void ContactSpacePlanning::exportContactSequenceOptimizationConfigFiles(std::shared_ptr<OptimizationInterface> optimizer_interface,
                                                                        std::vector< std::shared_ptr<ContactState> > contact_sequence,
                                                                        std::string optimization_config_template_path,
                                                                        std::string optimization_config_output_path,
                                                                        std::string objects_config_output_path)
{
    // construct the initial state for the optimization
    setupStateReachabilityIK(contact_sequence[0], general_ik_interface_);
    general_ik_interface_->returnClosest() = true;
    general_ik_interface_->executeMotion() = true;
    std::pair<bool,std::vector<OpenRAVE::dReal> > ik_result = general_ik_interface_->solve();

    std::vector<OpenRAVE::dReal> init_robot_config;
    general_ik_interface_->robot_->GetDOFValues(init_robot_config);

    std::map<ContactManipulator, RPYTF> floating_initial_contact_poses;

    for(auto manip : ALL_MANIPULATORS)
    {
        if(!contact_sequence[0]->manip_in_contact(manip))
        {
            std::string manip_name = robot_properties_->manipulator_name_map_[manip];
            OpenRAVE::Transform floating_manip_transform = general_ik_interface_->robot_->GetManipulator(manip_name)->GetTransform();
            TransformationMatrix floating_manip_transform_matrix = constructTransformationMatrix(floating_manip_transform);
            RPYTF floating_manip_pose = SE3ToXYZRPY(floating_manip_transform_matrix);

            floating_initial_contact_poses[manip] = floating_manip_pose;
        }
    }

    // generate the files
    optimizer_interface->updateContactSequence(contact_sequence);
    optimizer_interface->exportConfigFiles(optimization_config_template_path, optimization_config_output_path,
                                           objects_config_output_path, floating_initial_contact_poses,
                                           robot_properties_, init_robot_config);

}

void getAllContactPoseCombinations(std::vector< std::vector<RPYTF> >& all_contact_pose_combinations, const std::vector<std::vector<RPYTF> >& possible_contact_pose_representation, size_t vec_index, std::vector<RPYTF>& contact_pose_combination)
{
    if (vec_index >= possible_contact_pose_representation.size())
    {
        all_contact_pose_combinations.push_back(contact_pose_combination);
        return;
    }

    if(possible_contact_pose_representation[vec_index].size() != 0)
    {
        for (size_t i = 0; i < possible_contact_pose_representation[vec_index].size(); i++)
        {
            contact_pose_combination[vec_index] = possible_contact_pose_representation[vec_index][i];
            getAllContactPoseCombinations(all_contact_pose_combinations, possible_contact_pose_representation, vec_index+1, contact_pose_combination);
        }
    }
    else
    {
        getAllContactPoseCombinations(all_contact_pose_combinations, possible_contact_pose_representation, vec_index+1, contact_pose_combination);
    }
}

void ContactSpacePlanning::collectTrainingData(BranchingManipMode branching_mode, bool sample_feet_only_state,
                                               bool sample_feet_and_one_hand_state, bool sample_feet_and_two_hands_state)
{
    // sample the initial states
    std::vector<std::shared_ptr<ContactState> > initial_states;

    ContactManipulator left_leg = ContactManipulator::L_LEG;
    ContactManipulator right_leg = ContactManipulator::R_LEG;
    ContactManipulator left_arm = ContactManipulator::L_ARM;
    ContactManipulator right_arm = ContactManipulator::R_ARM;

    std::uniform_real_distribution<float> projection_height_unif(-0.1, 0.1);
    std::uniform_real_distribution<float> unit_unif(-1.0, 1.0);
    std::uniform_real_distribution<float> unsigned_unit_unif(0.0, 1.0);

    Translation3D left_relative_shoulder_position, right_relative_shoulder_position;

    left_relative_shoulder_position[0] = 0;
    left_relative_shoulder_position[1] = robot_properties_->shoulder_w_/2.0;
    left_relative_shoulder_position[2] = robot_properties_->shoulder_z_;

    right_relative_shoulder_position[0] = 0;
    right_relative_shoulder_position[1] = -robot_properties_->shoulder_w_/2.0;
    right_relative_shoulder_position[2] = robot_properties_->shoulder_z_;

    // sample 3 feet stances
    std::uniform_int_distribution<unsigned int> foot_transition_size_unif(0, foot_transition_model_.size()-1);

    for(int i = 0; i < 3; i++)
    {
        unsigned int foot_transition_index = foot_transition_size_unif(rng_);

        auto foot_transition = foot_transition_model_[foot_transition_index];

        int invalid_sampling_counter = 0;

        std::array<float,6> l_foot_xyzrpy, r_foot_xyzrpy;
        l_foot_xyzrpy[0] = foot_transition[0]/2.0; l_foot_xyzrpy[1] = foot_transition[1]/2.0; l_foot_xyzrpy[2] = 99.0;
        l_foot_xyzrpy[3] = 0; l_foot_xyzrpy[4] = 0; l_foot_xyzrpy[5] = foot_transition[2]/2.0;

        r_foot_xyzrpy[0] = -foot_transition[0]/2.0; r_foot_xyzrpy[1] = -foot_transition[1]/2.0; r_foot_xyzrpy[2] = 99.0;
        r_foot_xyzrpy[3] = 0; r_foot_xyzrpy[4] = 0; r_foot_xyzrpy[5] = -foot_transition[2]/2.0;

        RPYTF left_foot_pose = RPYTF(l_foot_xyzrpy);
        RPYTF right_foot_pose = RPYTF(r_foot_xyzrpy);

        float left_leg_projection_height = projection_height_unif(rng_);
        float right_leg_projection_height = projection_height_unif(rng_);

        footPoseSampling(left_leg, left_foot_pose, left_leg_projection_height);
        footPoseSampling(right_leg, right_foot_pose, right_leg_projection_height);

        // sample the initial CoM and CoM velocity
        Translation3D initial_com;
        initial_com[0] = 0.1 * unit_unif(rng_);
        initial_com[1] = 0.1 * unit_unif(rng_);
        initial_com[2] = 0.95 + 0.1 * unit_unif(rng_);

        Vector3D initial_com_dot;
        float initial_com_dot_pan_angle = 180.0 * unit_unif(rng_) * DEG2RAD;
        float initial_com_dot_tilt_angle = 45.0 * unit_unif(rng_) * DEG2RAD;
        float initial_com_dot_magnitude = 0.3 * unsigned_unit_unif(rng_);
        initial_com_dot[0] = initial_com_dot_magnitude * std::cos(initial_com_dot_tilt_angle) * std::cos(initial_com_dot_pan_angle);
        initial_com_dot[1] = initial_com_dot_magnitude * std::cos(initial_com_dot_tilt_angle) * std::sin(initial_com_dot_pan_angle);
        initial_com_dot[2] = initial_com_dot_magnitude * std::sin(initial_com_dot_tilt_angle);

        // sample a feet only state
        std::array<bool,ContactManipulator::MANIP_NUM> feet_only_contact_status = {true,true,false,false};
        std::shared_ptr<Stance> feet_only_stance = std::make_shared<Stance>(left_foot_pose, right_foot_pose,
                                                                            RPYTF(-99.0,-99.0,-99.0,-99.0,-99.0,-99.0),
                                                                            RPYTF(-99.0,-99.0,-99.0,-99.0,-99.0,-99.0),
                                                                            feet_only_contact_status);
        std::shared_ptr<ContactState> feet_only_state = std::make_shared<ContactState>(feet_only_stance, initial_com, initial_com_dot, 1);

        if(sample_feet_only_state)
        {
            initial_states.push_back(feet_only_state);
        }

        // sample a feet + one hand state
        RPYTF left_hand_pose, right_hand_pose;
        const float mean_horizontal_yaw = feet_only_state->getFeetMeanHorizontalYaw();
        const RotationMatrix robot_yaw_rotation = RPYToSO3(RPYTF(0, 0, 0, 0, 0, mean_horizontal_yaw));

        Translation3D global_left_shoulder_position = robot_yaw_rotation * left_relative_shoulder_position + feet_only_state->mean_feet_position_;
        std::array<float,2> left_arm_orientation = {mean_horizontal_yaw + 90.0 - 60.0 * unit_unif(rng_), 20.0 * unit_unif(rng_)};
        handPoseSampling(left_arm, global_left_shoulder_position, left_arm_orientation, left_hand_pose);

        while((left_hand_pose.getXYZ() - initial_com).norm() > 0.8 && invalid_sampling_counter < 100)
        {
            handPoseSampling(left_arm, global_left_shoulder_position, left_arm_orientation, left_hand_pose);
            invalid_sampling_counter++;
        }
        if(invalid_sampling_counter >= 100)
        {
            continue;
        }

        invalid_sampling_counter = 0;

        std::array<bool,ContactManipulator::MANIP_NUM> feet_and_one_hand_contact_status = {true,true,true,false};
        std::shared_ptr<Stance> feet_and_one_hand_stance = std::make_shared<Stance>(left_foot_pose, right_foot_pose, left_hand_pose,
                                                                                    RPYTF(-99.0,-99.0,-99.0,-99.0,-99.0,-99.0),
                                                                                    feet_and_one_hand_contact_status);
        std::shared_ptr<ContactState> feet_and_one_hand_state = std::make_shared<ContactState>(feet_and_one_hand_stance, initial_com, initial_com_dot, 1);

        if(sample_feet_and_one_hand_state)
        {
            initial_states.push_back(feet_and_one_hand_state);
        }

        // sample a feet + two hands state
        Translation3D global_right_shoulder_position = robot_yaw_rotation * right_relative_shoulder_position + feet_only_state->mean_feet_position_;
        std::array<float,2> right_arm_orientation = {mean_horizontal_yaw - 90.0 + 60.0 * unit_unif(rng_), 20.0 * unit_unif(rng_)};
        handPoseSampling(right_arm, global_right_shoulder_position, right_arm_orientation, right_hand_pose);

        while((right_hand_pose.getXYZ() - initial_com).norm() > 0.8 && invalid_sampling_counter < 100)
        {
            handPoseSampling(right_arm, global_right_shoulder_position, right_arm_orientation, right_hand_pose);
        }
        if(invalid_sampling_counter >= 100)
        {
            continue;
        }

        std::array<bool,ContactManipulator::MANIP_NUM> feet_and_two_hands_contact_status = {true,true,true,true};
        std::shared_ptr<Stance> feet_and_two_hands_stance = std::make_shared<Stance>(left_foot_pose, right_foot_pose, left_hand_pose, right_hand_pose,
                                                                                     feet_and_two_hands_contact_status);
        std::shared_ptr<ContactState> feet_and_two_hands_state = std::make_shared<ContactState>(feet_and_two_hands_stance, initial_com, initial_com_dot, 1);

        if(sample_feet_and_two_hands_state)
        {
            initial_states.push_back(feet_and_two_hands_state);
        }
    }

    heuristics_type_ = PlanningHeuristicsType::EUCLIDEAN; // supress the error message

    std::vector<ContactManipulator> branching_manips = ALL_MANIPULATORS;
    for(auto & initial_state : initial_states)
    {
        // RAVELOG_WARN("New initial state.\n");
        branchingContacts(initial_state, branching_mode);
    }
}

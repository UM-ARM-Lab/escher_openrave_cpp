#include "Utilities.hpp"
// #include <omp.h>

ContactSpacePlanning::ContactSpacePlanning(std::shared_ptr<RobotProperties> _robot_properties,
                                           std::vector< std::array<float,3> > _foot_transition_model,
                                           std::vector< std::array<float,2> > _hand_transition_model,
                                           std::vector< std::array<float,3> > _disturbance_rejection_foot_transition_model,
                                           std::vector< std::array<float,2> > _disturbance_rejection_hand_transition_model,
                                           std::vector< std::shared_ptr<TrimeshSurface> > _structures,
                                           std::map<int, std::shared_ptr<TrimeshSurface> > _structures_dict,
                                           std::shared_ptr<MapGrid> _map_grid,
                                           std::shared_ptr<GeneralIKInterface> _general_ik_interface,
                                           int _num_stance_in_state,
                                           int _thread_num,
                                           std::shared_ptr< DrawingHandler > _drawing_handler,
                                           int _planning_id,
                                           bool _use_dynamics_planning,
                                           std::vector<std::pair<Vector6D, float> > _disturbance_samples,
                                           PlanningApplication _planning_application,
                                           bool _check_zero_step_capturability,
                                           bool _check_one_step_capturability,
                                           bool _check_contact_transition_feasibility):
robot_properties_(_robot_properties),
foot_transition_model_(_foot_transition_model),
hand_transition_model_(_hand_transition_model),
disturbance_rejection_foot_transition_model_(_disturbance_rejection_foot_transition_model),
disturbance_rejection_hand_transition_model_(_disturbance_rejection_hand_transition_model),
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
        // neural_network_interface_vector_[i] = std::make_shared<NeuralNetworkInterface>("../data/dynopt_result/objective_regression_nn_models/",
        //                                                                                "../data/dynopt_result/feasibility_classification_nn_models/",
        //                                                                                "../data/dynopt_result/zero_step_capture_feasibility_classification_nn_models_2/",
        //                                                                                "../data/dynopt_result/one_step_capture_feasibility_classification_nn_models_2/");
        // neural_network_interface_vector_[i] = std::make_shared<NeuralNetworkInterface>("../data/dynopt_result/contact_transition_objective_regression_nn_models/",
        //                                                                                "../data/dynopt_result/feasibility_classification_nn_models/",
        //                                                                                "../data/dynopt_result/zero_step_capture_feasibility_classification_nn_models_no_angular_momentum/",
        //                                                                                "../data/dynopt_result/one_step_capture_feasibility_classification_nn_models_no_angular_momentum/");
        neural_network_interface_vector_[i] = std::make_shared<NeuralNetworkInterface>("../data/dynopt_result/contact_transition_objective_regression_nn_models/",
                                                                                       "../data/dynopt_result/feasibility_classification_nn_models/",
                                                                                       "../data/dynopt_result/zero_step_capture_feasibility_classification_nn_models_new_round/",
                                                                                       "../data/dynopt_result/one_step_capture_feasibility_classification_nn_models_new_round/");
    }

    RAVELOG_INFO("Initialize dynamics optimizer interface.\n");
    dynamics_optimizer_interface_vector_.resize(2 * (foot_transition_model_.size() + hand_transition_model_.size()));

    for(unsigned int i = 0; i < dynamics_optimizer_interface_vector_.size(); i++)
    {
        if(robot_properties_->name_ == "athena")
        {
            dynamics_optimizer_interface_vector_[i] = std::make_shared<OptimizationInterface>(STEP_TRANSITION_TIME, SUPPORT_PHASE_TIME,
                                                                                          "../data/SL_optim_config_template/cfg_kdopt_demo.yaml",
                                                                                          _robot_properties->ee_offset_transform_to_dynopt_);
        }
        else if(robot_properties_->name_ == "hermes_full")
        {
            dynamics_optimizer_interface_vector_[i] = std::make_shared<OptimizationInterface>(STEP_TRANSITION_TIME, SUPPORT_PHASE_TIME,
                                                                                          "../data/SL_optim_config_template/cfg_kdopt_demo_contact_transition_hermes_full.yaml",
                                                                                          _robot_properties->ee_offset_transform_to_dynopt_);

            // dynamics_optimizer_interface_vector_[i] = std::make_shared<OptimizationInterface>(STEP_TRANSITION_TIME, SUPPORT_PHASE_TIME,
            //                                                                               "../data/SL_optim_config_template/cfg_kdopt_demo_invdynkin_template_hermes_full.yaml",
            //                                                                               _robot_properties->ee_offset_transform_to_dynopt_);
        }

    }

    one_step_capture_dynamics_optimizer_interface_vector_.resize(2 * (foot_transition_model_.size() + hand_transition_model_.size()));

    for(unsigned int i = 0; i < one_step_capture_dynamics_optimizer_interface_vector_.size(); i++)
    {
        one_step_capture_dynamics_optimizer_interface_vector_[i] = std::make_shared<OptimizationInterface>(0.5, 2.0, "../data/SL_optim_config_template/cfg_kdopt_demo_capture_motion_" + robot_properties_->name_ + ".yaml",
                                                                                                           _robot_properties->ee_offset_transform_to_dynopt_,
                                                                                                           DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT);

        // one_step_capture_dynamics_optimizer_interface_vector_[i] = std::make_shared<OptimizationInterface>(0.5, 2.0, "../data/SL_optim_config_template/cfg_kdopt_demo_one_step_capturability.yaml",
        //                                                                                                    DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT);
    }

    // one_step_capture_dynamics_optimizer_interface_vector_[1] = std::make_shared<OptimizationInterface>(0.5, 2.0, "../data/SL_optim_config_template/cfg_kdopt_demo_one_step_capturability_relaxed_0_1.yaml",
    //                                                                                                    DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT);

    // one_step_capture_dynamics_optimizer_interface_vector_[2] = std::make_shared<OptimizationInterface>(0.5, 2.0, "../data/SL_optim_config_template/cfg_kdopt_demo_one_step_capturability_relaxed_1_0.yaml",
    //                                                                                                    DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT);

    // one_step_capture_dynamics_optimizer_interface_vector_[3] = std::make_shared<OptimizationInterface>(0.5, 2.0, "../data/SL_optim_config_template/cfg_kdopt_demo_one_step_capturability_unlimited.yaml",
    //                                                                                                    DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT);

    // one_step_capture_dynamics_optimizer_interface_vector_[4] = std::make_shared<OptimizationInterface>(0.5, 2.0, "../data/SL_optim_config_template/cfg_kdopt_demo_one_step_capturability_original.yaml",
    //                                                                                                    DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT);

    zero_step_capture_dynamics_optimizer_interface_vector_.resize(2 * (foot_transition_model_.size() + hand_transition_model_.size()));

    for(unsigned int i = 0; i < zero_step_capture_dynamics_optimizer_interface_vector_.size(); i++)
    {
        zero_step_capture_dynamics_optimizer_interface_vector_[i] = std::make_shared<OptimizationInterface>(0.0, 2.0, "../data/SL_optim_config_template/cfg_kdopt_demo_one_step_capturability.yaml",
                                                                                                            _robot_properties->ee_offset_transform_to_dynopt_,
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

    // set up the data collecting steps
    if(planning_application_ == PlanningApplication::COLLECT_DATA)
    {
        // construct the training sample config file storing path
        // training_sample_config_folder_ = "/home/yuchi/amd_workspace_video/workspace/src/catkin/humanoids/humanoid_control/motion_planning/momentumopt_sl/momentumopt_" + robot_properties_->name_ + "/config/capture_test_high_disturbance/";
        // training_sample_config_folder_ = "/home/yuchi/amd_workspace_video/workspace/src/catkin/humanoids/humanoid_control/motion_planning/momentumopt_sl/momentumopt_" + robot_properties_->name_ + "/config/capture_test_testing_data/";
        training_sample_config_folder_ = "/home/yuchi/amd_workspace_video/workspace/src/catkin/humanoids/humanoid_control/motion_planning/momentumopt_sl/momentumopt_" + robot_properties_->name_ + "/config/capture_test_new_round/";
        std::string file_path;

        for (int motion_code_int = ZeroStepCaptureCode::ONE_FOOT; motion_code_int <= ZeroStepCaptureCode::FEET_AND_ONE_HAND; motion_code_int++)
        {
            ZeroStepCaptureCode motion_code = static_cast<ZeroStepCaptureCode>(motion_code_int);

            int file_index = 0;
            while(true)
            {
                file_path = training_sample_config_folder_ + "zero_step_capture_" + std::to_string(motion_code_int) + "/zero_step_capture_feature_" + std::to_string(file_index) + ".txt";
                if(!file_exist(file_path))
                {
                    break;
                }
                file_index++;
            }

            zero_step_capture_file_index_.insert(std::make_pair(motion_code, file_index));
        }

        for (int motion_code_int = OneStepCaptureCode::ONE_FOOT_ADD_FOOT; motion_code_int <= OneStepCaptureCode::TWO_FEET_AND_ONE_HAND_ADD_HAND; motion_code_int++)
        {
            OneStepCaptureCode motion_code = static_cast<OneStepCaptureCode>(motion_code_int);

            int file_index = 0;
            while(true)
            {
                file_path = training_sample_config_folder_ + "one_step_capture_" + std::to_string(motion_code_int) + "/one_step_capture_feature_" + std::to_string(file_index) + ".txt";
                if(!file_exist(file_path))
                {
                    break;
                }
                file_index++;
            }

            one_step_capture_file_index_.insert(std::make_pair(motion_code, file_index));
        }

        // for (int motion_code_int = ContactTransitionCode::FEET_ONLY_MOVE_FOOT; motion_code_int != ContactTransitionCode::FEET_AND_TWO_HANDS_MOVE_HAND; motion_code_int++)
        // {
        //     ContactTransitionCode motion_code = static_cast<ContactTransitionCode>(motion_code_int);

        //     int file_index = 0;
        //     while(true)
        //     {
        //         file_path = training_sample_config_folder_ + "contact_transition_" + std::to_string(motion_code_int) + "/contact_transition_feature_" + std::to_string(file_index) + ".txt";
        //         if(!file_exist(file_path))
        //         {
        //             break;
        //         }
        //         file_index++;
        //     }

        //     contact_transition_file_index_.insert(std::make_pair(motion_code, file_index));
        // }
    }
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

    // int depth_limit = 15;
    int depth_limit = 100000;

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

                    if(current_state->depth_ > depth_limit)
                    {
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
                        float total_disturbance_cost = 0;
                        for(unsigned int i = 0; i < solution_contact_path.size(); i++)
                        {
                            if(!solution_contact_path[i]->is_root_)
                            {
                                solution_contact_path[i]->parent_ = solution_contact_path[i-1];
                                std::cout << solution_contact_path[i]->prev_disturbance_cost_ << "(" << 100*exp(-solution_contact_path[i]->prev_disturbance_cost_) << "%), ";
                                total_disturbance_cost += solution_contact_path[i]->prev_disturbance_cost_;

                                // std::cout << "prev move manip: " << solution_contact_path[i]->prev_move_manip_;
                                // std::cout << ", disturbance cost: " << solution_contact_path[i]->prev_disturbance_cost_ << std::endl;
                            }
                        }
                        std::cout << std::endl;
                        all_solution_contact_paths.push_back(solution_contact_path);

                        current_time = std::chrono::high_resolution_clock::now();

                        float planning_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - time_before_ANA_start_planning).count() /1000.0;

                        all_solution_planning_times.push_back(planning_time);

                        RAVELOG_INFO("Solution Found: T = %5.3f, G = %5.3f, E = %5.3f, DynCost: %5.3f, DisturbCost: %5.3f, Success Prob: %5.2f\%, # of Steps: %d. \n", planning_time, G_, E_, current_state->accumulated_dynamics_cost_, total_disturbance_cost, 100*exp(-total_disturbance_cost), step_count);

                        drawing_handler_->ClearHandler();
                        drawing_handler_->DrawContactPath(solution_contact_path[solution_contact_path.size()-1]);

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

        // std::cout << "start from middle planning test" << std::endl;
        // std::shared_ptr<ContactState> restart_state = all_solution_contact_paths[all_solution_contact_paths.size()-1][3];
        // std::shared_ptr<ContactState> new_initial_state = std::make_shared<ContactState>(restart_state->stances_vector_[0], restart_state->com_, restart_state->com_dot_, restart_state->lmom_, restart_state->amom_, 1);
        // std::cout << "left foot: "; restart_state->stances_vector_[0]->left_foot_pose_.printPose();
        // std::cout << "right foot: "; restart_state->stances_vector_[0]->right_foot_pose_.printPose();
        // std::cout << "left hand: "; restart_state->stances_vector_[0]->left_hand_pose_.printPose();
        // std::cout << "right hand: "; restart_state->stances_vector_[0]->left_hand_pose_.printPose();
        // getchar();
        // drawing_handler_->ClearHandler();
        // ANAStarPlanning(new_initial_state, goal, goal_radius, heuristics_type,
        //                 branching_method, time_limit, epsilon, output_first_solution, goal_as_exact_poses,
        //                 use_learned_dynamics_model, enforce_stop_in_the_end);

        // return contact_state_path;

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

            std::cout << "State CoMs: " << std::endl;
            for(int state_id = 1; state_id < all_solution_contact_paths[i].size(); state_id++)
            {
                std::vector<int> capture_pose_num_vec(int(ContactManipulator::MANIP_NUM), 0);
                std::vector<int> valid_capture_pose_num_vec(int(ContactManipulator::MANIP_NUM), 0);

                for(int capture_id = 0; capture_id < all_solution_contact_paths[i][state_id]->transition_phase_capture_poses_vector_.size(); capture_id++)
                {
                    auto capture_pose = all_solution_contact_paths[i][state_id]->transition_phase_capture_poses_vector_[capture_id];
                    int capture_pose_prediction = all_solution_contact_paths[i][state_id]->transition_phase_capture_poses_prediction_vector_[capture_id];

                    capture_pose_num_vec[int(capture_pose.contact_manip_)]++;
                    valid_capture_pose_num_vec[int(capture_pose.contact_manip_)] += capture_pose_prediction;
                }

                std::cout << "State " << state_id <<
                          ", prev move manip: " << all_solution_contact_paths[i][state_id]->prev_move_manip_ <<
                          ", CoM: " << all_solution_contact_paths[i][state_id]->com_.transpose();

                std::cout << ", Capture Pose Num: [";
                for(int manip_id = 0; manip_id < int(ContactManipulator::MANIP_NUM); manip_id++)
                {
                    std::cout << capture_pose_num_vec[manip_id];
                    if(manip_id < int(ContactManipulator::MANIP_NUM)-1)
                        std::cout << ",";
                }
                std::cout << "]";

                std::cout << ", Valid Capture Pose Num: [";
                for(int manip_id = 0; manip_id < int(ContactManipulator::MANIP_NUM); manip_id++)
                {
                    std::cout << valid_capture_pose_num_vec[manip_id];
                    if(manip_id < int(ContactManipulator::MANIP_NUM)-1)
                        std::cout << ",";
                }
                std::cout << "]";




                std::cout << std::endl;


                // std::shared_ptr<ContactState> current_state = all_solution_contact_paths[i][state_id];
                // std::shared_ptr<ContactState> prev_state = current_state->parent_;
                // ContactManipulator move_manip = current_state->prev_move_manip_;
                // Vector3D moving_direction = (current_state->mean_feet_position_ - prev_state->mean_feet_position_).normalized();
                // if(move_manip == ContactManipulator::L_ARM || move_manip == ContactManipulator::R_ARM)
                // {
                //     current_state->com_ = prev_state->com_;
                //     // if(!prev_state->manip_in_contact(move_manip) || !current_state->manip_in_contact(move_manip))
                //     // {
                //     //     current_state->com_ = prev_state->com_;
                //     // }
                //     // else
                //     // {
                //     //     current_state->com_ = prev_state->com_ + (current_state->stances_vector_[0]->ee_contact_poses_[int(move_manip)].getXYZ() - prev_state->stances_vector_[0]->ee_contact_poses_[int(move_manip)].getXYZ()) * 0.1;
                //     // }
                // }
                // else if(move_manip == ContactManipulator::L_LEG)
                // {
                //     current_state->com_ = 0.6 * current_state->stances_vector_[0]->left_foot_pose_.getXYZ() +
                //                           0.4 * current_state->stances_vector_[0]->right_foot_pose_.getXYZ() + Vector3D(0,0,0.7);
                // }
                // else if(move_manip == ContactManipulator::R_LEG)
                // {
                //     current_state->com_ = 0.6 * current_state->stances_vector_[0]->right_foot_pose_.getXYZ() +
                //                           0.4 * current_state->stances_vector_[0]->left_foot_pose_.getXYZ() + Vector3D(0,0,0.7);
                // }
            }

            if(use_dynamics_planning_)
            {
                kinematicsVerification_StateOnly(all_solution_contact_paths[i]);
                std::cout << "Predicted CoM trajectory(Red)." << std::endl;
                getchar();
            }

            // for(auto & state : all_solution_contact_paths[i])
            // {
            //     std::cout << "Contact state: " << state->stances_vector_[0]->ee_contact_status_[0] << " "
            //                                    << state->stances_vector_[0]->ee_contact_status_[1] << " "
            //                                    << state->stances_vector_[0]->ee_contact_status_[2] << " "
            //                                    << state->stances_vector_[0]->ee_contact_status_[3] << std::endl;

            //     std::cout << "Left foot: ";
            //     state->stances_vector_[0]->ee_contact_poses_[0].printPose();
            //     std::cout << "Right foot: ";
            //     state->stances_vector_[0]->ee_contact_poses_[1].printPose();

            //     std::cout << "CoM: " << state->com_[0] << " " << state->com_[1] << " " << state->com_[2] << std::endl;
            // }

            // total_dynamics_cost_segment_by_segment = fillDynamicsSequenceSegmentBySegment(all_solution_contact_paths[i]);
            // std::cout << "Piecewise Optimization CoM trajectory(Green)." << std::endl;

            // std::vector< std::shared_ptr<ContactState> > test_contact_sequence;
            // test_contact_sequence.push_back(all_solution_contact_paths[i][0]);
            // test_contact_sequence.push_back(all_solution_contact_paths[i][1]);
            // test_contact_sequence.push_back(all_solution_contact_paths[i][2]);
            // test_contact_sequence.push_back(all_solution_contact_paths[i][3]);
            // test_contact_sequence[1]->parent_ = test_contact_sequence[0];
            // test_contact_sequence[2]->parent_ = test_contact_sequence[1];
            // test_contact_sequence[3]->parent_ = test_contact_sequence[2];

            total_dynamics_cost = fillDynamicsSequence(all_solution_contact_paths[i]);
            // total_dynamics_cost = fillDynamicsSequence(test_contact_sequence);
            std::cout << "Whole Contact Sequence Optimization CoM trajectory(Blue)." << std::endl;
            momentumopt::DynamicsSequence optimized_dynamics_sequence = dynamics_optimizer_interface_vector_[0]->getDynamicsSequence();
            getchar();

            if(total_dynamics_cost != 99999.0)
            // if(false)
            {
                float time_step = 0.1;
                float duration_per_state = dynamics_optimizer_interface_vector_[0]->support_phase_time_ + dynamics_optimizer_interface_vector_[0]->step_transition_time_;
                float start_sample_time = dynamics_optimizer_interface_vector_[0]->support_phase_time_ + time_step; // we do not sample the initial support phase
                float end_sample_time = duration_per_state * (all_solution_contact_paths[i].size()-1); // we do not sample the final support phase
                int sample_id = -1;
                bool enable_file_output = false;

                std::string config_path = "/home/yuchi/amd_workspace_video/workspace/src/catkin/humanoids/humanoid_control/motion_planning/momentumopt_sl/momentumopt_hermes_full/config/";
                std::string experiment_name = "narrw_flat_ground_new_round_baseline";
                std::string kindynopt_config_locomotion_template_path = "../data/SL_optim_config_template/cfg_kdopt_demo_invdynkin_template_hermes_full.yaml";
                // std::string kindynopt_config_capture_template_path = "../data/SL_optim_config_template/cfg_kdopt_demo_invdynkin_template_capture_motion_hermes_full.yaml";
                std::string kindynopt_config_capture_template_path = "../data/SL_optim_config_template/cfg_kdopt_demo_capture_motion_hermes_full.yaml";
                std::string kindynopt_config_test_path = config_path + experiment_name + "/";

                std::ofstream fout_prediction((kindynopt_config_test_path + "network_predictions.txt").c_str());
                std::ofstream fout_feature_vector((kindynopt_config_test_path + "feature_vectors.txt").c_str());
                std::ofstream fout_capture_pose_num((kindynopt_config_test_path + "capture_pose_num.txt").c_str());

                // for(int sample_id = 0; sample_id < 100; sample_id++)
                for(double sampled_impact_time = start_sample_time; sampled_impact_time < end_sample_time; sampled_impact_time = sampled_impact_time + time_step)
                {
                    float phase_time = round(fmod(sampled_impact_time + 0.0001, duration_per_state) * 100) * 0.01;
                    if(phase_time <= dynamics_optimizer_interface_vector_[0]->support_phase_time_ + 0.001)
                    {
                        continue;
                    }

                    sample_id++;

                    // getchar();
                    num_contact_sequence_tried = all_solution_contact_paths.size() - i;

                    std::string kindynopt_config_output_path = kindynopt_config_test_path + experiment_name + "_" + std::to_string(sample_id) + "/";

                    if(enable_file_output)
                    {
                        if(!directory_exist(kindynopt_config_test_path))
                        {
                            mkdir(kindynopt_config_test_path.c_str(), 0777);
                        }

                        if(!directory_exist(kindynopt_config_output_path))
                        {
                            mkdir(kindynopt_config_output_path.c_str(), 0777);
                        }

                        // export the movement & capture contact sequence for kinematics optimization
                        exportContactSequenceOptimizationConfigFiles(dynamics_optimizer_interface_vector_[0],
                                                                    all_solution_contact_paths[i],
                                                                    kindynopt_config_locomotion_template_path,
                                                                    kindynopt_config_output_path + "cfg_kdopt_demo.yaml",
                                                                    kindynopt_config_output_path + "Objects.cf");

                        exportContactSequenceOptimizationConfigFiles(dynamics_optimizer_interface_vector_[0],
                                                                    all_solution_contact_paths[i],
                                                                    kindynopt_config_locomotion_template_path,
                                                                    kindynopt_config_output_path + "initial_motion_cfg_kdopt_demo.yaml",
                                                                    kindynopt_config_output_path + "Objects.cf");
                    }

                    // sample the time to take impact
                    // std::uniform_real_distribution<double> sample_time_distribution(start_sample_time, end_sample_time);
                    // float sampled_impact_time = floor(sample_time_distribution(rng_)/time_step) * time_step;

                    // while(phase_time <= dynamics_optimizer_interface_vector_[0]->support_phase_time_ + 0.001) // sample only the transition phases
                    // {
                    //     sampled_impact_time = floor(sample_time_distribution(rng_)/time_step) * time_step;
                    //     phase_time = round(fmod(sampled_impact_time + 0.0001, duration_per_state) * 100) * 0.01;
                    // }

                    int impact_time_id = int((sampled_impact_time + 0.00001) / time_step) - 1;

                    std::cout << "Sampled impact time: " << sampled_impact_time << " s." << std::endl;
                    std::cout << "Sampled impact time id: " << impact_time_id << std::endl;

                    // sample the disturbance
                    std::vector<float> disturbance_probability_weights;
                    for(auto & disturbance : disturbance_samples_)
                    {
                        disturbance_probability_weights.push_back(disturbance.second);
                    }
                    std::discrete_distribution<int> disturbance_distribution(disturbance_probability_weights.begin(), disturbance_probability_weights.end());
                    Vector6D sampled_disturbance = disturbance_samples_[disturbance_distribution(rng_)].first;

                    std::cout << "Sampled disturbance: (" << sampled_disturbance[0] << ", "
                                                          << sampled_disturbance[1] << ", "
                                                          << sampled_disturbance[2] << ")" << std::endl;

                    // check what is the corresponding state and in support or transition phase
                    std::vector<CapturePose> capture_poses_vector;
                    int impact_state_id = int(sampled_impact_time/duration_per_state);
                    std::shared_ptr<ContactState> impact_state; // the state-to-be if the disturbance does not happen
                    std::shared_ptr<Stance> post_impact_stance;
                    std::array<bool,ContactManipulator::MANIP_NUM> ee_contact_status;
                    std::array<RPYTF, ContactManipulator::MANIP_NUM> ee_contact_poses;

                    // std::cout << "remaining time: " << phase_time << std::endl;
                    // std::cout << "impact state id: " << impact_state_id << std::endl;

                    if(phase_time <= dynamics_optimizer_interface_vector_[0]->support_phase_time_ + 0.001) // support phase
                    {
                        std::cout << "Disturbance on support phase." << std::endl;
                        capture_poses_vector = all_solution_contact_paths[i][impact_state_id]->support_phase_capture_poses_vector_;
                        impact_state = all_solution_contact_paths[i][impact_state_id];
                        post_impact_stance = impact_state->stances_vector_[0];

                        std::cout << "Planned State CoM: (" << impact_state->com_[0] << ", "
                                                            << impact_state->com_[1] << ", "
                                                            << impact_state->com_[2] << ")" << std::endl;

                        std::cout << "Planned State LMOM: (" << impact_state->lmom_[0] << ", "
                                                             << impact_state->lmom_[1] << ", "
                                                             << impact_state->lmom_[2] << ")" << std::endl;
                    }
                    else // contact transition phase
                    {
                        std::cout << "Disturbance on contact transition phase." << std::endl;
                        capture_poses_vector = all_solution_contact_paths[i][impact_state_id+1]->transition_phase_capture_poses_vector_;
                        impact_state = all_solution_contact_paths[i][impact_state_id+1];

                        std::shared_ptr<ContactState> prev_state = impact_state->parent_;
                        ContactManipulator move_manip = impact_state->prev_move_manip_;

                        ee_contact_status = prev_state->stances_vector_[0]->ee_contact_status_;
                        ee_contact_status[move_manip] = false;
                        ee_contact_poses = prev_state->stances_vector_[0]->ee_contact_poses_;
                        ee_contact_poses[move_manip] = RPYTF(-99.0, -99.0, -99.0, -99.0, -99.0, -99.0);
                        post_impact_stance = std::make_shared<Stance>(ee_contact_poses[0], ee_contact_poses[1],
                                                                      ee_contact_poses[2], ee_contact_poses[3],
                                                                      ee_contact_status);

                        std::cout << "Planned State CoM: (" << impact_state->parent_->com_[0] << ", "
                                                            << impact_state->parent_->com_[1] << ", "
                                                            << impact_state->parent_->com_[2] << ")" << std::endl;

                        std::cout << "Planned State LMOM: (" << impact_state->parent_->lmom_[0] << ", "
                                                             << impact_state->parent_->lmom_[1] << ", "
                                                             << impact_state->parent_->lmom_[2] << ")" << std::endl;
                    }

                    Vector3D sampled_disturbance_lmom = sampled_disturbance.block(0,0,3,1);
                    auto sampled_disturbance_lmom_SL = rotateVectorFromOpenraveToSL(sampled_disturbance_lmom);
                    if(enable_file_output)
                    {
                        // store the disturbance configuration
                        int disturbance_counter = 0;
                        for(auto & disturbance : disturbance_samples_)
                        {
                            Vector3D disturbance_lmom = disturbance.first.block(0,0,3,1);
                            auto disturbance_lmom_SL = rotateVectorFromOpenraveToSL(disturbance_lmom);
                            YAML::Node disturbance_cfg;
                            disturbance_cfg["impact_time_id"] = impact_time_id;
                            for(int axis_id = 0; axis_id < 3; axis_id++)
                                disturbance_cfg["disturbance"][axis_id] = disturbance_lmom_SL[axis_id];

                            std::ofstream fout((kindynopt_config_output_path + "disturbance_config_" + std::to_string(disturbance_counter) + ".yaml").c_str());
                            fout << disturbance_cfg;
                            fout.close();
                            disturbance_counter++;
                        }
                    }

                    // get the post impact com and com dot from the optimized com trajectory (Blue)
                    Translation3D post_impact_com = transformPositionFromSLToOpenrave(optimized_dynamics_sequence.dynamicsState(impact_time_id).centerOfMass());
                    Vector3D post_impact_lmom = rotateVectorFromSLToOpenrave(optimized_dynamics_sequence.dynamicsState(impact_time_id).linearMomentum()) + sampled_disturbance_lmom;

                    // Translation3D post_impact_com = impact_state->parent_->com_;
                    // Vector3D post_impact_lmom = impact_state->parent_->lmom_ + sampled_disturbance_lmom;


                    // Vector3D post_impact_amom = rotateVectorFromSLToOpenrave(optimized_dynamics_sequence.dynamicsState(impact_time_id).angularMomentum());
                    Vector3D post_impact_amom = Vector3D(0,0,0);
                    Vector3D post_impact_com_dot = post_impact_lmom / robot_properties_->mass_;

                    std::cout << "Post-impact CoM: (" << post_impact_com[0] << ", "
                                                      << post_impact_com[1] << ", "
                                                      << post_impact_com[2] << ")" << std::endl;

                    std::cout << "Post-impact LMOM: (" << post_impact_lmom[0] << ", "
                                                       << post_impact_lmom[1] << ", "
                                                       << post_impact_lmom[2] << ")" << std::endl;

                    std::shared_ptr<ContactState> post_impact_state = std::make_shared<ContactState>(post_impact_stance, post_impact_com, post_impact_com_dot, post_impact_lmom, post_impact_amom, 1);

                    std::cout << "Sampled Disturbance: " << sampled_disturbance_lmom_SL.transpose() << std::endl;
                    std::cout << "Initial CoM: " << transformPositionFromOpenraveToSL(post_impact_state->com_).transpose() << std::endl;
                    std::cout << "Initial LMOM: " << rotateVectorFromOpenraveToSL(post_impact_state->lmom_).transpose() << std::endl;
                    std::cout << "Initial AMOM: " << rotateVectorFromOpenraveToSL(post_impact_state->amom_).transpose() << std::endl;
                    std::cout << "Left Foot: "; post_impact_state->stances_vector_[0]->left_foot_pose_.printPose();
                    std::cout << "Right Foot: "; post_impact_state->stances_vector_[0]->right_foot_pose_.printPose();


                    // zero step capture
                    std::vector< std::shared_ptr<ContactState> > zero_step_capture_path;
                    zero_step_capture_path.push_back(post_impact_state);

                    if(enable_file_output)
                    {
                        exportContactSequenceOptimizationConfigFiles(zero_step_capture_dynamics_optimizer_interface_vector_[0],
                                                                     zero_step_capture_path,
                                                                     kindynopt_config_capture_template_path,
                                                                     kindynopt_config_output_path + "capture_motion_cfg_kdopt_demo_0.yaml",
                                                                     kindynopt_config_output_path + "Objects.cf");
                    }


                    // one step capture
                    // std::cout << "Number of capture contacts: " << capture_poses_vector.size() << std::endl;
                    int capture_pose_id = 1;
                    int predicted_capture_pose_num = 0;

                    std::cout << "time: " << sampled_impact_time << ": ";
                    for(auto & capture_info : capture_poses_vector)
                    {
                        // construct the contact floating state, the capture state, and the come back state
                        std::shared_ptr<ContactState> capture_state, come_back_state;
                        std::vector< std::shared_ptr<ContactState> > one_step_capture_path, later_path;

                        RPYTF capture_pose = capture_info.capture_pose_;
                        ContactManipulator capture_contact_manip = capture_info.contact_manip_;

                        post_impact_state = std::make_shared<ContactState>(post_impact_stance, post_impact_com, post_impact_com_dot, post_impact_lmom, post_impact_amom, 1);
                        ee_contact_status = post_impact_state->stances_vector_[0]->ee_contact_status_;
                        ee_contact_status[capture_contact_manip] = true;
                        ee_contact_poses = post_impact_state->stances_vector_[0]->ee_contact_poses_;
                        ee_contact_poses[capture_contact_manip] = capture_pose;

                        std::shared_ptr<Stance> capture_stance = std::make_shared<Stance>(ee_contact_poses[0], ee_contact_poses[1],
                                                                                          ee_contact_poses[2], ee_contact_poses[3],
                                                                                          ee_contact_status);

                        capture_state = std::make_shared<ContactState>(capture_stance, post_impact_state, capture_contact_manip, 1, robot_properties_->robot_z_);

                        // std::cout << "Postimpact Left Foot: "; post_impact_state->stances_vector_[0]->left_foot_pose_.printPose();
                        // std::cout << "Postimpact Right Foot: "; post_impact_state->stances_vector_[0]->right_foot_pose_.printPose();
                        // std::cout << "Capture Left Foot: "; capture_state->stances_vector_[0]->left_foot_pose_.printPose();
                        // std::cout << "Capture Right Foot: "; capture_state->stances_vector_[0]->right_foot_pose_.printPose();

                        bool one_step_dynamically_feasible = neural_network_interface_vector_[0]->predictOneStepCaptureDynamics(capture_state, NeuralNetworkModelType::FRUGALLY_DEEP);

                        std::cout << one_step_dynamically_feasible << " ";
                        // fout_prediction << one_step_dynamically_feasible << " ";

                        Eigen::VectorXd feature_vector = neural_network_interface_vector_[0]->getOneStepCaptureFeatureVector(capture_state).transpose();

                        // std::cout << std::endl << feature_vector.transpose() << std::endl;
                        fout_feature_vector << feature_vector.transpose() << std::endl;

                        if(one_step_dynamically_feasible)
                        {
                            predicted_capture_pose_num++;
                        }

                        // ee_contact_status = capture_state->stances_vector_[0]->ee_contact_status_;
                        // ee_contact_status[move_manip] = impact_state->stances_vector_[0]->ee_contact_status_[move_manip];
                        // ee_contact_poses = capture_state->stances_vector_[0]->ee_contact_poses_;
                        // ee_contact_poses[move_manip] = impact_state->stances_vector_[0]->ee_contact_poses_[move_manip];

                        // std::shared_ptr<Stance> come_back_stance = std::make_shared<Stance>(ee_contact_poses[0], ee_contact_poses[1],
                        //                                                                     ee_contact_poses[2], ee_contact_poses[3],
                        //                                                                     ee_contact_status);

                        // come_back_state = std::make_shared<ContactState>(come_back_stance, capture_state, move_manip, 1, robot_properties_->robot_z_);

                        // impact_state->prev_move_manip_ = capture_contact_manip;
                        // impact_state->parent_ = come_back_state;

                        one_step_capture_path.push_back(post_impact_state);
                        one_step_capture_path.push_back(capture_state);

                        // later_path.push_back(capture_state);
                        // later_path.push_back(come_back_state);
                        // later_path.insert(later_path.end(), all_solution_contact_paths[i].begin()+impact_state_id, all_solution_contact_paths[i].end());

                        if(enable_file_output)
                        {
                            exportContactSequenceOptimizationConfigFiles(one_step_capture_dynamics_optimizer_interface_vector_[0],
                                                                         one_step_capture_path,
                                                                         kindynopt_config_capture_template_path,
                                                                         kindynopt_config_output_path + "capture_motion_cfg_kdopt_demo_" + std::to_string(capture_pose_id) + ".yaml",
                                                                         kindynopt_config_output_path + "Objects.cf");
                        }

                        // exportContactSequenceOptimizationConfigFiles(dynamics_optimizer_interface_vector_[0],
                        //                                              later_path,
                        //                                              kindynopt_config_template_path,
                        //                                              kindynopt_config_output_path + "final_motion_cfg_kdopt_demo.yaml",
                        //                                              kindynopt_config_output_path + "Objects.cf");

                        capture_pose_id++;
                    }

                    std::cout << std::endl;
                    fout_prediction << sampled_impact_time << " "
                                    << sampled_disturbance_lmom_SL[0] << " "
                                    << sampled_disturbance_lmom_SL[1] << " "
                                    << sampled_disturbance_lmom_SL[2] << " "
                                    << predicted_capture_pose_num << " "
                                    << capture_poses_vector.size() << " "
                                    << impact_state->prev_move_manip_ << std::endl;
                    // getchar();

                    // std::ofstream planning_contact_list_fstream("contact_list.txt", std::ofstream::out);

                    // for(auto & state : all_solution_contact_paths[i])
                    // {
                    //     if(!state->is_root_)
                    //     {
                    //         int move_manip = int(state->prev_move_manip_);
                    //         planning_contact_list_fstream << move_manip << " "
                    //                                       << state->stances_vector_[0]->ee_contact_poses_[move_manip].x_ << " "
                    //                                       << state->stances_vector_[0]->ee_contact_poses_[move_manip].y_ << " "
                    //                                       << state->stances_vector_[0]->ee_contact_poses_[move_manip].z_ << " "
                    //                                       << state->stances_vector_[0]->ee_contact_poses_[move_manip].roll_ << " "
                    //                                       << state->stances_vector_[0]->ee_contact_poses_[move_manip].pitch_ << " "
                    //                                       << state->stances_vector_[0]->ee_contact_poses_[move_manip].yaw_ << " "
                    //                                       << state->com_[0] << " " << state->com_[1] << " " << state->com_[2] << std::endl;
                    //     }
                    // }
                }

                fout_prediction.close();
                fout_feature_vector.close();

                break;
            }

            if(total_dynamics_cost != 99999.0)
            {
                return all_solution_contact_paths[i];
            }
        }

        std::ofstream planning_dynamics_cost_fstream("contact_planning_test_results.txt", std::ofstream::app);

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
    bool left_foot_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::L_LEG];
    bool right_foot_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::R_LEG];
    bool left_hand_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::L_ARM];
    bool right_hand_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::R_ARM];

    general_ik_interface->resetContactStateRelatedParameters();

    // Contact Manipulator Pose
    if(left_foot_in_contact)
    {
        general_ik_interface->addNewManipPose("l_leg", current_state->stances_vector_[0]->left_foot_pose_.GetRaveTransform());
    }
    if(right_foot_in_contact)
    {
        general_ik_interface->addNewManipPose("r_leg", current_state->stances_vector_[0]->right_foot_pose_.GetRaveTransform());
    }
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
            dynamically_feasible = neural_network_interface_vector_[0]->predictContactTransitionDynamics(current_state, dynamics_cost, NeuralNetworkModelType::FRUGALLY_DEEP);
            current_state->lmom_ = robot_properties_->mass_ * current_state->com_dot_;
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

            dynamics_optimizer_interface_vector_[index]->step_transition_time_ = STEP_TRANSITION_TIME;
            dynamics_optimizer_interface_vector_[index]->support_phase_time_ = SUPPORT_PHASE_TIME;
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
        bool state_feasibility = kinematicsFeasibilityCheck(current_state, index);
        // bool state_feasibility = kinematicsFeasibilityCheck(current_state, index) && dynamicsFeasibilityCheck(current_state, dynamics_cost, index);
        // bool state_feasibility = dynamicsFeasibilityCheck(current_state, dynamics_cost, index);
        current_state->lmom_ = robot_properties_->mass_ * current_state->com_dot_;

        dynamics_cost = 0;
        std::shared_ptr<ContactState> prev_state = current_state->parent_;
        ContactManipulator move_manip = current_state->prev_move_manip_;
        Vector3D moving_direction = (current_state->mean_feet_position_ - prev_state->mean_feet_position_).normalized();
        if(move_manip == ContactManipulator::L_ARM || move_manip == ContactManipulator::R_ARM)
        {
            current_state->com_ = prev_state->com_;
            // if(!prev_state->manip_in_contact(move_manip) || !current_state->manip_in_contact(move_manip))
            // {
            //     current_state->com_ = prev_state->com_;
            // }
            // else
            // {
            //     current_state->com_ = prev_state->com_ + (current_state->stances_vector_[0]->ee_contact_poses_[int(move_manip)].getXYZ() - prev_state->stances_vector_[0]->ee_contact_poses_[int(move_manip)].getXYZ()) * 0.1;
            // }
        }
        else if(move_manip == ContactManipulator::L_LEG)
        {
            // current_state->com_ = 0.6 * current_state->stances_vector_[0]->left_foot_pose_.getXYZ() +
            //                       0.4 * current_state->stances_vector_[0]->right_foot_pose_.getXYZ() + Vector3D(0,0,0.7);
            current_state->com_ = 0.7 * current_state->stances_vector_[0]->left_foot_pose_.getXYZ() +
                                  0.3 * current_state->stances_vector_[0]->right_foot_pose_.getXYZ() + Vector3D(0,0,0.7);
        }
        else if(move_manip == ContactManipulator::R_LEG)
        {
            // current_state->com_ = 0.6 * current_state->stances_vector_[0]->right_foot_pose_.getXYZ() +
            //                       0.4 * current_state->stances_vector_[0]->left_foot_pose_.getXYZ() + Vector3D(0,0,0.7);
            current_state->com_ = 0.7 * current_state->stances_vector_[0]->right_foot_pose_.getXYZ() +
                                  0.3 * current_state->stances_vector_[0]->left_foot_pose_.getXYZ() + Vector3D(0,0,0.7);
        }
        current_state->com_dot_ = (current_state->com_ - current_state->parent_->com_) / (STEP_TRANSITION_TIME+SUPPORT_PHASE_TIME);
        current_state->lmom_ = robot_properties_->mass_ * current_state->com_dot_;

        bool left_hand_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::L_ARM];
        bool right_hand_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::R_ARM];

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

std::vector< std::shared_ptr<ContactState> > ContactSpacePlanning::getBranchingStates(std::shared_ptr<ContactState> current_state, std::vector<ContactManipulator>& branching_manips, std::vector< std::array<float,3> > foot_transition_model, std::vector< std::array<float,2> > hand_transition_model)
{
    std::vector< std::shared_ptr<ContactState> > branching_states;
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

    std::vector<bool> has_branch(ContactManipulator::MANIP_NUM, false);
    // std::vector< std::vector<std::shared_ptr<ContactState> > > branching_states_by_move_manip(ContactManipulator::MANIP_NUM);

    // get all the possible branching states of the current state
    for(auto & move_manip : branching_manips)
    {
        if(move_manip == ContactManipulator::L_LEG || move_manip == ContactManipulator::R_LEG)
        {
            for(auto & step : foot_transition_model)
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
            for(auto & arm_orientation : hand_transition_model)
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

    return branching_states;
}

void ContactSpacePlanning::branchingContacts(std::shared_ptr<ContactState> current_state, BranchingManipMode branching_mode, int specified_motion_code)
{
    std::vector< std::shared_ptr<ContactState> > branching_states, disturbance_rejection_branching_states;
    std::vector<ContactManipulator> branching_manips;
    std::vector<ContactManipulator> dummy_branching_manips = ALL_MANIPULATORS;
    std::vector< std::array<float,2> > hand_transition_model;
    std::vector< std::array<float,3> > foot_transition_model = foot_transition_model_;
    std::vector< std::array<float,2> > disturbance_rejection_hand_transition_model = disturbance_rejection_hand_transition_model_;
    std::vector< std::array<float,3> > disturbance_rejection_foot_transition_model = disturbance_rejection_foot_transition_model_;
    bool printing_capturability_info = false;

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

    if(planning_application_ == PlanningApplication::COLLECT_DATA)
    {
        branching_manips.clear();
        branching_manips.push_back(ContactManipulator::R_LEG);
    }

    branching_states = getBranchingStates(current_state, branching_manips, foot_transition_model, hand_transition_model);
    disturbance_rejection_branching_states = getBranchingStates(current_state, dummy_branching_manips, disturbance_rejection_foot_transition_model, disturbance_rejection_hand_transition_model);


    // Find the forces rejected by each category of the moving manipulator
    std::vector< std::unordered_map<int,int> > disturbance_rejection_contact_num_by_manip(ContactManipulator::MANIP_NUM);
    std::vector<float> disturbance_costs(ContactManipulator::MANIP_NUM, 0.0);
    std::vector< std::vector<CapturePose> > capture_poses_by_manip(int(ContactManipulator::MANIP_NUM)); // the vector of capture poses by moving manipulator
    // std::unordered_map< std::array<bool, ContactManipulator::MANIP_NUM>, std::vector<CapturePose> > capture_poses_by_transition_contact_status; // the vector of capture poses by transition contact status
    std::vector< std::vector<int> > capture_poses_prediction_by_manip(int(ContactManipulator::MANIP_NUM)); // the vector of capture poses prediction by moving manipulator
    std::uniform_real_distribution<double> unsigned_unit_unif(0, 1.0);

    if(!disturbance_samples_.empty() && (check_zero_step_capturability_ || check_one_step_capturability_))
    {
        // get all the possible contacts available for the support phase
        current_state->support_phase_capture_poses_vector_.clear();
        for(auto & branching_state : disturbance_rejection_branching_states)
        {
            ContactManipulator capture_contact_manip = branching_state->prev_move_manip_;
            if(branching_state->manip_in_contact(capture_contact_manip) && // only consider branches making a new contact
               !current_state->manip_in_contact(capture_contact_manip)) // only consider manip that are free to make contacts
            {
                // if(capture_contact_manip == ContactManipulator::L_ARM || capture_contact_manip == ContactManipulator::R_ARM)
                // {
                //     float min_xy_dist_hand_foot = 9999.0;

                //     for(auto & leg_manip : LEG_MANIPULATORS)
                //     {
                //         float xy_dist_hand_foot = (branching_state->stances_vector_[0]->ee_contact_poses_[capture_contact_manip].getXY()-current_state->stances_vector_[0]->ee_contact_poses_[leg_manip].getXY()).norm();
                //         if(xy_dist_hand_foot < min_xy_dist_hand_foot)
                //         {
                //             min_xy_dist_hand_foot = xy_dist_hand_foot;
                //         }
                //     }

                //     if(min_xy_dist_hand_foot > 0.8)
                //     {
                //         continue;
                //     }
                // }
                // else if(capture_contact_manip == ContactManipulator::L_LEG || capture_contact_manip == ContactManipulator::R_LEG)
                // {
                //     continue;
                // }

                current_state->support_phase_capture_poses_vector_.push_back( CapturePose(capture_contact_manip,
                                                                                          branching_state->stances_vector_[0]->ee_contact_poses_[capture_contact_manip]));
            }
        }

        // // new disturbance check
        // // 1. diturbance cost depends on the edge, not node.
        // // 2. disturbance query should be collected together, and query as a matrix
        // for(auto & move_manip : branching_manips)
        // {
        //     // get all possible transiiton contact stance
        //     std::array<bool,ContactManipulator::MANIP_NUM> ee_contact_status = current_state->stances_vector_[0]->ee_contact_status_;
        //     ee_contact_status[move_manip] = false;

        //     if(check_one_step_capturability_)
        //     {
        //         for(auto & branching_state : disturbance_rejection_branching_states)
        //         {
        //             ContactManipulator capture_contact_manip = branching_state->prev_move_manip_;
        //             if(branching_state->manip_in_contact(capture_contact_manip) && // only consider branches making a new contact
        //                !ee_contact_status[capture_contact_manip]) // only consider manip that are free to make contacts
        //             {
        //                 capture_poses_by_transition_contact_status[ee_contact_status].push_back( CapturePose(capture_contact_manip,
        //                                                                                        branching_state->stances_vector_[0]->ee_contact_poses_[capture_contact_manip]));
        //             }
        //         }
        //     }
        // }

        std::map< std::array<bool, ContactManipulator::MANIP_NUM>, std::unordered_map<int,int> > checked_zero_capture_state_disturbance_rejection_contact_num;
        for(auto & move_manip : branching_manips)
        {
            // std::cout << "Branch manip: " << move_manip << std::endl;

            auto time_start_new_manip = std::chrono::high_resolution_clock::now();

            // get the initial state for this move_manip
            std::unordered_map<int,int> disturbance_rejection_contact_num_map;
            std::vector< std::shared_ptr<ContactState> > zero_step_capture_state_vec(disturbance_samples_.size());
            std::unordered_map<int,std::vector< std::shared_ptr<ContactState> > > disturbance_one_step_capture_state_vec_map;
            std::vector<bool> zero_step_dynamically_feasible_vec(disturbance_samples_.size());
            std::unordered_map<int,std::vector<bool> > disturbance_one_step_dynamically_feasible_map;

            // construct the state for the floating moving end-effector
            std::array<bool,ContactManipulator::MANIP_NUM> ee_contact_status = current_state->stances_vector_[0]->ee_contact_status_;
            ee_contact_status[move_manip] = false;

            if(check_one_step_capturability_)
            {
                for(auto & branching_state : disturbance_rejection_branching_states)
                {
                    ContactManipulator capture_contact_manip = branching_state->prev_move_manip_;
                    if(branching_state->manip_in_contact(capture_contact_manip) && // only consider branches making a new contact
                       !ee_contact_status[capture_contact_manip]) // only consider manip that are free to make contacts
                    {
                        capture_poses_by_manip[int(move_manip)].push_back( CapturePose(capture_contact_manip,
                                                                                       branching_state->stances_vector_[0]->ee_contact_poses_[capture_contact_manip]));
                    }
                }
            }

            // the zero capture state has already been checked.
            if(checked_zero_capture_state_disturbance_rejection_contact_num.find(ee_contact_status) != checked_zero_capture_state_disturbance_rejection_contact_num.end())
            {
                disturbance_rejection_contact_num_by_manip[move_manip] = checked_zero_capture_state_disturbance_rejection_contact_num.find(ee_contact_status)->second;
                continue;
            }

            std::array<RPYTF, ContactManipulator::MANIP_NUM> ee_contact_poses = current_state->stances_vector_[0]->ee_contact_poses_;
            ee_contact_poses[move_manip] = RPYTF(-99.0, -99.0, -99.0, -99.0, -99.0, -99.0);

            std::shared_ptr<Stance> zero_step_capture_stance = std::make_shared<Stance>(ee_contact_poses[0], ee_contact_poses[1],
                                                                                        ee_contact_poses[2], ee_contact_poses[3],
                                                                                        ee_contact_status);

            Translation3D initial_com = current_state->com_;

            if(printing_capturability_info)
            {
                std::cout << "==========================" << std::endl;
                std::cout << "Initial CoM: (" << initial_com[0] << ", " << initial_com[1] << ", " << initial_com[2] << ")" << std::endl;
                std::cout << "Initial LMOM: (" << current_state->lmom_[0] << ", " << current_state->lmom_[1] << ", " << current_state->lmom_[2] << ")" << std::endl;
                std::cout << "EE Contact Status: (" << ee_contact_status[0] << ", " << ee_contact_status[1] << ", " << ee_contact_status[2] << ", " << ee_contact_status[3] << ")" << std::endl;
                std::cout << "Left Foot: "; ee_contact_poses[0].printPose();
                std::cout << "Right Foot: "; ee_contact_poses[1].printPose();
                std::cout << "Left Hand: "; ee_contact_poses[2].printPose();
                std::cout << "Right Hand: "; ee_contact_poses[3].printPose();
                std::cout << std::endl;
            }

            // old disturbance check
            for(int disturb_id = 0; disturb_id < disturbance_samples_.size(); disturb_id++)
            {
                auto disturbance = disturbance_samples_[disturb_id];
                Vector3D post_impact_com_dot = current_state->com_dot_ + disturbance.first.head(3) / robot_properties_->mass_;
                Vector3D post_impact_lmom = current_state->lmom_ + disturbance.first.head(3);
                Vector3D post_impact_amom = disturbance.first.tail(3);

                if(printing_capturability_info)
                {
                    std::cout << "Disturbance ID: " << disturb_id << ", (" << disturbance.first[0] << ", " << disturbance.first[1] << ", " << disturbance.first[2] << ")"
                            << ", Post-Impact LMOM: (" << post_impact_lmom[0] << ", " << post_impact_lmom[1] << ", " << post_impact_lmom[2] << ")" << std::endl;
                }

                // zero step capturability
                std::shared_ptr<ContactState> zero_step_capture_contact_state = std::make_shared<ContactState>(zero_step_capture_stance, initial_com, post_impact_com_dot, post_impact_lmom, post_impact_amom, 1);
                std::vector< std::shared_ptr<ContactState> > zero_step_capture_contact_state_sequence = {zero_step_capture_contact_state};

                // std::cout << "Zero Step Capture Check:" << std::endl;

                if(check_zero_step_capturability_)
                {
                    if(planning_application_ == PlanningApplication::PLAN_IN_ENV)
                    {
                        if(use_learned_dynamics_model_)
                        {
                            zero_step_capture_state_vec[disturb_id] = zero_step_capture_contact_state;
                            // zero_step_dynamically_feasible_vec[disturb_id] = neural_network_interface_vector_[0]->predictZeroStepCaptureDynamics(zero_step_capture_contact_state, NeuralNetworkModelType::FRUGALLY_DEEP);
                            // if(printing_capturability_info)
                            //     std::cout << "Zero Step Capture Check: " << zero_step_dynamically_feasible_vec[disturb_id] << std::endl;
                        }
                        else
                        {
                            // drawing_handler_->ClearHandler();
                            // drawing_handler_->DrawContactPath(zero_step_capture_contact_state);
                            // // getchar();

                            zero_step_capture_dynamics_optimizer_interface_vector_[0]->updateContactSequence(zero_step_capture_contact_state_sequence);

                            float zero_step_dummy_dynamics_cost = 0.0;
                            zero_step_dynamically_feasible_vec[disturb_id] = zero_step_capture_dynamics_optimizer_interface_vector_[0]->dynamicsOptimization(zero_step_dummy_dynamics_cost);

                            // zero_step_capture_dynamics_optimizer_interface_vector_[0]->storeDynamicsOptimizationResult(zero_step_capture_contact_state, zero_step_dummy_dynamics_cost, zero_step_dynamically_feasible_vec[disturb_id], planning_id_);
                        }
                    }
                    else if(planning_application_ == PlanningApplication::COLLECT_DATA)
                    {
                        if(unsigned_unit_unif(rng_) < 0.95)
                        {
                            continue;
                        }

                        // get the motion code and path
                        zero_step_capture_dynamics_optimizer_interface_vector_[0]->updateContactSequence(zero_step_capture_contact_state_sequence);
                        std::shared_ptr<ContactState> standard_zero_step_capture_contact_state = zero_step_capture_contact_state->getStandardInputState(DynOptApplication::ZERO_STEP_CAPTURABILITY_DYNOPT);
                        auto motion_code_poses_pair = standard_zero_step_capture_contact_state->getZeroStepCapturabilityCodeAndPoses();
                        ZeroStepCaptureCode motion_code = motion_code_poses_pair.first;

                        if(specified_motion_code == -1 || specified_motion_code == int(motion_code))
                        {
                            if(zero_step_capture_file_index_[motion_code] > 104000)
                            {
                                break;
                            }
                            std::string motion_code_str = std::to_string(int(motion_code));
                            std::string file_number_str = std::to_string(zero_step_capture_file_index_[motion_code]);

                            // store the feature vector file
                            std::string training_sample_path = training_sample_config_folder_ + "zero_step_capture_" + motion_code_str + "/";

                            std::cout << "Export zero step capture data: motion code = " << motion_code_str << ", file number = " << file_number_str << std::endl;

                            // store the config file
                            exportContactSequenceOptimizationConfigFiles(zero_step_capture_dynamics_optimizer_interface_vector_[0],
                                                                         zero_step_capture_contact_state_sequence,
                                                                         "../data/SL_optim_config_template/cfg_kdopt_demo_capture_motion_" + robot_properties_->name_ + ".yaml",
                                                                         training_sample_path + "cfg_kdopt_zero_step_capture_" + motion_code_str + "_" + file_number_str + ".yaml",
                                                                         training_sample_path + "Objects_" + motion_code_str + "_" + file_number_str + ".cf");

                            zero_step_capture_dynamics_optimizer_interface_vector_[0]->storeDynamicsOptimizationFeature(zero_step_capture_contact_state,
                                                                                                                        training_sample_path,
                                                                                                                        zero_step_capture_file_index_[motion_code]);

                            zero_step_capture_file_index_[motion_code]++;
                        }

                        zero_step_dynamically_feasible_vec[disturb_id] = false; // for collecting data
                    }
                }

                // one step capturability
                if(check_one_step_capturability_)
                {
                    int capture_pose_num = capture_poses_by_manip[int(move_manip)].size();
                    std::vector<bool> one_step_dynamically_feasible_vec(capture_pose_num);
                    std::vector< std::shared_ptr<ContactState> > one_step_capture_contact_state_vec(capture_pose_num);

                    std::shared_ptr<ContactState> prev_contact_state = std::make_shared<ContactState>(*zero_step_capture_contact_state);
                    int capture_pose_id = 0;

                    auto time_before_generate_capture_poses = std::chrono::high_resolution_clock::now();

                    for(auto & capture_pose : capture_poses_by_manip[int(move_manip)])
                    {
                        ContactManipulator capture_contact_manip = capture_pose.contact_manip_;

                        // std::shared_ptr<ContactState> prev_contact_state = std::make_shared<ContactState>(*zero_step_capture_contact_state);

                        // construct the state for the floating moving end-effector
                        std::array<bool,ContactManipulator::MANIP_NUM> ee_contact_status = prev_contact_state->stances_vector_[0]->ee_contact_status_;
                        ee_contact_status[capture_contact_manip] = true;
                        std::array<RPYTF, ContactManipulator::MANIP_NUM> ee_contact_poses = prev_contact_state->stances_vector_[0]->ee_contact_poses_;
                        ee_contact_poses[capture_contact_manip] = capture_pose.capture_pose_;

                        std::shared_ptr<Stance> one_step_capture_stance = std::make_shared<Stance>(ee_contact_poses[0], ee_contact_poses[1],
                                                                                                    ee_contact_poses[2], ee_contact_poses[3],
                                                                                                    ee_contact_status);

                        one_step_capture_contact_state_vec[capture_pose_id] = std::make_shared<ContactState>(one_step_capture_stance, prev_contact_state, capture_contact_manip, 1, robot_properties_->robot_z_);

                        // std::shared_ptr<ContactState> one_step_capture_contact_state = std::make_shared<ContactState>(one_step_capture_stance, prev_contact_state, capture_contact_manip, 1, robot_properties_->robot_z_);
                        // std::vector< std::shared_ptr<ContactState> > one_step_capture_contact_state_sequence = {prev_contact_state, one_step_capture_contact_state};

                        capture_pose_id++;
                    }

                    auto time_after_generate_capture_poses = std::chrono::high_resolution_clock::now();

                    if(planning_application_ == PlanningApplication::PLAN_IN_ENV)
                    {
                        if(use_learned_dynamics_model_)
                        {
                            disturbance_one_step_capture_state_vec_map[disturb_id] = one_step_capture_contact_state_vec;

                            // // tf multiple query
                            // one_step_dynamically_feasible_vec = neural_network_interface_vector_[0]->predictOneStepCaptureDynamics(one_step_capture_contact_state_vec, NeuralNetworkModelType::TENSORFLOW);

                            // auto time_after_tensorflow_multiple_query = std::chrono::high_resolution_clock::now();

                            // // tf single query
                            // capture_pose_id = 0;
                            // for(auto & one_step_capture_contact_state : one_step_capture_contact_state_vec)
                            // {
                            //     one_step_dynamically_feasible_vec[capture_pose_id] = neural_network_interface_vector_[0]->predictOneStepCaptureDynamics(one_step_capture_contact_state, NeuralNetworkModelType::TENSORFLOW);
                            //     capture_pose_id++;
                            // }

                            // auto time_after_tensorflow_single_query = std::chrono::high_resolution_clock::now();

                            // // fdeep multiple query (construct query matrix and query it as a whole)
                            // one_step_dynamically_feasible_vec = neural_network_interface_vector_[0]->predictOneStepCaptureDynamics(one_step_capture_contact_state_vec, NeuralNetworkModelType::FRUGALLY_DEEP);

                            // auto time_after_fdeep_multiple_query = std::chrono::high_resolution_clock::now();

                            // // fdeep single query (for loop through all the capture poses)
                            // capture_pose_id = 0;
                            // for(auto & one_step_capture_contact_state : one_step_capture_contact_state_vec)
                            // {
                            //     // bool tmp_result = neural_network_interface_vector_[0]->predictOneStepCaptureDynamics(one_step_capture_contact_state, NeuralNetworkModelType::FRUGALLY_DEEP);
                            //     // if(tmp_result != one_step_dynamically_feasible_vec[capture_pose_id])
                            //     // {
                            //     //     std::cout << "bug!!!!" << std::endl;
                            //     //     getchar();
                            //     // }
                            //     one_step_dynamically_feasible_vec[capture_pose_id] = neural_network_interface_vector_[0]->predictOneStepCaptureDynamics(one_step_capture_contact_state, NeuralNetworkModelType::FRUGALLY_DEEP);
                            //     capture_pose_id++;
                            // }

                            // auto time_after_fdeep_single_query = std::chrono::high_resolution_clock::now();

                            // // std::cout << "capture pose generation time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_after_generate_capture_poses - time_before_generate_capture_poses).count() << std::endl;
                            // std::cout << "tensorflow multiple query time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_after_tensorflow_multiple_query - time_after_generate_capture_poses).count() << std::endl;
                            // // std::cout << "tensorflow single query time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_after_tensorflow_single_query - time_after_tensorflow_multiple_query).count() << std::endl;
                            // // std::cout << "fdeep multiple query time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_after_fdeep_multiple_query - time_after_tensorflow_single_query).count() << std::endl;
                            // // std::cout << "fdeep single query time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_after_fdeep_single_query - time_after_fdeep_multiple_query).count() << std::endl;
                            // std::cout << "Capture Pose Num: " << capture_pose_num << std::endl;
                            // std::cout << "------------------------------------" << std::endl;

                            // if(unsigned_unit_unif(rng_) < 0.1)
                            // {
                            //     getchar();
                            // }

                        }
                        else
                        {
                            capture_pose_id = 0;
                            for(auto & one_step_capture_contact_state : one_step_capture_contact_state_vec)
                            {
                                std::vector< std::shared_ptr<ContactState> > one_step_capture_contact_state_sequence = {prev_contact_state, one_step_capture_contact_state};

                                one_step_capture_dynamics_optimizer_interface_vector_[0]->updateContactSequence(one_step_capture_contact_state_sequence);

                                float one_step_dummy_dynamics_cost = 0.0;
                                one_step_dynamically_feasible_vec[capture_pose_id] = one_step_capture_dynamics_optimizer_interface_vector_[0]->dynamicsOptimization(one_step_dummy_dynamics_cost);

                                one_step_capture_dynamics_optimizer_interface_vector_[0]->storeDynamicsOptimizationResult(one_step_capture_contact_state, one_step_dummy_dynamics_cost, one_step_dynamically_feasible_vec[capture_pose_id], planning_id_);

                                capture_pose_id++;
                            }
                        }
                    }
                    else if(planning_application_ == PlanningApplication::COLLECT_DATA)
                    {
                        capture_pose_id = 0;
                        for(auto & one_step_capture_contact_state : one_step_capture_contact_state_vec)
                        {
                            if(unsigned_unit_unif(rng_) < 0.95)
                            {
                                continue;
                            }

                            std::vector< std::shared_ptr<ContactState> > one_step_capture_contact_state_sequence = {prev_contact_state, one_step_capture_contact_state};

                            // get the motion code and path
                            one_step_capture_dynamics_optimizer_interface_vector_[0]->updateContactSequence(one_step_capture_contact_state_sequence);
                            std::shared_ptr<ContactState> standard_one_step_capture_contact_state = one_step_capture_contact_state->getStandardInputState(DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT);
                            auto motion_code_poses_pair = standard_one_step_capture_contact_state->getOneStepCapturabilityCodeAndPoses();
                            OneStepCaptureCode motion_code = motion_code_poses_pair.first;

                            if(specified_motion_code == -1 || specified_motion_code == int(motion_code))
                            {
                                if(one_step_capture_file_index_[motion_code] > 104000)
                                {
                                    break;
                                }

                                std::string motion_code_str = std::to_string(int(motion_code));
                                std::string file_number_str = std::to_string(one_step_capture_file_index_[motion_code]);

                                // store the feature vector file
                                std::string training_sample_path = training_sample_config_folder_ + "one_step_capture_" + motion_code_str + "/";

                                std::cout << "Export one step capture data: motion code = " << motion_code_str << ", file number = " << file_number_str << std::endl;

                                // store the config file
                                exportContactSequenceOptimizationConfigFiles(one_step_capture_dynamics_optimizer_interface_vector_[0],
                                                                            one_step_capture_contact_state_sequence,
                                                                            "../data/SL_optim_config_template/cfg_kdopt_demo_capture_motion_" + robot_properties_->name_ + ".yaml",
                                                                            training_sample_path + "cfg_kdopt_one_step_capture_" + motion_code_str + "_" + file_number_str + ".yaml",
                                                                            training_sample_path + "Objects_" + motion_code_str + "_" + file_number_str + ".cf");

                                one_step_capture_dynamics_optimizer_interface_vector_[0]->storeDynamicsOptimizationFeature(one_step_capture_contact_state,
                                                                                                                            training_sample_path,
                                                                                                                            one_step_capture_file_index_[motion_code]);

                                one_step_capture_file_index_[motion_code]++;
                            }

                            one_step_dynamically_feasible_vec[capture_pose_id] = false; // for collecting data

                            capture_pose_id++;
                        }
                    }

                    disturbance_one_step_dynamically_feasible_map[disturb_id] = one_step_dynamically_feasible_vec;
                }

                // std::cout << "====================" << std::endl;
                // getchar();
            }

            // getchar();

            auto time_before_query_network = std::chrono::high_resolution_clock::now();

            // query the neural networks to get the result
            if(planning_application_ == PlanningApplication::PLAN_IN_ENV)
            {
                if(use_learned_dynamics_model_)
                {
                    if(check_zero_step_capturability_)
                    {
                        zero_step_dynamically_feasible_vec = neural_network_interface_vector_[0]->predictZeroStepCaptureDynamics(zero_step_capture_state_vec, NeuralNetworkModelType::TENSORFLOW);
                    }

                    if(check_one_step_capturability_)
                    {
                        // combine all the one_step_capture_state into one vector
                        std::vector< std::shared_ptr<ContactState> > one_step_capture_state_vec;
                        for(int disturb_id = 0; disturb_id < disturbance_samples_.size(); disturb_id++)
                        {
                            one_step_capture_state_vec.insert(one_step_capture_state_vec.end(), disturbance_one_step_capture_state_vec_map[disturb_id].begin(), disturbance_one_step_capture_state_vec_map[disturb_id].end());
                        }

                        std::vector<bool> query_one_step_capture_state = neural_network_interface_vector_[0]->predictOneStepCaptureDynamics(one_step_capture_state_vec, NeuralNetworkModelType::TENSORFLOW);

                        int disturb_start_one_step_capture_index = 0;
                        for(int disturb_id = 0; disturb_id < disturbance_samples_.size(); disturb_id++)
                        {
                            int disturb_one_step_capture_num = disturbance_one_step_capture_state_vec_map[disturb_id].size();
                            disturbance_one_step_dynamically_feasible_map[disturb_id] = std::vector<bool>(query_one_step_capture_state.begin()+disturb_one_step_capture_num,
                                                                                                          query_one_step_capture_state.begin()+disturb_one_step_capture_num+disturb_one_step_capture_num);
                            disturb_start_one_step_capture_index += disturb_one_step_capture_num;
                        }
                    }
                }
            }

            auto time_after_query_network = std::chrono::high_resolution_clock::now();

            // collect the information
            for(int disturb_id = 0; disturb_id < disturbance_samples_.size(); disturb_id++)
            {
                int disturbance_rejection_contact_num = 0;
                if(check_zero_step_capturability_)
                {
                    if(zero_step_dynamically_feasible_vec[disturb_id])
                    {
                        disturbance_rejection_contact_num++;
                    }
                }

                if(check_one_step_capturability_)
                {
                    for(auto one_step_dynamically_feasible : disturbance_one_step_dynamically_feasible_map[disturb_id])
                    {
                        if(one_step_dynamically_feasible)
                        {
                            disturbance_rejection_contact_num++;
                            capture_poses_prediction_by_manip[int(move_manip)].push_back(1);
                        }
                        else
                        {
                            capture_poses_prediction_by_manip[int(move_manip)].push_back(0);
                        }
                    }
                }
                disturbance_rejection_contact_num_map.insert(std::make_pair(disturb_id, disturbance_rejection_contact_num));
            }

            checked_zero_capture_state_disturbance_rejection_contact_num.insert(std::make_pair(ee_contact_status, disturbance_rejection_contact_num_map));
            disturbance_rejection_contact_num_by_manip[move_manip] = disturbance_rejection_contact_num_map;

            auto time_finish_a_manip = std::chrono::high_resolution_clock::now();

            // std::cout << "pre computation time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_before_query_network - time_start_new_manip).count() << std::endl;
            // std::cout << "network query time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_after_query_network - time_before_query_network).count() << std::endl;
            // std::cout << "post computation time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_finish_a_manip - time_after_query_network).count() << std::endl;
            // getchar();
        }

        // std::cout << "Disturbance samples: " << disturbance_samples_.size() << std::endl;

        for(auto & manip_id : branching_manips)
        {
            // std::cout << "Manip ID: " << manip_id << std::endl;

            // for(int disturb_id = 0; disturb_id < disturbance_samples_.size(); disturb_id++)
            // {
            //     int disturbance_rejection_contact_num = disturbance_rejection_contact_num_by_manip[int(manip_id)][disturb_id];
            //     // disturbance_costs[int(manip_id)] += disturbance_samples_[disturb_id].second * exp(-disturbance_rejection_contact_num*0.1);
            //     disturbance_costs[int(manip_id)] += disturbance_samples_[disturb_id].second * exp(-disturbance_rejection_contact_num*0.1);
            // }

            for(int disturb_id = 0; disturb_id < disturbance_samples_.size(); disturb_id++)
            {
                int disturbance_rejection_contact_num = disturbance_rejection_contact_num_by_manip[int(manip_id)][disturb_id];
                disturbance_costs[int(manip_id)] += (disturbance_samples_[disturb_id].second * exp(-disturbance_rejection_contact_num*0.1));

                // std::cout << "Branching Manip: " << manip_id << ", Disturbance ID: " << disturb_id << ", Capture Pose Num: " << disturbance_rejection_contact_num << "/" << capture_poses_by_manip[int(manip_id)].size()+1 << std::endl;
            }

            // std::cout << disturbance_costs[int(manip_id)] << std::endl;

            float disturbance_probability = 0.1;
            float disturbance_rejection_success_rate = (1-disturbance_probability) + disturbance_probability * (1-disturbance_costs[int(manip_id)]);
            disturbance_costs[int(manip_id)] = std::min(-log(disturbance_rejection_success_rate), float(100.0));

            // std::cout << disturbance_costs[int(manip_id)] << std::endl;
        }

        // getchar();
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
            branching_state->transition_phase_capture_poses_vector_ = capture_poses_by_manip[int(branching_state->prev_move_manip_)];
            branching_state->transition_phase_capture_poses_prediction_vector_ = capture_poses_prediction_by_manip[int(branching_state->prev_move_manip_)];
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
            branching_state->transition_phase_capture_poses_vector_ = capture_poses_by_manip[int(branching_state->prev_move_manip_)];
            branching_state->transition_phase_capture_poses_prediction_vector_ = capture_poses_prediction_by_manip[int(branching_state->prev_move_manip_)];
            state_feasibility_check_result[i] = std::make_tuple(true, branching_state, dynamics_cost, disturbance_cost);
        }
    }

    for(auto & check_result : state_feasibility_check_result)
    {
        bool pass_state_feasibility_check = std::get<0>(check_result);
        std::shared_ptr<ContactState> branching_state = std::get<1>(check_result);
        float dynamics_cost = std::get<2>(check_result);
        float disturbance_cost = std::get<3>(check_result);

        if(pass_state_feasibility_check)
        // if(pass_state_feasibility_check && (branching_state->prev_move_manip_ == ContactManipulator::L_LEG || branching_state->prev_move_manip_ == ContactManipulator::R_LEG))
        // if(pass_state_feasibility_check && (current_state->is_root_ || current_state->parent_->is_root_ || (branching_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::L_ARM] && branching_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::R_ARM])))
        // if(pass_state_feasibility_check && (branching_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::L_ARM] || branching_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::R_ARM]))
        {
            // std::cout << "disturbance cost: " << disturbance_cost << std::endl;
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
    std::uniform_real_distribution<double> arm_length_unif(0.3, 0.9);

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

    current_state->prev_disturbance_cost_ = disturbance_cost;

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
            existing_state->depth_ = current_state->depth_;

            existing_state->prev_disturbance_cost_ = disturbance_cost;

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
        // float euclidean_distance_to_goal = fabs(current_state->max_manip_x_ - goal_[0]);
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
    // // construct the initial state for the optimization
    // setupStateReachabilityIK(contact_sequence[0], general_ik_interface_);
    // general_ik_interface_->returnClosest() = true;
    // general_ik_interface_->executeMotion() = true;
    // std::pair<bool,std::vector<OpenRAVE::dReal> > ik_result = general_ik_interface_->solve();

    // std::vector<OpenRAVE::dReal> init_robot_config;
    // general_ik_interface_->robot_->GetDOFValues(init_robot_config);

    std::map<ContactManipulator, RPYTF> floating_initial_contact_poses;

    for(auto manip : ALL_MANIPULATORS)
    {
        if(!contact_sequence[0]->manip_in_contact(manip))
        {
            RPYTF floating_manip_pose;

            // // set the initial contact pose to be close to the one-step capture contacts
            // if(contact_sequence.size() > 1 && manip == contact_sequence[1]->prev_move_manip_ && contact_sequence[1]->manip_in_contact(manip))
            // {
            //     Translation3D contact_position = contact_sequence[1]->stances_vector_[0]->ee_contact_poses_[manip].getXYZ();
            //     Vector3D contact_com_unit_vector = contact_sequence[0]->com_ - contact_position;
            //     contact_com_unit_vector.normalize();
            //     float contact_distance = 0.05;

            //     floating_manip_pose = contact_sequence[1]->stances_vector_[0]->ee_contact_poses_[manip];
            //     floating_manip_pose.x_ += contact_distance * contact_com_unit_vector[0];
            //     floating_manip_pose.y_ += contact_distance * contact_com_unit_vector[1];
            //     floating_manip_pose.z_ += contact_distance * contact_com_unit_vector[2];
            // }
            // else
            // {
            //     // std::string manip_name = robot_properties_->manipulator_name_map_[manip];
            //     // OpenRAVE::Transform floating_manip_transform = general_ik_interface_->robot_->GetManipulator(manip_name)->GetTransform();
            //     // floating_manip_pose = SE3ToXYZRPY(constructTransformationMatrix(floating_manip_transform));
            //     floating_manip_pose = RPYTF(0, 0, 0, 0, 0, 0);
            // }

            // std::cout << "manip: " << manip << ", " << floating_manip_pose.x_ << ", " << floating_manip_pose.y_ << ", " << floating_manip_pose.z_ << std::endl;

            floating_manip_pose = RPYTF(0, 0, 0, 0, 0, 0);

            floating_initial_contact_poses[manip] = floating_manip_pose;
        }
    }

    // generate the files
    optimizer_interface->updateContactSequence(contact_sequence);
    optimizer_interface->exportConfigFiles(optimization_config_template_path, optimization_config_output_path,
                                           objects_config_output_path, floating_initial_contact_poses,
                                           robot_properties_);

}

void ContactSpacePlanning::collectTrainingData(BranchingManipMode branching_mode, bool sample_feet_only_state,
                                               bool sample_feet_and_one_hand_state, bool sample_feet_and_two_hands_state,
                                               int specified_motion_code)
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

    float max_com_hand_dist, max_com_foot_dist;

    if(robot_properties_->name_ == "athena")
    {
        max_com_foot_dist = 1.1;
        max_com_hand_dist = 0.8;
    }
    else if(robot_properties_->name_ == "hermes_full")
    {
        max_com_foot_dist = 0.8;
        max_com_hand_dist = 0.6;
    }

    for(int i = 0; i < 3; i++)
    {
        unsigned int foot_transition_index = foot_transition_size_unif(rng_);

        auto foot_transition = foot_transition_model_[foot_transition_index];

        int invalid_sampling_counter = 0;

        // sample a feet only state

        // set up foot pose projection
        std::array<float,6> l_foot_xyzrpy, r_foot_xyzrpy;
        l_foot_xyzrpy[0] = foot_transition[0]/2.0; l_foot_xyzrpy[1] = foot_transition[1]/2.0; l_foot_xyzrpy[2] = 99.0;
        l_foot_xyzrpy[3] = 0; l_foot_xyzrpy[4] = 0; l_foot_xyzrpy[5] = foot_transition[2]/2.0;

        r_foot_xyzrpy[0] = -foot_transition[0]/2.0; r_foot_xyzrpy[1] = -foot_transition[1]/2.0; r_foot_xyzrpy[2] = 99.0;
        r_foot_xyzrpy[3] = 0; r_foot_xyzrpy[4] = 0; r_foot_xyzrpy[5] = -foot_transition[2]/2.0;

        RPYTF left_foot_pose = RPYTF(l_foot_xyzrpy);
        RPYTF right_foot_pose = RPYTF(r_foot_xyzrpy);

        float left_leg_projection_height = projection_height_unif(rng_);
        footPoseSampling(left_leg, left_foot_pose, left_leg_projection_height);

        float right_leg_projection_height = projection_height_unif(rng_);
        footPoseSampling(right_leg, right_foot_pose, right_leg_projection_height);

        // sample the initial CoM and CoM velocity
        Translation3D initial_com;
        initial_com[0] = std::max(float(0.1), fabs(right_foot_pose.x_-left_foot_pose.x_)) * unit_unif(rng_);
        initial_com[1] = std::max(float(0.1), fabs(right_foot_pose.y_-left_foot_pose.y_)) * unit_unif(rng_);
        initial_com[2] = robot_properties_->robot_z_ + 0.1 * unit_unif(rng_);

        Vector3D initial_com_dot;
        float initial_com_dot_pan_angle = 180.0 * unit_unif(rng_) * DEG2RAD;
        float initial_com_dot_tilt_angle = 45.0 * unit_unif(rng_) * DEG2RAD;
        // float initial_com_dot_magnitude = 0.3 * unsigned_unit_unif(rng_);    // small disturbance
        // float initial_com_dot_magnitude = 0.3 + 0.7 * unsigned_unit_unif(rng_); // large disturbance
        float initial_com_dot_magnitude = 1.0 * unsigned_unit_unif(rng_); // all disturbance
        initial_com_dot[0] = initial_com_dot_magnitude * std::cos(initial_com_dot_tilt_angle) * std::cos(initial_com_dot_pan_angle);
        initial_com_dot[1] = initial_com_dot_magnitude * std::cos(initial_com_dot_tilt_angle) * std::sin(initial_com_dot_pan_angle);
        initial_com_dot[2] = initial_com_dot_magnitude * std::sin(initial_com_dot_tilt_angle);

        Vector3D initial_lmom = initial_com_dot * robot_properties_->mass_;
        Vector3D initial_amom = Vector3D::Zero();

        Vector3D initial_post_impact_com = initial_com + initial_com_dot * 0.1;
        double xy_dist_to_foot = (initial_post_impact_com.head(2)-left_foot_pose.getXY()).norm();

        // while(((left_foot_pose.getXYZ() - initial_com).norm() > max_com_foot_dist || (right_foot_pose.getXYZ() - initial_com).norm() > max_com_foot_dist || xy_dist_to_foot > 0.3) && invalid_sampling_counter < 100)
        while(((left_foot_pose.getXYZ() - initial_com).norm() > max_com_foot_dist || (right_foot_pose.getXYZ() - initial_com).norm() > max_com_foot_dist) && invalid_sampling_counter < 100)
        {
            initial_com[0] = std::max(float(0.1), fabs(right_foot_pose.x_-left_foot_pose.x_)) * unit_unif(rng_);
            initial_com[1] = std::max(float(0.1), fabs(right_foot_pose.y_-left_foot_pose.y_)) * unit_unif(rng_);
            initial_com[2] = robot_properties_->robot_z_ + 0.1 * unit_unif(rng_);

            initial_com_dot_pan_angle = 180.0 * unit_unif(rng_) * DEG2RAD;
            initial_com_dot_tilt_angle = 45.0 * unit_unif(rng_) * DEG2RAD;
            initial_com_dot_magnitude = 1.0 * unsigned_unit_unif(rng_);
            initial_com_dot[0] = initial_com_dot_magnitude * std::cos(initial_com_dot_tilt_angle) * std::cos(initial_com_dot_pan_angle);
            initial_com_dot[1] = initial_com_dot_magnitude * std::cos(initial_com_dot_tilt_angle) * std::sin(initial_com_dot_pan_angle);
            initial_com_dot[2] = initial_com_dot_magnitude * std::sin(initial_com_dot_tilt_angle);

            initial_post_impact_com = initial_com + initial_com_dot * 0.1;
            xy_dist_to_foot = (initial_post_impact_com.head(2)-left_foot_pose.getXY()).norm();

            invalid_sampling_counter++;
        }
        if(invalid_sampling_counter >= 100)
        {
            continue;
        }

        std::array<bool,ContactManipulator::MANIP_NUM> feet_only_contact_status = {true,true,false,false};
        std::shared_ptr<Stance> feet_only_stance = std::make_shared<Stance>(left_foot_pose, right_foot_pose,
                                                                            RPYTF(-99.0,-99.0,-99.0,-99.0,-99.0,-99.0),
                                                                            RPYTF(-99.0,-99.0,-99.0,-99.0,-99.0,-99.0),
                                                                            feet_only_contact_status);
        std::shared_ptr<ContactState> feet_only_state = std::make_shared<ContactState>(feet_only_stance, initial_com, initial_com_dot, initial_lmom, initial_amom, 1);

        if(sample_feet_only_state)
        {
            initial_states.push_back(feet_only_state);
        }

        // sample a feet + one hand state
        if(sample_feet_and_one_hand_state || sample_feet_and_two_hands_state)
        {
            RPYTF left_hand_pose, right_hand_pose;
            const float mean_horizontal_yaw = feet_only_state->getFeetMeanHorizontalYaw();
            const RotationMatrix robot_yaw_rotation = RPYToSO3(RPYTF(0, 0, 0, 0, 0, mean_horizontal_yaw));

            Translation3D global_left_shoulder_position = robot_yaw_rotation * left_relative_shoulder_position + feet_only_state->mean_feet_position_;
            std::array<float,2> left_arm_orientation = {mean_horizontal_yaw + 90.0 - 60.0 * unit_unif(rng_), 20.0 * unit_unif(rng_)};
            handPoseSampling(left_arm, global_left_shoulder_position, left_arm_orientation, left_hand_pose);

            while((left_hand_pose.getXYZ() - initial_com).norm() > max_com_hand_dist && invalid_sampling_counter < 100)
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
            std::shared_ptr<ContactState> feet_and_one_hand_state = std::make_shared<ContactState>(feet_and_one_hand_stance, initial_com, initial_com_dot, initial_lmom, initial_amom, 1);

            if(sample_feet_and_one_hand_state)
            {
                initial_states.push_back(feet_and_one_hand_state);
            }

            if(sample_feet_and_two_hands_state)
            {
                // sample a feet + two hands state
                Translation3D global_right_shoulder_position = robot_yaw_rotation * right_relative_shoulder_position + feet_only_state->mean_feet_position_;
                std::array<float,2> right_arm_orientation = {mean_horizontal_yaw - 90.0 + 60.0 * unit_unif(rng_), 20.0 * unit_unif(rng_)};
                handPoseSampling(right_arm, global_right_shoulder_position, right_arm_orientation, right_hand_pose);

                while((right_hand_pose.getXYZ() - initial_com).norm() > max_com_hand_dist && invalid_sampling_counter < 100)
                {
                    handPoseSampling(right_arm, global_right_shoulder_position, right_arm_orientation, right_hand_pose);
                    invalid_sampling_counter++;
                }
                if(invalid_sampling_counter >= 100)
                {
                    continue;
                }

                std::array<bool,ContactManipulator::MANIP_NUM> feet_and_two_hands_contact_status = {true,true,true,true};
                std::shared_ptr<Stance> feet_and_two_hands_stance = std::make_shared<Stance>(left_foot_pose, right_foot_pose, left_hand_pose, right_hand_pose,
                                                                                            feet_and_two_hands_contact_status);
                std::shared_ptr<ContactState> feet_and_two_hands_state = std::make_shared<ContactState>(feet_and_two_hands_stance, initial_com, initial_com_dot, initial_lmom, initial_amom, 1);

                initial_states.push_back(feet_and_two_hands_state);
            }
        }
    }

    heuristics_type_ = PlanningHeuristicsType::EUCLIDEAN; // supress the error message

    std::vector<ContactManipulator> branching_manips = ALL_MANIPULATORS;
    for(auto & initial_state : initial_states)
    {
        // RAVELOG_WARN("New initial state.\n");
        branchingContacts(initial_state, branching_mode, specified_motion_code);
    }
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


std::vector< std::vector<RPYTF> > getAllContactPoseCombinations(std::vector<RPYTF> contact_poses_vector)
{
    unsigned int contact_pose_num = contact_poses_vector.size();
    std::vector< std::vector<RPYTF> > possible_contact_pose_representation(contact_pose_num);
    float angle_duplication_range = 90;

    for(unsigned int i = 0; i < contact_pose_num; i++)
    {
        RPYTF contact_pose = contact_poses_vector[i];
        std::array<std::vector<float>,3> possible_rpy;

        for(int j = 3; j < 6; j++)
        {
            possible_rpy[j-3].push_back(contact_pose.getXYZRPY()[j]);
            if(contact_pose.getXYZRPY()[j] > 180-angle_duplication_range/2.0)
            {
                possible_rpy[j-3].push_back(contact_pose.getXYZRPY()[j]-360);
            }
            else if(contact_pose.getXYZRPY()[j] < -180+angle_duplication_range/2.0)
            {
                possible_rpy[j-3].push_back(contact_pose.getXYZRPY()[j]+360);
            }
        }

        for(auto & roll : possible_rpy[0])
        {
            for(auto & pitch : possible_rpy[1])
            {
                for(auto & yaw : possible_rpy[2])
                {
                    possible_contact_pose_representation[i].push_back(RPYTF(contact_pose.x_, contact_pose.y_, contact_pose.z_, roll, pitch, yaw));
                }
            }
        }
    }

    std::vector< std::vector<RPYTF> > all_contact_pose_combinations;
    std::vector<RPYTF> contact_pose_combination_placeholder(contact_pose_num);
    getAllContactPoseCombinations(all_contact_pose_combinations, possible_contact_pose_representation, 0, contact_pose_combination_placeholder);

    return all_contact_pose_combinations;
}
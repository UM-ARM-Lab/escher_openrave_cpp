#include "Utilities.hpp"
#include <omp.h>

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
                                           bool _use_dynamics_planning):
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
general_ik_interface_(_general_ik_interface)
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

    RAVELOG_INFO("Initialize dynamics optimizer interface.\n");
    dynamics_optimizer_interface_vector_.resize(2 * std::max(foot_transition_model_.size(), hand_transition_model_.size()));

    for(unsigned int i = 0; i < dynamics_optimizer_interface_vector_.size(); i++)
    {
        dynamics_optimizer_interface_vector_[i] = std::make_shared<OptimizationInterface>(STEP_TRANSITION_TIME, "SL_optim_config_template/cfg_kdopt_demo.yaml");
    }

    // RAVELOG_INFO("Initialize neural network interface.\n");
    // // neural_network_interface_vector_.resize(2 * std::max(foot_transition_model_.size(), hand_transition_model_.size()));
    // neural_network_interface_vector_.resize(1);

    // for(unsigned int i = 0; i < neural_network_interface_vector_.size(); i++)
    // {
    //     neural_network_interface_vector_[i] = std::make_shared<NeuralNetworkInterface>("dynopt_result/objective_regression_nn_models/", "dynopt_result/feasibility_classification_nn_models/");
    // }

    // general_ik_interface_vector_.resize(2 * std::max(foot_transition_model_.size(), hand_transition_model_.size()));

    // for(int i = 0; i < general_ik_interface_vector_.size(); i++)
    // {
    //     // OpenRAVE::EnvironmentBasePtr cloned_env = general_ik_interface_->env_->CloneSelf(OpenRAVE::Clone_Bodies);
    //     // general_ik_interface_vector_[i] = std::make_shared<GeneralIKInterface>(cloned_env, cloned_env->GetRobot(general_ik_interface_->robot_->GetName()));
    //     general_ik_interface_vector_[i] = general_ik_interface_;
    // }
}

std::vector< std::shared_ptr<ContactState> > ContactSpacePlanning::ANAStarPlanning(std::shared_ptr<ContactState> initial_state, std::array<float,3> goal,
                                                                                   float goal_radius, PlanningHeuristicsType heuristics_type,
                                                                                   BranchingMethod branching_method,
                                                                                   float time_limit, float epsilon,
                                                                                   bool output_first_solution, bool goal_as_exact_poses,
                                                                                   bool use_learned_dynamics_model, bool enforce_stop_in_the_end)
{
    RAVELOG_INFO("Start ANA* planning.\n");

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

    std::vector< std::vector<std::shared_ptr<ContactState> > > all_solution_contact_paths;

    {
        OpenRAVE::EnvironmentMutex::scoped_lock lockenv(general_ik_interface_->env_->GetMutex());

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
                    branchingSearchTree(current_state, branching_method);
                }
            }

            if(output_first_solution || over_time_limit)
            {
                break;
            }

        }

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

        std::ofstream planning_dynamics_cost_fstream("contact_planning_dynamics_costs_weight_3_0_test_env_6_epsilon_0.txt", std::ofstream::app);

        float total_learned_dynamics_cost = contact_state_path[contact_state_path.size()-1]->accumulated_dynamics_cost_;
        float total_dynamics_cost_segment_by_segment = 0;
        float total_dynamics_cost = 0;
        if(use_dynamics_planning_)
        {
            // // getchar();

            // total_dynamics_cost_segment_by_segment = fillDynamicsSequenceSegmentBySegment(contact_state_path);
            // total_dynamics_cost = fillDynamicsSequence(contact_state_path);

            // // kinematicsVerification(contact_state_path);
            // // kinematicsVerification_StateOnly(contact_state_path);

            // std::cout << planning_id_ << " " << total_learned_dynamics_cost << " "
            //                                  << total_dynamics_cost_segment_by_segment << " "
            //                                  << total_dynamics_cost << std::endl;

            // planning_dynamics_cost_fstream << planning_id_ << " " << total_learned_dynamics_cost << " "
            //                                                       << total_dynamics_cost_segment_by_segment << " "
            //                                                       << total_dynamics_cost << std::endl;

            // // for(int i = all_solution_contact_paths.size()-1; i >= 0; i--)
            // // {
            // //     std::cout << "Solution Path " << i << ": " << std::endl;
            // //     verifyContactSequenceDynamicsFeasibilityPrediction(all_solution_contact_paths[i]);

            // //     // total_dynamics_cost_segment_by_segment = fillDynamicsSequenceSegmentBySegment(all_solution_contact_paths[i]);
            // //     // total_dynamics_cost = fillDynamicsSequence(all_solution_contact_paths[i]);

            // //     // std::cout << planning_id_ << " " << all_solution_contact_paths[i][all_solution_contact_paths[i].size()-1]->accumulated_dynamics_cost_
            // //     //                           << " " << total_dynamics_cost_segment_by_segment
            // //     //                           << " " << total_dynamics_cost << std::endl;
            // // }
        }
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
        }
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
            std::cout << "dynamics cost: "<< contact_state->accumulated_dynamics_cost_ - contact_state->parent_->accumulated_dynamics_cost_ << std::endl;
        }
        std::cout << "com: " << contact_state->com_[0] << " " << contact_state->com_[1] << " " << contact_state->com_[2] << std::endl;
        std::cout << "com_dot: " << contact_state->com_dot_[0] << " " << contact_state->com_dot_[1] << " " << contact_state->com_dot_[2] << std::endl;
    }

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
    for(auto & contact_state : contact_state_path)
    {
        general_ik_interface_->resetContactStateRelatedParameters();

        // get the pose of contacting end-effectors
        for(auto & manip : ALL_MANIPULATORS)
        {
            if(contact_state->stances_vector_[0]->ee_contact_status_[manip])
            {
                std::cout << robot_properties_->manipulator_name_map_[manip] << ": "
                          << contact_state->stances_vector_[0]->ee_contact_poses_[manip].x_ << " "
                          << contact_state->stances_vector_[0]->ee_contact_poses_[manip].y_ << " "
                          << contact_state->stances_vector_[0]->ee_contact_poses_[manip].z_ << std::endl;

                general_ik_interface_->addNewManipPose(robot_properties_->manipulator_name_map_[manip], contact_state->stances_vector_[0]->ee_contact_poses_[manip].GetRaveTransform());
                general_ik_interface_->addNewContactManip(robot_properties_->manipulator_name_map_[manip], MU);
            }
        }

        // get the CoM and transform it from SL frame to OpenRAVE frame
        Translation3D com = contact_state->com_;
        general_ik_interface_->CenterOfMass()[0] = com[0];
        general_ik_interface_->CenterOfMass()[1] = com[1];
        general_ik_interface_->CenterOfMass()[2] = com[2];
        ik_result = general_ik_interface_->solve();
        general_ik_interface_->q0() = ik_result.second;

        std::cout << "com: " << com[0] << ", " << com[1] << ", " << com[2] << std::endl;
        std::cout << "result: " << ik_result.first << std::endl;

        getchar();
    }
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
        }
        std::cout << "com: " << contact_state->com_[0] << " " << contact_state->com_[1] << " " << contact_state->com_[2] << std::endl;
        std::cout << "com_dot: " << contact_state->com_dot_[0] << " " << contact_state->com_dot_[1] << " " << contact_state->com_dot_[2] << std::endl;
    }

    /*********************************FAILED ATTEMPT TO USE KINEMATICS SOLVER BASED ON SL*********************************/
    std::shared_ptr<OptimizationInterface> kinematics_optimization_interface =
    std::make_shared<OptimizationInterface>(STEP_TRANSITION_TIME, "SL_optim_config_template/cfg_kdopt_demo.yaml");

    kinematics_optimization_interface->updateContactSequence(contact_state_path);
    kinematics_optimization_interface->dynamicsSequenceConcatenation(dynamics_sequence_vector);

    // // kinematics optimization
    // std::cout << "Start the Kinematics Optimization." << std::endl;
    // kinematics_optimization_interface->simplifiedKinematicsOptimization();
    // std::cout << "Finished the Kinematics Optimization." << std::endl;
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
    std::vector<OpenRAVE::dReal> DOF0 = robot_properties_->IK_init_DOF_Values_;
    DOF0[robot_properties_->DOFName_index_map_["x_prismatic_joint"]] = current_state->mean_feet_position_[0];
    DOF0[robot_properties_->DOFName_index_map_["y_prismatic_joint"]] = current_state->mean_feet_position_[1];
    DOF0[robot_properties_->DOFName_index_map_["z_prismatic_joint"]] = current_state->mean_feet_position_[2] + 1.0;
    DOF0[robot_properties_->DOFName_index_map_["yaw_revolute_joint"]] = current_state->getFeetMeanHorizontalYaw() * DEG2RAD - M_PI/2.0;
    general_ik_interface->robot_->SetDOFValues(DOF0);
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
            dynamically_feasible = neural_network_interface_vector_[index]->dynamicsPrediction(current_state, dynamics_cost);
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
            std::vector< std::shared_ptr<ContactState> > contact_state_sequence = {current_state->parent_, current_state};

            dynamics_optimizer_interface_vector_[index]->updateContactSequence(contact_state_sequence);

            // bool dynamically_feasible = dynamics_optimizer_interface_vector_[index]->simplifiedDynamicsOptimization(dynamics_cost);
            dynamically_feasible = dynamics_optimizer_interface_vector_[index]->dynamicsOptimization(dynamics_cost);

            if(dynamically_feasible)
            {
                // update com, com_dot, and parent edge dynamics sequence of the current_state
                dynamics_optimizer_interface_vector_[index]->updateStateCoM(current_state);
                dynamics_optimizer_interface_vector_[index]->recordEdgeDynamicsSequence(current_state);
            }

            dynamics_optimizer_interface_vector_[index]->storeDynamicsOptimizationResult(current_state, dynamics_cost, dynamically_feasible, planning_id_);
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

        bool left_hand_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::L_ARM];
        bool right_hand_in_contact = current_state->stances_vector_[0]->ee_contact_status_[ContactManipulator::R_ARM];

        // for sampling contact transition mode 0 1
        // if(left_hand_in_contact || right_hand_in_contact)
        // {
        //     return false;
        // }

        // for sampling contact transition mode 2 3 4 5 6 7 8 9
        if(!(left_hand_in_contact || right_hand_in_contact || current_state->is_root_))
        {
            return false;
        }

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
                projection_is_successful = footProjection(move_manip, new_left_foot_pose);
            }
            else if(move_manip == ContactManipulator::R_LEG)
            {
                projection_is_successful = footProjection(move_manip, new_right_foot_pose);
            }

            if(projection_is_successful)
            {
                // RAVELOG_INFO("construct state.\n");

                // construct the new state
                std::shared_ptr<Stance> new_stance(new Stance(new_left_foot_pose, new_right_foot_pose, new_left_hand_pose, new_right_hand_pose, new_ee_contact_status));

                std::shared_ptr<ContactState> new_contact_state(new ContactState(new_stance, current_state, move_manip, 1, robot_properties_->robot_z_));

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
    const float mean_hotizontal_yaw = current_state->getFeetMeanHorizontalYaw();
    const RotationMatrix robot_yaw_rotation = RPYToSO3(RPYTF(0, 0, 0, 0, 0, mean_hotizontal_yaw));

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
                        global_arm_orientation[0] = mean_hotizontal_yaw + 90.0 - arm_orientation[0];
                        global_arm_orientation[1] = arm_orientation[1];
                    }
                    else if(manip == ContactManipulator::R_ARM)
                    {
                        global_arm_orientation[0] = mean_hotizontal_yaw - 90.0 + arm_orientation[0];
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

    #pragma omp parallel num_threads(thread_num_) shared (branching_hands_combination, state_feasibility_check_result)
    {
        #pragma omp for schedule(static)
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
                if(move_manip == ContactManipulator::L_ARM)
                {
                    projection_is_successful = handProjection(move_manip, global_left_shoulder_position, global_arm_orientation, new_left_hand_pose);
                }
                else if(move_manip == ContactManipulator::R_ARM)
                {
                    projection_is_successful = handProjection(move_manip, global_right_shoulder_position, global_arm_orientation, new_right_hand_pose);
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

void ContactSpacePlanning::insertState(std::shared_ptr<ContactState> current_state, float dynamics_cost)
{
    std::shared_ptr<ContactState> prev_state = current_state->parent_;
    // calculate the edge cost and the cost to come
    current_state->g_ = prev_state->g_ + getEdgeCost(prev_state, current_state, dynamics_cost);

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

float ContactSpacePlanning::getEdgeCost(std::shared_ptr<ContactState> prev_state, std::shared_ptr<ContactState> current_state, float dynamics_cost)
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

    return (traveling_distance_cost + orientation_cost + step_cost + dynamics_cost_weight_ * dynamics_cost);
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
    // return std::sqrt(std::pow(goal_[0]-current_state->mean_feet_position_[0], 2) + std::pow(goal_[1]-current_state->mean_feet_position_[1], 2)) <= goal_radius_;
    // return current_state->mean_feet_position_[0] > goal_[0]-goal_radius_;
    return current_state->max_manip_x_ > (goal_[0] - goal_radius_);
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
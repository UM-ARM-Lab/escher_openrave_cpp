#include "Utilities.hpp"

bool TorsoPoseState::operator==(const TorsoPoseState& other) const
{
    return (this->pose_ == other.pose_);
}

bool TorsoPoseState::operator!=(const TorsoPoseState& other) const
{
    return !(*this == other);
}

TorsoPathPlanning::TorsoPathPlanning(OpenRAVE::EnvironmentBasePtr _env,
                                     std::shared_ptr<RobotProperties> _robot_properties,
                                     float _position_transition_max_radius,
                                     float _orientation_transition_max_radius,
                                     std::array<float,3> _position_resolution,
                                     std::vector< std::shared_ptr<TrimeshSurface> > _structures,
                                     std::map<int, std::shared_ptr<TrimeshSurface> > _structures_dict,
                                     int _thread_num,
                                     std::shared_ptr<DrawingHandler> _drawing_handler,
                                     int _planning_id):
    position_transition_max_radius_(_position_transition_max_radius),
    orientation_transition_max_radius_(_orientation_transition_max_radius),
    robot_properties_(_robot_properties),
    position_resolution_(_position_resolution),
    structures_(_structures),
    structures_dict_(_structures_dict),
    thread_num_(_thread_num),
    env_(_env),
    drawing_handler_(_drawing_handler),
    planning_id_(_planning_id)
{
    // construct the position transition model
    int x_step_num = int(floor(_position_transition_max_radius / _position_resolution[0]));
    int y_step_num = int(floor(_position_transition_max_radius / _position_resolution[1]));
    int z_step_num = int(floor(_position_transition_max_radius / _position_resolution[2]));

    for(int ix = -x_step_num; ix <= x_step_num; ix++)
    {
        for(int iy = -y_step_num; iy <= y_step_num; iy++)
        {
            for(int iz = -z_step_num; iz <= z_step_num; iz++)
            {
                Translation3D translation(ix*_position_resolution[0], iy*_position_resolution[1], iz*_position_resolution[2]);
                if(translation.norm() <= _position_transition_max_radius)
                {
                    position_transition_model_.push_back(translation);
                }
            }
        }
    }

    // construct the orientation transition model
    // read all the orientation into quaternion
    std::vector<Vector3D> dummy_successor(1, Vector3D(0,0,0));
    orientation_transition_model_.insert({ Vector3D(0,0,0), dummy_successor});

    torso_kinbody_ = env_->GetKinBody("body_collision_box");
}

std::vector< std::shared_ptr<TorsoPoseState> > TorsoPathPlanning::AStarPlanning(std::shared_ptr<TorsoPoseState> initial_state, RPYTF goal, float time_limit)
{
    RAVELOG_INFO("Start A* planning.\n");

    // initialize parameters
    goal_ = goal;
    time_limit_ = time_limit;

    // clear the heap and state list, and add initial state in the list
    while(!open_heap_.empty())
    {
        open_heap_.pop();
    }
    pose_states_map_.clear();

    open_heap_.push(initial_state);
    pose_states_map_.insert(std::make_pair(std::hash<TorsoPoseState>()(*initial_state), initial_state));

    auto time_before_ANA_start_planning = std::chrono::high_resolution_clock::now();

    // main exploration loop
    bool over_time_limit = false;
    std::shared_ptr<TorsoPoseState> goal_state;

    RAVELOG_INFO("Enter the exploration loop.\n");

    int drawing_counter = 0;

    std::vector< std::tuple<int,float,float,float,int> > planning_result; // planning time, path cost, dynamics cost, step num

    {
        OpenRAVE::EnvironmentMutex::scoped_lock lockenv(env_->GetMutex());

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
                std::shared_ptr<TorsoPoseState> current_state;
                current_state = open_heap_.top();
                open_heap_.pop();

                if(current_state->explore_state_ == ExploreState::OPEN)
                {
                    if(!feasibilityCheck(current_state))
                    {
                        current_state->explore_state_ = ExploreState::CLOSED;
                        continue;
                    }

                    current_state->explore_state_ = ExploreState::CLOSED;

                    // plotting the contact sequence so far
                    // RAVELOG_INFO("Plot the contact sequence.\n");
                    // drawing_handler_->ClearHandler();
                    // drawing_handler_->DrawTorsoPath(current_state);
                    if(drawing_counter == 10)
                    {
                        drawing_handler_->ClearHandler();
                        drawing_handler_->DrawTorsoPath(current_state);
                        drawing_counter = 0;
                    }

                    drawing_counter++;

                    // check if it reaches the goal
                    // RAVELOG_INFO("Check if it reaches the goal.\n");
                    if(current_state->pose_ == goal_)
                    {
                        goal_state = current_state;

                        int step_count = 0;
                        std::shared_ptr<TorsoPoseState> path_state = goal_state;

                        std::vector<std::shared_ptr<TorsoPoseState> > solution_torso_path;

                        while(true)
                        {
                            solution_torso_path.push_back(std::make_shared<TorsoPoseState>(*path_state));
                            if(path_state->is_root_)
                            {
                                break;
                            }

                            path_state = path_state->parent_;
                            step_count++;
                        }

                        std::reverse(solution_torso_path.begin(), solution_torso_path.end());
                        for(unsigned int i = 0; i < solution_torso_path.size(); i++)
                        {
                            if(!solution_torso_path[i]->is_root_)
                            {
                                solution_torso_path[i]->parent_ = solution_torso_path[i-1];
                            }
                        }

                        current_time = std::chrono::high_resolution_clock::now();

                        float planning_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - time_before_ANA_start_planning).count() /1000.0;


                        RAVELOG_INFO("Solution Found: T = %5.3f, # of steps: %d, total cost: %5.3f. \n", planning_time, step_count, goal_state->g_);

                        drawing_handler_->ClearHandler();
                        drawing_handler_->DrawTorsoPath(goal_state);

                        return solution_torso_path;
                    }

                    branchingSearchTree(current_state);

                }
            }

            if(over_time_limit)
            {
                break;
            }
        }

    }

    drawing_handler_->ClearHandler();

    return std::vector<std::shared_ptr<TorsoPoseState> >();
}


bool TorsoPathPlanning::feasibilityCheck(std::shared_ptr<TorsoPoseState> current_state)
{
    // std::shapred_ptr<TorsoPoseState> prev_state = current_state->parent_;

    // TODO: add the collision check on the edge
    torso_kinbody_->SetTransform(current_state->pose_.GetRaveTransform());

    return !env_->CheckCollision(torso_kinbody_);

    // for(auto & structure : structures_)
    // {
    //     if(env_->CheckCollision(torso_kinbody_, structure->getKinbody()))
    //     {
    //         return false;
    //     }
    // }

    // return true;
}

float TorsoPathPlanning::getHeuristics(std::shared_ptr<TorsoPoseState> current_state)
{
    return euclideanDistance3D(goal_.getXYZ(), current_state->pose_.getXYZ());
    // TODO: add orientation distance
}

float TorsoPathPlanning::getEdgeCost(std::shared_ptr<TorsoPoseState> prev_state, std::shared_ptr<TorsoPoseState> current_state)
{
    return euclideanDistance3D(current_state->pose_.getXYZ(), prev_state->pose_.getXYZ());
    // TODO: add orientation distance
}

void TorsoPathPlanning::branchingSearchTree(std::shared_ptr<TorsoPoseState> current_state)
{
    Translation3D current_translation = current_state->pose_.getXYZ();
    Vector3D current_orientation = current_state->pose_.getRPY();
    vector<Vector3D> next_orientation_vector = orientation_transition_model_.find(current_orientation)->second;

    for(auto & position_transition : position_transition_model_)
    {
        Translation3D next_position = current_translation + position_transition;

        for(auto & next_orientation : next_orientation_vector)
        {
            RPYTF next_pose(next_position[0], next_position[1], next_position[2], next_orientation[0], next_orientation[1], next_orientation[2]);

            std::shared_ptr<TorsoPoseState> new_pose_state = std::make_shared<TorsoPoseState>(next_pose, current_state);

            insertState(new_pose_state);
        }
    }

    if(euclideanDistance3D(current_translation, goal_.getXYZ()) < position_transition_max_radius_)
    {
        Translation3D next_position = goal_.getXYZ();

        for(auto & next_orientation : next_orientation_vector)
        {
            RPYTF next_pose(next_position[0], next_position[1], next_position[2], next_orientation[0], next_orientation[1], next_orientation[2]);

            std::shared_ptr<TorsoPoseState> new_pose_state = std::make_shared<TorsoPoseState>(next_pose, current_state);

            insertState(new_pose_state);
        }

        if(orientationDistance(current_orientation, goal_.getRPY()) < orientation_transition_max_radius_)
        {
            std::shared_ptr<TorsoPoseState> new_pose_state = std::make_shared<TorsoPoseState>(goal_, current_state);
            insertState(new_pose_state);
        }

    }
}

void TorsoPathPlanning::insertState(std::shared_ptr<TorsoPoseState> current_state)
{
    std::shared_ptr<TorsoPoseState> prev_state = current_state->parent_;
    // calculate the edge cost and the cost to come
    current_state->g_ = prev_state->g_ + getEdgeCost(prev_state, current_state);

    // calculate the heuristics (cost to go)
    current_state->h_ = getHeuristics(current_state);

    // find if there already exists this state
    std::size_t current_state_hash = std::hash<TorsoPoseState>()(*current_state);
    std::unordered_map<std::size_t, std::shared_ptr<TorsoPoseState> >::iterator pose_state_iterator = pose_states_map_.find(current_state_hash);

    // add the state to the state vector and/or the open heap
    if (pose_state_iterator == pose_states_map_.end()) // the state is not in the set
    {
        // RAVELOG_INFO("New state.\n");
        current_state->priority_value_ = -current_state->getF();
        pose_states_map_.insert(std::make_pair(current_state_hash, current_state));
        open_heap_.push(current_state);
    }
    else
    {
        // RAVELOG_INFO("Existing state.\n");
        std::shared_ptr<TorsoPoseState> existing_state = pose_state_iterator->second;

        if(existing_state->explore_state_ != ExploreState::CLOSED && current_state->getF() < existing_state->getF())
        {
            existing_state->g_ = current_state->g_;
            existing_state->parent_ = current_state->parent_;
            existing_state->priority_value_ = -existing_state->getF();
            open_heap_.push(existing_state);
            current_state->priority_value_ = existing_state->priority_value_;
        }
    }
}
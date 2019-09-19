#ifndef TORSOPATHPLANNING_HPP
#define TORSOPATHPLANNING_HPP

class TorsoPoseState
{
    public:
        TorsoPoseState(RPYTF _pose):
        pose_(_pose),
        is_root_(true),
        explore_state_(ExploreState::OPEN),
        g_(0.0),
        h_(0.0),
        priority_value_(-9999.0) {};

        TorsoPoseState(RPYTF _pose, std::shared_ptr<TorsoPoseState> _parent):
        pose_(_pose),
        is_root_(false),
        parent_(_parent),
        explore_state_(ExploreState::OPEN),
        g_(_parent->g_),
        h_(0) {};

        RPYTF pose_;

        const bool is_root_;

        ExploreState explore_state_;

        float g_;
        float h_;
        float priority_value_;

        std::shared_ptr<TorsoPoseState> parent_;

        bool operator==(const TorsoPoseState& other) const;
        bool operator!=(const TorsoPoseState& other) const;

        inline bool operator<(const TorsoPoseState& other) const {return (this->priority_value_ < other.priority_value_);}

        // inline std::shared_ptr<ContactState> getParent() {return parent_;}
        inline const float getF() const {return (g_ + h_);}

};

namespace std
{
    template <>
    class hash<TorsoPoseState>{
        public:
            size_t operator()(const TorsoPoseState &torso_pose_state) const
            {
                size_t hash_number = hash<RPYTF>()(torso_pose_state.pose_);

                return hash_number;
            }
    };
}

class TorsoPathPlanning
{
    public:
        TorsoPathPlanning(OpenRAVE::EnvironmentBasePtr _env,
                          std::shared_ptr<RobotProperties> _robot_properties,
                        //   std::vector< Translation3D > _position_transition_model,
                        //   std::unordered_map< Vector3D, std::pair<Vector3D, float> > _orientation_transition_model,
                          float _position_transition_max_radius,
                          float _orientation_transition_max_radius,
                          std::array<float,3> _position_resolution,
                          std::vector< std::shared_ptr<TrimeshSurface> > _structures,
                          std::map<int, std::shared_ptr<TrimeshSurface> > _structures_dict,
                          int _thread_num,
                          std::shared_ptr<DrawingHandler> _drawing_handler,
                          int _planning_id);

        std::vector< std::shared_ptr<TorsoPoseState> > AStarPlanning(std::shared_ptr<TorsoPoseState> initial_state, RPYTF goal, float time_limit);

    private:
        std::priority_queue< std::shared_ptr<TorsoPoseState>, std::vector< std::shared_ptr<TorsoPoseState> >, pointer_less > open_heap_;

        // the vector of ContactStates
        std::unordered_map<std::size_t, std::shared_ptr<TorsoPoseState> > pose_states_map_;

        // the A* parameters
        float time_limit_;

        RPYTF goal_;

        // the robot priorities
        std::shared_ptr<RobotProperties> robot_properties_;

        // planner options
        float position_transition_max_radius_;
        float orientation_transition_max_radius_;
        // const float collision_check_step_size =

        // transition models
        const std::array<float,3> position_resolution_;
        std::vector< Translation3D > position_transition_model_; // incremental change in position to the successors
        std::unordered_map< Vector3D, std::vector<Vector3D>, EigenVectorHash> orientation_transition_model_; // mapping from current orientation to new orientations

        // cost parameters
        const float step_cost_weight_ = 3.0;
        const float orientation_cost_weight_ = 0.1;

        // thread number for OpenMP
        const int thread_num_;

        // idicate which planning trial it is, for record.
        const int planning_id_;

        // the environment structures
        const std::vector< std::shared_ptr<TrimeshSurface> > structures_;
        const std::map<int, std::shared_ptr<TrimeshSurface> > structures_dict_;

        // OpenRAVE object
        OpenRAVE::EnvironmentBasePtr env_;
        OpenRAVE::KinBodyPtr torso_kinbody_;
        // std::map<ContactManipulator, OpenRAVE::KinBodyPtr> manipulator_workspace_kinbodies_;

        // the drawing handler
        std::shared_ptr<DrawingHandler> drawing_handler_;

        bool feasibilityCheck(std::shared_ptr<TorsoPoseState> current_state);

        float getHeuristics(std::shared_ptr<TorsoPoseState> current_state);
        float getEdgeCost(std::shared_ptr<TorsoPoseState> prev_state, std::shared_ptr<TorsoPoseState> current_state);

        void branchingSearchTree(std::shared_ptr<TorsoPoseState> current_state);

        void insertState(std::shared_ptr<TorsoPoseState> current_state);
};


#endif
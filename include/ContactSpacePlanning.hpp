#ifndef CONTACTSPACEPLANNING_HPP
#define CONTACTSPACEPLANNING_HPP

class ContactSpacePlanning
{
    public:
        ContactSpacePlanning(std::shared_ptr<RobotProperties> _robot_properties,
                             std::vector< std::array<float,3> > _foot_transition_model,
                             std::vector< std::array<float,2> > _hand_transition_model,
                             std::vector< std::shared_ptr<TrimeshSurface> > _structures,
                             std::map<int, std::shared_ptr<TrimeshSurface> > _structures_dict,
                             int _num_stance_in_state,
                             std::shared_ptr<DrawingHandler> _drawing_handler);

        std::vector< std::shared_ptr<ContactState> > ANAStarPlanning(std::shared_ptr<ContactState> initial_state, std::array<float,3> goal,
                                                                     float goal_radius, PlanningHeuristicsType heuristics_type,
                                                                     float time_limit, bool output_first_solution, bool goal_as_exact_poses);

    private:
        // std::set< std::shared_ptr<ContactState>, ContactState::pointer_less > openHeap;
        std::priority_queue< std::shared_ptr<ContactState>, std::vector< std::shared_ptr<ContactState> >, ContactState::pointer_less > open_heap_;

        // the vector of ContactStates
        std::unordered_map<std::size_t, ContactState> contact_states_map_;

        // the ANA* parameters
        float G_, E_;
        std::array<float,3> goal_; // x y theta
        float goal_radius_;
        float time_limit_;

        // the robot priorities
        std::shared_ptr<RobotProperties> robot_properties_;

        // planner options
        bool goal_as_exact_poses_;
        PlanningHeuristicsType heuristics_type_;
        int num_stance_in_state_;

        // transition models
        const std::vector< std::array<float,3> > foot_transition_model_;
        const std::vector< std::array<float,2> > hand_transition_model_;

        // cost parameters
        const float step_cost_weight_ = 3.0;

        // the environment structures
        const std::vector< std::shared_ptr<TrimeshSurface> > structures_;
        std::vector< std::shared_ptr<TrimeshSurface> > hand_structures_;
        std::vector< std::shared_ptr<TrimeshSurface> > foot_structures_;
        const std::map<int, std::shared_ptr<TrimeshSurface> > structures_dict_;

        // the drawing handler
        std::shared_ptr< DrawingHandler > drawing_handler_;

        // the dynamics optimizer interface
        std::shared_ptr< DynOptInterface > dynamics_optimizer_interface_;

        bool kinematicFeasibilityCheck(std::shared_ptr<ContactState> current_state);
        bool dynamicFeasibilityCheck(std::shared_ptr<ContactState> current_state, float& dynamics_cost);
        bool stateFeasibilityCheck(std::shared_ptr<ContactState> current_state, float& dynamics_cost);

        float getHeuristics(std::shared_ptr<ContactState> current_state);
        float getEdgeCost(std::shared_ptr<ContactState> prev_state, std::shared_ptr<ContactState> current_state, float dynamics_cost);

        void branchingSearchTree(std::shared_ptr<ContactState> current_state);
        void branchingFootContacts(std::shared_ptr<ContactState> current_state, std::vector<ContactManipulator> branching_manips);
        void branchingHandContacts(std::shared_ptr<ContactState> current_state, std::vector<ContactManipulator> branching_manips);
        bool footProjection(RPYTF& projection_pose);
        bool handProjection();

        void insertState(std::shared_ptr<ContactState> current_state, float dynamics_cost);

        void updateExploreStatesAndOpenHeap();
        bool isReachedGoal(std::shared_ptr<ContactState> current_state);

};

#endif
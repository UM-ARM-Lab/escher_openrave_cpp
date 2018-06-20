
#ifndef CONTACTSPACEPLANNING_HPP
#define CONTACTSPACEPLANNING_HPP

class ContactSpacePlanning
{
    public:
        ContactSpacePlanning(bool _goal_as_exact_poses, 
                             RobotProperties _robot_properties,
                             std::vector< std::array<float,3> > _foot_transition_model, 
                             std::vector< std::array<float,2> > _hand_transition_model,
                             PlanningHeuristicsType _heuristics_type,
                             std::vector< std::shared_ptr<TrimeshSurface> > _structures,
                             std::map<int, std::shared_ptr<TrimeshSurface> > _structures_dict);

    private:
        // std::set< std::shared_ptr<ContactState>, ContactState::pointer_less > openHeap;
        std::priority_queue< std::shared_ptr<ContactState>, std::vector< std::shared_ptr<ContactState> >, ContactState::pointer_less > open_heap_;

        // the vector of ContactStates
        std::vector<std::shared_ptr<ContactState> > contact_states_vector_;

        // the ANA* parameters
        float G_, E_;
        std::array<float,3> goal_; // x y theta
        float goal_radius_;
        float time_limit_;

        // the robot priorities
        const RobotProperties robot_properties_;

        // planner options
        const bool goal_as_exact_poses_;
        const PlanningHeuristicsType heuristics_type_;
        
        // transition models
        const std::vector< std::array<float,3> > foot_transition_model_;
        const std::vector< std::array<float,2> > hand_transition_model_;

        // the environment structures
        const std::vector< std::shared_ptr<TrimeshSurface> > structures_;
        const std::map<int, std::shared_ptr<TrimeshSurface> > structures_dict_;

        std::vector< std::shared_ptr<ContactState> > ANAStarPlanning(std::shared_ptr<ContactState> initial_state, std::array<float,3> goal, float goal_radius, float time_limit);

        bool kinematicFeasibilityCheck(std::shared_ptr<ContactState> current_state);
        float getDynamicScore();

        void branchingSearchTree(std::shared_ptr<ContactState> current_state);
        void footProjection();
        void handProjection();

        float getHeuristics(std::shared_ptr<ContactState> current_state);
        float getEdgeCost(std::shared_ptr<ContactState> prev_state, std::shared_ptr<ContactState> current_state);

        void updateExploreStates();

};

#endif
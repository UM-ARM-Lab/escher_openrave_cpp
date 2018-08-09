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
                             std::shared_ptr<MapGrid> _map_grid,
                             std::shared_ptr<GeneralIKInterface> _general_ik_interface,
                             int _num_stance_in_state,
                             int _thread_num,
                             std::shared_ptr<DrawingHandler> _drawing_handler,
                             int _planning_id,
                             bool _use_dynamics_planning);

        std::vector< std::shared_ptr<ContactState> > ANAStarPlanning(std::shared_ptr<ContactState> initial_state, std::array<float,3> goal,
                                                                     float goal_radius, PlanningHeuristicsType heuristics_type,
                                                                     BranchingMethod branching_method,
                                                                     float time_limit, bool output_first_solution, bool goal_as_exact_poses);

    private:
        // std::set< std::shared_ptr<ContactState>, ContactState::pointer_less > openHeap;
        std::priority_queue< std::shared_ptr<ContactState>, std::vector< std::shared_ptr<ContactState> >, ContactState::pointer_less > open_heap_;

        // the vector of ContactStates
        std::unordered_map<std::size_t, std::shared_ptr<ContactState> > contact_states_map_;

        // the ANA* parameters
        float G_, E_;
        std::array<float,3> goal_; // x y theta
        float goal_radius_;
        float time_limit_;

        // the robot priorities
        std::shared_ptr<RobotProperties> robot_properties_;

        // planner options
        bool use_dynamics_planning_;
        bool goal_as_exact_poses_;
        PlanningHeuristicsType heuristics_type_;
        int num_stance_in_state_;

        // transition models
        const std::vector< std::array<float,3> > foot_transition_model_;
        const std::vector< std::array<float,2> > hand_transition_model_;

        // cost parameters
        const float step_cost_weight_ = 3.0;
        const float dynamics_cost_weight_ = 0.1; // original
        // const float dynamics_cost_weight_ = 1.0; // simplified

        // random parameters
        const float epsilon_ = 0.1;

        // thread number for OpenMP
        const int thread_num_;

        // idicate which planning trial it is, for record.
        const int planning_id_;

        // the environment structures
        const std::vector< std::shared_ptr<TrimeshSurface> > structures_;
        std::vector< std::shared_ptr<TrimeshSurface> > hand_structures_;
        std::vector< std::shared_ptr<TrimeshSurface> > foot_structures_;
        const std::map<int, std::shared_ptr<TrimeshSurface> > structures_dict_;

        // the map grid
        std::shared_ptr<MapGrid> map_grid_;

        // the drawing handler
        std::shared_ptr<DrawingHandler> drawing_handler_;

        // the dynamics optimizer interface
        std::vector< std::shared_ptr<OptimizationInterface> > dynamics_optimizer_interface_vector_;

        // the dynamics prediciton neural network interface
        // std::vector< std::shared_ptr<NeuralNetworkInterface> > neural_network_interface_vector_;

        // the general_ik interface
        std::vector< std::shared_ptr<GeneralIKInterface> > general_ik_interface_vector_;
        std::shared_ptr<GeneralIKInterface> general_ik_interface_;

        void setupStateReachabilityIK(std::shared_ptr<ContactState> current_state, std::shared_ptr<GeneralIKInterface> general_ik_interface);

        bool kinematicsFeasibilityCheck(std::shared_ptr<ContactState> current_state, int index);
        bool dynamicsFeasibilityCheck(std::shared_ptr<ContactState> current_state, float& dynamics_cost, int index);
        bool stateFeasibilityCheck(std::shared_ptr<ContactState> current_state, float& dynamics_cost, int index);

        float getHeuristics(std::shared_ptr<ContactState> current_state);
        float getEdgeCost(std::shared_ptr<ContactState> prev_state, std::shared_ptr<ContactState> current_state, float dynamics_cost);

        void branchingSearchTree(std::shared_ptr<ContactState> current_state, BranchingMethod branching_method);
        void branchingFootContacts(std::shared_ptr<ContactState> current_state, std::vector<ContactManipulator> branching_manips);
        void branchingHandContacts(std::shared_ptr<ContactState> current_state, std::vector<ContactManipulator> branching_manips);
        bool footProjection(ContactManipulator& contact_manipulator, RPYTF& projection_pose);
        bool handProjection(ContactManipulator& contact_manipulator, Translation3D& shoulder_point, std::array<float,2>& arm_orientation, RPYTF& projection_pose);

        void insertState(std::shared_ptr<ContactState> current_state, float dynamics_cost);

        void updateExploreStatesAndOpenHeap();
        bool isReachedGoal(std::shared_ptr<ContactState> current_state);

        void kinematicsVerification(std::vector< std::shared_ptr<ContactState> > contact_state_path);

        void storeDynamicsOptimizationResult(std::shared_ptr<ContactState> current_state, float& dynamics_cost);
};

#endif

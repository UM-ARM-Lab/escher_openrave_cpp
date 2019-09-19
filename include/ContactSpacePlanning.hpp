#ifndef CONTACTSPACEPLANNING_HPP
#define CONTACTSPACEPLANNING_HPP

void getAllContactPoseCombinations(std::vector< std::vector<RPYTF> >& all_contact_pose_combinations, const std::vector<std::vector<RPYTF> >& possible_contact_pose_representation, size_t vec_index, std::vector<RPYTF>& contact_pose_combination);
std::vector< std::vector<RPYTF> > getAllContactPoseCombinations(std::vector<RPYTF> contact_poses_vector);

class ContactSpacePlanning
{
    public:
        ContactSpacePlanning(std::shared_ptr<RobotProperties> _robot_properties,
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
                             std::shared_ptr<DrawingHandler> _drawing_handler,
                             int _planning_id,
                             bool _use_dynamics_planning,
                             std::vector<std::pair<Vector6D, float> > _disturbance_samples,
                             PlanningApplication _planning_application = PlanningApplication::PLAN_IN_ENV,
                             bool _check_zero_step_capturability=true,
                             bool _check_one_step_capturability=true,
                             bool _check_contact_transition_feasibility=true);

        std::vector< std::shared_ptr<ContactState> > ANAStarPlanning(std::shared_ptr<ContactState> initial_state, std::array<float,3> goal,
                                                                     float goal_radius, PlanningHeuristicsType heuristics_type,
                                                                     BranchingMethod branching_method,
                                                                     float time_limit, float epsilon,
                                                                     bool output_first_solution, bool goal_as_exact_poses,
                                                                     bool use_learned_dynamics_model, bool enforce_stop_in_the_end);
        void storeSLEnvironment();

        void collectTrainingData(BranchingManipMode branching_mode=BranchingManipMode::ALL,
                                 bool sample_feet_only_state=true, bool sample_feet_and_one_hand_state=true,
                                 bool sample_feet_and_two_hands_state=true, int specified_motion_code=-1);

    private:
        // std::set< std::shared_ptr<ContactState>, pointer_less > openHeap;
        std::priority_queue< std::shared_ptr<ContactState>, std::vector< std::shared_ptr<ContactState> >, pointer_less > open_heap_;

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
        bool use_dynamics_planning_ = false;
        bool goal_as_exact_poses_ = false;
        bool use_learned_dynamics_model_ = false;
        bool enforce_stop_in_the_end_ = false;
        PlanningHeuristicsType heuristics_type_;
        int num_stance_in_state_;
        PlanningApplication planning_application_;
        bool check_zero_step_capturability_;
        bool check_one_step_capturability_;
        bool check_contact_transition_feasibility_;

        // transition models
        const std::vector< std::array<float,3> > foot_transition_model_;
        const std::vector< std::array<float,2> > hand_transition_model_;
        const std::vector< std::array<float,3> > disturbance_rejection_foot_transition_model_;
        const std::vector< std::array<float,2> > disturbance_rejection_hand_transition_model_;

        // disturbance samples
        std::vector<std::pair<Vector6D, float> > disturbance_samples_;

        // cost parameters
        // const float step_cost_weight_ = 10.0;
        const float step_cost_weight_ = 3.0;
        // const float dynamics_cost_weight_ = 0.0; // test
        const float dynamics_cost_weight_ = 0.0; // original
        // const float dynamics_cost_weight_ = 1.0; // simplified
        // const float disturbance_cost_weight_ = 10000.0;
        const float disturbance_cost_weight_ = 1000.0;
        // const float disturbance_cost_weight_ = 0.0;

        // random number generator & randomness parameter
        std::mt19937_64 rng_;
        float epsilon_;

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
        std::vector< std::shared_ptr<OptimizationInterface> > one_step_capture_dynamics_optimizer_interface_vector_;
        std::vector< std::shared_ptr<OptimizationInterface> > zero_step_capture_dynamics_optimizer_interface_vector_;

        // the dynamics prediciton neural network interface
        std::vector< std::shared_ptr<NeuralNetworkInterface> > neural_network_interface_vector_;

        // the general_ik interface
        std::vector< std::shared_ptr<GeneralIKInterface> > general_ik_interface_vector_;
        std::shared_ptr<GeneralIKInterface> general_ik_interface_;

        // training sample config file export path
        std::string training_sample_config_folder_;
        std::map<ZeroStepCaptureCode, int> zero_step_capture_file_index_;
        std::map<OneStepCaptureCode, int> one_step_capture_file_index_;
        std::map<ContactTransitionCode, int> contact_transition_file_index_;

        void setupStateReachabilityIK(std::shared_ptr<ContactState> current_state, std::shared_ptr<GeneralIKInterface> general_ik_interface);

        bool kinematicsFeasibilityCheck(std::shared_ptr<ContactState> current_state, int index);
        bool dynamicsFeasibilityCheck(std::shared_ptr<ContactState> current_state, float& dynamics_cost, int index);
        bool stateFeasibilityCheck(std::shared_ptr<ContactState> current_state, float& dynamics_cost, int index);
        std::vector<bool> batchDynamicsFeasibilityCheck(std::vector< std::shared_ptr<ContactState> > state_vec, std::vector<float>& dynamics_cost_vector);
        std::vector<bool> batchStateFeasibilityCheck(std::vector< std::shared_ptr<ContactState> > current_states_vector, std::vector<float>& dynamics_cost_vector);

        float getHeuristics(std::shared_ptr<ContactState> current_state);
        float getEdgeCost(std::shared_ptr<ContactState> prev_state, std::shared_ptr<ContactState> current_state, float dynamics_cost=0.0, float disturbance_cost=0.0);

        std::vector< std::shared_ptr<ContactState> > getBranchingStates(std::shared_ptr<ContactState> current_state, std::vector<ContactManipulator>& branching_manips, std::vector< std::array<float,3> > foot_transition_model, std::vector< std::array<float,2> > hand_transition_model, bool remove_prev_move_manip=true);
        void branchingSearchTree(std::shared_ptr<ContactState> current_state, BranchingMethod branching_method);
        void branchingFootContacts(std::shared_ptr<ContactState> current_state, std::vector<ContactManipulator> branching_manips);
        void branchingHandContacts(std::shared_ptr<ContactState> current_state, std::vector<ContactManipulator> branching_manips);
        void branchingContacts(std::shared_ptr<ContactState> current_state, BranchingManipMode branching_mode=BranchingManipMode::ALL, int specified_motion_code=-1);
        bool footProjection(ContactManipulator& contact_manipulator, RPYTF& projection_pose);
        bool handProjection(ContactManipulator& contact_manipulator, Translation3D& shoulder_point, std::array<float,2>& arm_orientation, RPYTF& projection_pose);
        bool feetReprojection(std::shared_ptr<ContactState> state);
        bool footPoseSampling(ContactManipulator& contact_manipulator, RPYTF& projection_pose, double height);
        bool handPoseSampling(ContactManipulator& contact_manipulator, Translation3D& shoulder_position, std::array<float,2>& arm_orientation, RPYTF& projection_pose);

        void extendZeroStepCaptureStates(std::vector< std::shared_ptr<ContactState> >& zero_step_capture_contact_state_vector, std::shared_ptr<ContactState> current_state, ContactManipulator& move_manip, Translation3D& initial_com, Vector3D& initial_lmom);
        void extendOneStepCaptureStates(std::vector< std::shared_ptr<ContactState> >& one_step_capture_contact_state_vector, std::shared_ptr<ContactState> zero_step_capture_state, std::vector< std::shared_ptr<ContactState> >& disturbance_rejection_branching_states);

        void insertState(std::shared_ptr<ContactState> current_state, float dynamics_cost=0.0, float disturbance_cost=0.0);

        void updateExploreStatesAndOpenHeap();
        bool isReachedGoal(std::shared_ptr<ContactState> current_state);

        float fillDynamicsSequence(std::vector< std::shared_ptr<ContactState> > contact_state_path);
        float fillDynamicsSequenceSegmentBySegment(std::vector< std::shared_ptr<ContactState> > contact_state_path);
        void verifyContactSequenceDynamicsFeasibilityPrediction(std::vector< std::shared_ptr<ContactState> > contact_state_path);

        void kinematicsVerification(std::vector< std::shared_ptr<ContactState> > contact_state_path);
        void kinematicsVerification_StateOnly(std::vector< std::shared_ptr<ContactState> > contact_state_path);

        void exportContactSequenceOptimizationConfigFiles(std::shared_ptr<OptimizationInterface> optimizer_interface,
                                                          std::vector< std::shared_ptr<ContactState> > contact_sequence,
                                                          std::string optimization_config_template_path,
                                                          std::string optimization_config_output_path,
                                                          std::string objects_config_output_path,
                                                          float initial_time=0.0);
};

#endif

#ifndef OPTIMIZATIONINTERFACE_HPP
#define OPTIMIZATIONINTERFACE_HPP

#include <momentumopt/dynopt/DynamicsOptimizer.hpp>
#include <momentumopt/cntopt/ContactPlanInterface.hpp>
#include <momentumopt/kinopt/KinematicsOptimizer.hpp>
// #include <momentumopt_athena/motion_optimization/KinematicsInterfaceSl.hpp>

/**
 * This class transforms a vector of the escher_openrave_cpp ContactState
 * (a contact sequence) to a momentumopt ContactSequence for dynamics
 * optimization.
 */
// class ContactPlanFromOneStepCaptureCandidate : public momentumopt::ContactPlanInterface
// {
//     public:
//         ContactPlanFromOneStepCaptureCandidate(std::shared_ptr<ContactState> _capture_contact_state,
//                                                float _step_transition_time,  float _support_phase_time):
//         capture_contact_state_(_capture_contact_state),
//         step_transition_time_(_step_transition_time),
//         support_phase_time_(_support_phase_time),
//         timer_(0.0) {};

//         ContactPlanFromOneStepCaptureCandidate() {}
//         ~ContactPlanFromOneStepCaptureCandidate(){}

//         void customSaveToFile(){}
//         void customInitialization(){}
//         solver::ExitCode customContactsOptimization(const momentumopt::DynamicsState& ini_state, momentumopt::DynamicsSequence& dyn_seq);

//         void addContact(int eff_id, RPYTF& eff_pose, bool prev_in_contact);
//         // void addViapoint(int eff_id, RPYTF& prev_eff_pose, RPYTF& eff_pose);

//         std::shared_ptr<ContactState> capture_contact_state_;
//         momentumopt::DynamicsState dummy_ini_state_;
//         momentumopt::DynamicsSequence dummy_dyn_seq_;
//         float step_transition_time_;
//         float support_phase_time_;
//         float timer_;
//         Eigen::Matrix<int, momentumopt::Problem::n_endeffs_, 1> contacts_per_endeff_;
// };

class ContactPlanFromContactSequence : public momentumopt::ContactPlanInterface
{
    public:
        ContactPlanFromContactSequence(std::vector< std::shared_ptr<ContactState> > _input_contact_state_sequence,
                                       float _step_transition_time,  float _support_phase_time,
                                       DynOptApplication _dynamics_optimizer_application = DynOptApplication::CONTACT_TRANSITION_DYNOPT):
        input_contact_state_sequence_(_input_contact_state_sequence),
        step_transition_time_(_step_transition_time),
        support_phase_time_(_support_phase_time),
        timer_(0.0),
        dynamics_optimizer_application_(_dynamics_optimizer_application) {};

        ContactPlanFromContactSequence() {}
        ~ContactPlanFromContactSequence(){}

        void customSaveToFile(){}
        void customInitialization(){}
        solver::ExitCode customContactsOptimization(const momentumopt::DynamicsState& ini_state, momentumopt::DynamicsSequence& dyn_seq);

        void addContact(int eff_id, RPYTF& eff_pose, bool prev_in_contact);
        void addViapoint(int eff_id, RPYTF& prev_eff_pose, RPYTF& eff_pose);

        std::vector< std::shared_ptr<ContactState> > input_contact_state_sequence_;
        momentumopt::DynamicsState dummy_ini_state_;
        momentumopt::DynamicsSequence dummy_dyn_seq_;
        float step_transition_time_;
        float support_phase_time_;
        float timer_;
        Eigen::Matrix<int, momentumopt::Problem::n_endeffs_, 1> contacts_per_endeff_;

        DynOptApplication dynamics_optimizer_application_;
};

class DummyKinematicsInterface : public virtual momentumopt::KinematicsInterface
{
    public:
        DummyKinematicsInterface(){}
        ~DummyKinematicsInterface(){}

        // functions to be implemented
        void displayContactsPlan(const momentumopt::ContactSequence& contact_seq){}
        void displayPosture(const momentumopt::DynamicsState& state, double time_step){}
        double endeffectorOrientationError(int n_vars, const double* x) { return 0.0; }
        void updateJacobianAndState(Eigen::Ref<Eigen::MatrixXd> centroidal_momentum_matrix, Eigen::Ref<Eigen::MatrixXd> base_jacobian,
                        std::array<Eigen::MatrixXd, momentumopt::Problem::n_endeffs_>& endeffector_jacobian, momentumopt::DynamicsState& current_joints_state){}
        void updateInertiaAndNonlinearTerms(Eigen::Ref<Eigen::MatrixXd> inertia_matrix,
                        Eigen::Ref<Eigen::VectorXd> nonlinear_terms, const momentumopt::DynamicsState& current_joints_state){}
};

class OptimizationInterface
{
    public:
        OptimizationInterface(float _step_transition_time, float _support_phase_time,
                              std::string _cfg_file, DynOptApplication _dynamics_optimizer_application = DynOptApplication::CONTACT_TRANSITION_DYNOPT);

        // initialization functions
        // void initializeKinematicsInterface();
        void initializeKinematicsOptimizer();
        void initializeDynamicsOptimizer();

        // helper functions to load parameters from files
        void loadDynamicsOptimizerSetting(std::string _cfg_file);

        // helper functions to load parameters from planner information (Input: Planner --> Optimizer)
        void updateContactSequence(std::vector< std::shared_ptr<ContactState> > new_contact_state_sequence);
        void updateContactSequenceRelatedDynamicsOptimizerSetting();
        void updateReferenceDynamicsSequence(Translation3D com_translation, float desired_speed);
        void fillInitialRobotState();
        void fillContactSequence(momentumopt::DynamicsSequence& dynamics_sequence);

        // methods to generate kinematics sequence
        bool simplifiedKinematicsOptimization();
        bool kinematicsOptimization();

        // methods to generate dynamics sequence
        bool simplifiedDynamicsOptimization(float& dynamics_cost);
        bool dynamicsOptimization(float& dynamics_cost);
        void dynamicsSequenceConcatenation(std::vector<momentumopt::DynamicsSequence>& dynamics_sequence_vector);

        // update planner information and logging data (Output: Optimizer --> Planner)
        void updateStateCoM(std::shared_ptr<ContactState> contact_state);
        void recordEdgeDynamicsSequence(std::shared_ptr<ContactState> contact_state);
        void storeResultDigest(solver::ExitCode solver_exitcode, std::ofstream& file_stream);

        void storeDynamicsOptimizationResult(std::shared_ptr<ContactState> input_current_state, float& dynamics_cost, bool dynamically_feasible, int planning_id);
        void storeDynamicsOptimizationFeature(std::shared_ptr<ContactState> input_current_state, std::string config_file_folder, int file_index);

        void drawCoMTrajectory(std::shared_ptr<DrawingHandler> drawing_handler, Vector3D color);

        void recordDynamicsMetrics();

        // export cfg_kdopt_demo.yaml and Objects.cf for kinematics optimization
        void exportConfigFiles(std::string optimization_config_template_path, std::string optimization_config_output_path,
                               std::string objects_config_output_path,
                               std::map<ContactManipulator, RPYTF> floating_initial_contact_poses,
                               std::shared_ptr<RobotProperties> robot_properties);
        void exportOptimizationConfigFile(std::string template_path, std::string output_path,
                                          std::map<ContactManipulator, RPYTF> floating_initial_contact_poses,
                                          std::shared_ptr<RobotProperties> robot_properties);
        void exportSLObjectsFile(std::string output_path, std::shared_ptr<RobotProperties> robot_properties);

        float step_transition_time_;
        float support_phase_time_;

        // dynamics metrics
        double mean_min_force_dist_to_boundary_, mean_min_cop_dist_to_boundary_, mean_max_force_angle_, mean_max_lateral_force_, \
               mean_mean_cop_dist_to_boundary_, mean_mean_force_dist_to_boundary_, mean_mean_force_angle_, mean_mean_lateral_force_, \
               mean_force_rms_, mean_max_torque_, mean_mean_torque_, \
               mean_lmom_x_, mean_lmom_y_, mean_lmom_z_, mean_lmom_norm_, \
               mean_amom_x_, mean_amom_y_, mean_amom_z_, mean_amom_norm_, \
               mean_lmom_rate_x_, mean_lmom_rate_y_, mean_lmom_rate_z_, mean_lmom_rate_norm_, \
               mean_amom_rate_x_, mean_amom_rate_y_, mean_amom_rate_z_, mean_amom_rate_norm_;


    private:
        ContactPlanFromContactSequence          contact_sequence_interpreter_;
        DummyKinematicsInterface                dummy_kinematics_interface_;
        // momentumopt_sl::KinematicsInterfaceSl   kinematics_interface_;
        momentumopt::PlannerSetting             optimizer_setting_;
        momentumopt::DynamicsState              initial_state_;
        momentumopt::DynamicsOptimizer          dynamics_optimizer_;
        momentumopt::KinematicsOptimizer        kinematics_optimizer_;
        momentumopt::DynamicsSequence           reference_dynamics_sequence_;

        std::vector< std::shared_ptr<ContactState> > contact_state_sequence_;

        DynOptApplication dynamics_optimizer_application_;

        // std::ofstream dynopt_result_digest_;
        // std::ofstream simplified_dynopt_result_digest_;
};

#endif

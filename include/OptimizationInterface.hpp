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
class ContactPlanFromContactSequence : public momentumopt::ContactPlanInterface
{
    public:
        ContactPlanFromContactSequence(std::vector< std::shared_ptr<ContactState> > _input_contact_state_sequence, float _step_transition_time):
        input_contact_state_sequence_(_input_contact_state_sequence),
        step_transition_time_(_step_transition_time),
        timer_(0.0) {};

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
        float timer_;
        Eigen::Matrix<int, momentumopt::Problem::n_endeffs_, 1> contacts_per_endeff_;
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
        OptimizationInterface(float _step_transition_time, std::string _cfg_file);

        // initialization functions
        // void initializeKinematicsInterface();
        void initializeKinematicsOptimizer();
        void initializeDynamicsOptimizer();

        // helper functions to load parameters from files
        void loadDynamicsOptimizerSetting(std::string _cfg_file);

        // helper functions to load parameters from planner information (Input: Planner --> Optimizer)
        void updateContactSequence(std::vector< std::shared_ptr<ContactState> > new_contact_state_sequence);
        void updateContactSequenceRelatedDynamicsOptimizerSetting();
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

        void drawCoMTrajectory(std::shared_ptr<DrawingHandler> drawing_handler, Vector3D color);

    private:
        ContactPlanFromContactSequence          contact_sequence_interpreter_;
        DummyKinematicsInterface                dummy_kinematics_interface_;
        // momentumopt_sl::KinematicsInterfaceSl   kinematics_interface_;
        momentumopt::PlannerSetting             optimizer_setting_;
        momentumopt::DynamicsState              initial_state_;
        momentumopt::DynamicsOptimizer          dynamics_optimizer_;
        momentumopt::KinematicsOptimizer        kinematics_optimizer_;
        momentumopt::DynamicsSequence           reference_dynamics_sequence_;

        const float step_transition_time_;
        std::vector< std::shared_ptr<ContactState> > contact_state_sequence_;

        // std::ofstream dynopt_result_digest_;
        // std::ofstream simplified_dynopt_result_digest_;
};

#endif

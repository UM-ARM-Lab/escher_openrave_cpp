#ifndef DYNOPTINTERFACE_HPP
#define DYNOPTINTERFACE_HPP

#include <momentumopt/dynopt/DynamicsOptimizer.hpp>
#include <momentumopt/cntopt/ContactPlanInterface.hpp>


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

        void addContact(int eff_id, RPYTF& eff_pose);

        std::vector< std::shared_ptr<ContactState> > input_contact_state_sequence_;
        momentumopt::DynamicsState dummy_ini_state_;
        momentumopt::DynamicsSequence dummy_dyn_seq_;
        float step_transition_time_;
        float timer_;
        Eigen::Matrix<double, momentumopt::Problem::n_endeffs_, 1> contacts_per_endeff_;
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

class DynOptInterface
{
    public:
        DynOptInterface(float _step_transition_time, std::string cfg_file):
        step_transition_time_(_step_transition_time) {loadDynamicsOptimizerSetting(cfg_file);}

        void updateContactSequence(std::vector< std::shared_ptr<ContactState> > new_contact_state_sequence);
        void loadDynamicsOptimizerSetting(std::string cfg_file);
        void updateContactSequenceRelatedDynamicsOptimizerSetting();
        void initializeDynamicsOptimizer();
        void fillInitialRobotState();
        void fillContactSequence();
        bool dynamicsOptimization(float& dynamic_cost);
        void updateStateCoM(std::shared_ptr<ContactState> contact_state);

    private:
        ContactPlanFromContactSequence contact_sequence_interpreter_;
        DummyKinematicsInterface kinematics_interface_;
        momentumopt::PlannerSetting dynamics_optimizer_setting_;
        momentumopt::DynamicsState initial_state_;
        momentumopt::DynamicsOptimizer dynamics_optimizer_;
        momentumopt::DynamicsSequence reference_dynamics_sequence_;

        const float step_transition_time_;
        std::vector< std::shared_ptr<ContactState> > contact_state_sequence_;

};

#endif
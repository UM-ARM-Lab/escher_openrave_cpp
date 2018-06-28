#include "Utilities.hpp"

static const std::map<ContactManipulator, int> contact_manipulator_id_map_ = {{ContactManipulator::L_LEG, 1}, {ContactManipulator::R_LEG, 0},
                                                                              {ContactManipulator::L_ARM, 3}, {ContactManipulator::R_ARM, 2}};


void ContactPlanFromContactSequence::addContact(int eff_id, RPYTF& eff_pose)
{
    int cnt_id = this->contacts_per_endeff_[eff_id];
    this->contactSequence().endeffectorContacts(eff_id).push_back(momentumopt::ContactState());

    this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactActivationTime() = this->timer_;
    if(cnt_id != 0)
    {
        this->contactSequence().endeffectorContacts(eff_id)[cnt_id-1].contactDeactivationTime() = this->timer_ - this->step_transition_time_;
    }

    this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactPosition() = Eigen::Vector3d(eff_pose.x_, eff_pose.y_, eff_pose.z_);
    this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactType() = momentumopt::idToContactType(1);
    this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactOrientation() = Eigen::Quaternion<double>(Eigen::AngleAxisf(eff_pose.roll_ * DEG2RAD, Eigen::Vector3f::UnitX()) *
                                                                                                                 Eigen::AngleAxisf(eff_pose.pitch_ * DEG2RAD, Eigen::Vector3f::UnitY()) *
                                                                                                                 Eigen::AngleAxisf(eff_pose.yaw_ * DEG2RAD, Eigen::Vector3f::UnitZ()));

    if (eff_id == 0)      { this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactId() = 2*cnt_id;   }
    else if (eff_id == 1) { this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactId() = 2*cnt_id+1; }
    else if (eff_id == 2) { this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactId() = 2*cnt_id;   }
    else if (eff_id == 3) { this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactId() = 2*cnt_id+1; }

    this->contactSequence().numContacts()++;
    this->contacts_per_endeff_[eff_id]++;
}

solver::ExitCode ContactPlanFromContactSequence::customContactsOptimization(const momentumopt::DynamicsState& ini_state, momentumopt::DynamicsSequence& dyn_seq)
{
    this->contactSequence().numContacts() = 0;
    this->timer_ = 0.0;

    int state_counter = 0;
    int eff_id, cnt_id;
    RPYTF eff_pose;

    for(auto & contact_state : this->input_contact_state_sequence_)
    {
        std::shared_ptr<Stance> stance = contact_state->stances_vector_[0];

        if(state_counter == 0) // the initial state
        {
            for(auto & manip : ALL_MANIPULATORS)
            {
                if(stance->ee_contact_status_[manip]) // add the contact if it is in contact
                {
                    eff_id = contact_manipulator_id_map_.find(manip)->second;
                    eff_pose = stance->ee_contact_poses_[manip];

                    this->addContact(eff_id, eff_pose);
                }
            }
        }
        else
        {
            this->timer_ += this->step_transition_time_;

            ContactManipulator moving_manip = contact_state->prev_move_manip_;

            if(stance->ee_contact_status_[moving_manip]) // if the robot makes new contact, add it
            {
                eff_id = contact_manipulator_id_map_.find(moving_manip)->second;
                eff_pose = stance->ee_contact_poses_[moving_manip];

                this->addContact(eff_id, eff_pose);
            }
        }

        state_counter++;
    }

    // add the deactivation time for the last contact of each end-effector
    for(int eff_id = 0; eff_id < momentumopt::Problem::n_endeffs_; eff_id++)
    {
        int cnt_id = this->contacts_per_endeff_[eff_id];
        if(cnt_id > 0)
        {
            this->contactSequence().endeffectorContacts(eff_id)[cnt_id-1].contactDeactivationTime() = this->timer_ + 1.0;
        }
    }

    // update the planning variable time horizon, and active ee number

}

void DynOptInterface::updateContactSequence(std::vector< std::shared_ptr<ContactState> > new_contact_state_sequence)
{
    this->contact_state_sequence_ = new_contact_state_sequence;
    this->contact_sequence_interpreter_ = ContactPlanFromContactSequence(this->contact_state_sequence_, this->step_transition_time_);
    this->updateContactSequenceRelatedDynamicsOptimizerSetting();
}

void DynOptInterface::loadDynamicsOptimizerSetting(std::string cfg_file)
{
    dynamics_optimizer_setting_.initialize(cfg_file);
}

void DynOptInterface::updateContactSequenceRelatedDynamicsOptimizerSetting()
{
    int state_counter = 0;
    float total_time = 0.0;
    std::set<int> active_eff_set;
    double time_step = dynamics_optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_TimeStep);
    for(auto & contact_state : this->contact_state_sequence_)
    {
        std::shared_ptr<Stance> stance = contact_state->stances_vector_[0];

        if(state_counter == 0)
        {
            for(auto & manip : ALL_MANIPULATORS)
            {
                if(stance->ee_contact_status_[manip]) // add the contact if it is in contact
                {
                    active_eff_set.insert(int(contact_state->prev_move_manip_));
                }
            }
        }
        else
        {
            total_time += this->step_transition_time_;
            active_eff_set.insert(int(contact_state->prev_move_manip_));
        }

        state_counter++;
    }
    dynamics_optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumActiveEndeffectors) = active_eff_set.size();
    dynamics_optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_TimeHorizon) = (std::floor(total_time / time_step) + 1) * time_step;
}

void DynOptInterface::initializeDynamicsOptimizer()
{
    dynamics_optimizer_.initialize(dynamics_optimizer_setting_, &kinematics_interface_);
}

void DynOptInterface::fillInitialRobotState()
{
    std::shared_ptr<ContactState> initial_contact_state = this->contact_state_sequence_[0];
    std::shared_ptr<Stance> stance = initial_contact_state->stances_vector_[0];
    double robot_mass = dynamics_optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_RobotMass);

    // CoM and momenta
    initial_state_.centerOfMass() = Eigen::Vector3d(initial_contact_state->com_[0], initial_contact_state->com_[1], initial_contact_state->com_[2]);
    initial_state_.linearMomentum() = Eigen::Vector3d(robot_mass*initial_contact_state->com_dot_[0], robot_mass*initial_contact_state->com_dot_[1], robot_mass*initial_contact_state->com_dot_[2]);
    initial_state_.angularMomentum() = Eigen::Vector3d(0, 0, 0);

    // Contact poses, and forces
    int eff_id;
    RPYTF eff_pose;
    for(auto & manip : ALL_MANIPULATORS)
    {
        eff_id = contact_manipulator_id_map_.find(manip)->second;
        eff_pose = stance->ee_contact_poses_[manip];

        if(stance->ee_contact_status_[manip]) // add the contact if it is in contact
        {
            initial_state_.endeffectorActivation(eff_id) = 1;
            initial_state_.endeffectorPosition(eff_id) = Eigen::Vector3d(eff_pose.x_, eff_pose.y_, eff_pose.z_);
            initial_state_.endeffectorOrientation(eff_id) = Eigen::Quaternion<double>(Eigen::AngleAxisf(eff_pose.roll_ * DEG2RAD, Eigen::Vector3f::UnitX()) *
                                                                             Eigen::AngleAxisf(eff_pose.pitch_ * DEG2RAD, Eigen::Vector3f::UnitY()) *
                                                                             Eigen::AngleAxisf(eff_pose.yaw_ * DEG2RAD, Eigen::Vector3f::UnitZ()));
            initial_state_.endeffectorForce(eff_id) = Eigen::Vector3d(0, 0, 0.5);
        }
        else
        {
            initial_state_.endeffectorActivation(eff_id) = 0;
            // if it is okay to not specifying poses of the end-effectors not in contact
            initial_state_.endeffectorForce(eff_id) = Eigen::Vector3d(0, 0, 0);
        }
    }

    // Joint Positions
    // ignore joint positions for now. seems not used in dynopt.
    // Eigen::VectorXd joints_state = readParameter<Eigen::VectorXd>(ini_robo_cfg, "joints_state");
    // int ndofs = joints_state.size();
    // this->jointPositions().resize(ndofs+6);      this->jointPositions().setZero();
    // this->jointVelocities().resize(ndofs+6);     this->jointVelocities().setZero();
    // this->jointAccelerations().resize(ndofs+6);  this->jointAccelerations().setZero();
    // this->jointPositions().head(ndofs) = joints_state;

}

void DynOptInterface::fillContactSequence()
{
    contact_sequence_interpreter_.initialize(dynamics_optimizer_setting_, &kinematics_interface_);
    contact_sequence_interpreter_.optimize(initial_state_, dynamics_optimizer_.dynamicsSequence());
}

bool DynOptInterface::dynamicsOptimization(float& dynamic_cost)
{
    this->initializeDynamicsOptimizer();
    this->fillInitialRobotState();
    this->fillContactSequence();

    // optimize a motion
    reference_dynamics_sequence_.resize(dynamics_optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps));
    contact_sequence_interpreter_.optimize(initial_state_, dynamics_optimizer_.dynamicsSequence());
    solver::ExitCode solver_exitcode = dynamics_optimizer_.optimize(initial_state_, reference_dynamics_sequence_);

    if(int(solver_exitcode) <= 1)
    {
        return true;
    }
    else
    {
        return false;
    }

}

void DynOptInterface::updateStateCoM(std::shared_ptr<ContactState> contact_state)
{
    int final_time_id = dynamics_optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps) - 1;
    double robot_mass = dynamics_optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_RobotMass);

    Eigen::Vector3d goal_com = dynamics_optimizer_.dynamicsSequence().dynamicsState(final_time_id).centerOfMass();
    Eigen::Vector3d goal_lmom = dynamics_optimizer_.dynamicsSequence().dynamicsState(final_time_id).linearMomentum();

    contact_state->com_ = goal_com.cast<float>();
    contact_state->com_dot_ = goal_lmom.cast<float>() / robot_mass;
}

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

void ContactPlanFromContactSequence::addViapoint(int eff_id, RPYTF& prev_eff_pose, RPYTF& eff_pose)
{
    // add the viapoint
    this->viapointSequence().endeffectorViapoints(eff_id).push_back(momentumopt::ViapointState());
    int via_id = this->viapointSequence().endeffectorViapoints(eff_id).size() - 1;

    this->viapointSequence().endeffectorViapoints(eff_id)[via_id].viapointTime() = this->timer_ - 0.5 * this->step_transition_time_;
    this->viapointSequence().endeffectorViapoints(eff_id)[via_id].viapointPosition() = Eigen::Vector3d((prev_eff_pose.x_ + eff_pose.x_)/2.0,
                                                                                                       (prev_eff_pose.y_ + eff_pose.y_)/2.0,
                                                                                                       (prev_eff_pose.z_ + eff_pose.z_)/2.0 + 0.2);

    this->viapointSequence().endeffectorViapoints(eff_id)[via_id].viapointOrientation() = Eigen::Quaternion<double>(Eigen::AngleAxisf(getAngleMean(prev_eff_pose.roll_, eff_pose.roll_) * DEG2RAD, Eigen::Vector3f::UnitX()) *
                                                                                                                    Eigen::AngleAxisf(getAngleMean(prev_eff_pose.pitch_, eff_pose.pitch_) * DEG2RAD, Eigen::Vector3f::UnitY()) *
                                                                                                                    Eigen::AngleAxisf(getAngleMean(prev_eff_pose.yaw_, eff_pose.yaw_) * DEG2RAD, Eigen::Vector3f::UnitZ()));

    this->viapointSequence().numViapoints()++;
}

solver::ExitCode ContactPlanFromContactSequence::customContactsOptimization(const momentumopt::DynamicsState& ini_state, momentumopt::DynamicsSequence& dyn_seq)
{
    this->contactSequence().numContacts() = 0;
    this->viapointSequence().numViapoints() = 0;
    this->timer_ = 0.0;
    for(int eff_id = 0; eff_id < momentumopt::Problem::n_endeffs_; eff_id++)
    {
        this->contactSequence().endeffectorContacts(eff_id).clear();
        this->contacts_per_endeff_[eff_id] = 0;
        this->viapointSequence().endeffectorViapoints(eff_id).clear();
    }

    int state_counter = 0;
    int eff_id, cnt_id;
    RPYTF eff_pose, prev_eff_pose;

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
                    eff_pose = transformPoseFromOpenraveToSL(stance->ee_contact_poses_[manip]);

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
                eff_pose = transformPoseFromOpenraveToSL(stance->ee_contact_poses_[moving_manip]);

                this->addContact(eff_id, eff_pose);

                std::shared_ptr<Stance> prev_stance = contact_state->parent_->stances_vector_[0];

                if(prev_stance->ee_contact_status_[moving_manip])
                {
                    prev_eff_pose = transformPoseFromOpenraveToSL(prev_stance->ee_contact_poses_[moving_manip]);
                    this->addViapoint(eff_id, prev_eff_pose, eff_pose);
                }
            }
        }

        this->timer_ += 0.2;
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

void OptimizationInterface::updateContactSequence(std::vector< std::shared_ptr<ContactState> > new_contact_state_sequence)
{
    this->contact_state_sequence_.resize(new_contact_state_sequence.size());
    this->contact_state_sequence_ = new_contact_state_sequence;
    this->contact_sequence_interpreter_ = ContactPlanFromContactSequence(this->contact_state_sequence_, this->step_transition_time_);
    this->updateContactSequenceRelatedDynamicsOptimizerSetting();
}

void OptimizationInterface::loadDynamicsOptimizerSetting(std::string cfg_file)
{
    optimizer_setting_.initialize(cfg_file);
}

void OptimizationInterface::updateContactSequenceRelatedDynamicsOptimizerSetting()
{
    int state_counter = 0;
    float total_time = 0.0;
    std::set<int> active_eff_set;
    double time_step = optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_TimeStep);
    for(auto & contact_state : this->contact_state_sequence_)
    {
        std::shared_ptr<Stance> stance = contact_state->stances_vector_[0];

        if(state_counter == 0)
        {
            for(auto & manip : ALL_MANIPULATORS)
            {
                if(stance->ee_contact_status_[manip]) // add the contact if it is in contact
                {
                    active_eff_set.insert(int(manip));
                }
            }
        }
        else
        {
            total_time += this->step_transition_time_;
            active_eff_set.insert(int(contact_state->prev_move_manip_));
        }

        total_time += 0.2;
        state_counter++;
    }
    optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumActiveEndeffectors) = active_eff_set.size();
    optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_TimeHorizon) = std::floor(total_time / time_step) * time_step;
    optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps) = int(std::floor(total_time / time_step));
    Vector3D com_translation = this->contact_state_sequence_[state_counter-1]->com_ - this->contact_state_sequence_[0]->com_;
    optimizer_setting_.get(momentumopt::PlannerVectorParam::PlannerVectorParam_CenterOfMassMotion) = rotateVectorFromOpenraveToSL(com_translation);

    reference_dynamics_sequence_.clean();
    reference_dynamics_sequence_.resize(optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps));
}

void OptimizationInterface::initializeKinematicsInterface()
{
    kinematics_interface_ = momentumopt_sl::KinematicsInterfaceSl(optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_Frequency));
}

void OptimizationInterface::initializeKinematicsOptimizer()
{
    kinematics_optimizer_.initialize(optimizer_setting_, &kinematics_interface_);
}

void OptimizationInterface::initializeDynamicsOptimizer()
{
    dynamics_optimizer_.initialize(optimizer_setting_, &kinematics_interface_, &contact_sequence_interpreter_);
}

void OptimizationInterface::fillInitialRobotState()
{
    std::shared_ptr<ContactState> initial_contact_state = this->contact_state_sequence_[0];
    std::shared_ptr<Stance> stance = initial_contact_state->stances_vector_[0];
    double robot_mass = optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_RobotMass);

    // reset the initial state
    initial_state_ = momentumopt::DynamicsState();

    // CoM and momenta
    initial_state_.centerOfMass() = transformPositionFromOpenraveToSL(initial_contact_state->com_);
    initial_state_.linearMomentum() = robot_mass * rotateVectorFromOpenraveToSL(initial_contact_state->com_dot_);
    initial_state_.angularMomentum() = Eigen::Vector3d(0, 0, 0);

    // Contact poses, and forces
    int eff_id;
    RPYTF eff_pose;
    for(auto & manip : ALL_MANIPULATORS)
    {
        eff_id = contact_manipulator_id_map_.find(manip)->second;
        eff_pose = transformPoseFromOpenraveToSL(stance->ee_contact_poses_[manip]);

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
    int ndofs = optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumDofs);
    initial_state_.jointPositions().resize(ndofs+6);      initial_state_.jointPositions().setZero();
    initial_state_.jointVelocities().resize(ndofs+6);     initial_state_.jointVelocities().setZero();
    initial_state_.jointAccelerations().resize(ndofs+6);  initial_state_.jointAccelerations().setZero();
    // initial_state_.jointPositions().head(ndofs) = joints_state;

}

void OptimizationInterface::fillContactSequence(momentumopt::DynamicsSequence& dynamics_sequence)
{
    contact_sequence_interpreter_.initialize(optimizer_setting_, &kinematics_interface_);
    contact_sequence_interpreter_.optimize(initial_state_, dynamics_sequence);
}

bool OptimizationInterface::simplifiedKinematicsOptimization()
{
    this->initializeKinematicsOptimizer();
    this->fillInitialRobotState();
    this->fillContactSequence(kinematics_optimizer_.kinematicsSequence());

    kinematics_optimizer_.simplifiedOptimize(initial_state_, reference_dynamics_sequence_, false);
}

bool OptimizationInterface::kinematicsOptimization()
{
    this->initializeKinematicsOptimizer();
    this->fillInitialRobotState();
    this->fillContactSequence(kinematics_optimizer_.kinematicsSequence());

    kinematics_optimizer_.optimize(initial_state_, reference_dynamics_sequence_, false);
}

bool OptimizationInterface::simplifiedDynamicsOptimization(float& dynamics_cost)
{
    this->initializeDynamicsOptimizer();
    this->fillInitialRobotState();
    this->fillContactSequence(dynamics_optimizer_.dynamicsSequence());

    // std::cout << "TimeHorizon: " << optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_TimeHorizon) << std::endl;
    // std::cout << "TimeStep: " << optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_TimeStep) << std::endl;
    // std::cout << "NumTimeSteps: " << optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps) << std::endl;

    // optimize a motion
    solver::ExitCode solver_exitcode = dynamics_optimizer_.simplifiedOptimize(initial_state_, reference_dynamics_sequence_);

    // getchar();

    dynamics_cost = dynamics_optimizer_.problemInfo().get(solver::SolverDoubleParam_PrimalCost);

    // int final_time_id = optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps) - 1;
    // std::cout << "Total Timesteps: " << optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps) << std::endl;
    // std::cout << "solver code: " << int(solver_exitcode) << ", dynamics_cost: " << dynamics_optimizer_.problemInfo().get(solver::SolverDoubleParam_DualCost) << ", duality gap: " << dynamics_optimizer_.problemInfo().get(solver::SolverDoubleParam_DualityGap) << std::endl;

    // if(std::rand() % 10 < 1)
    // {
    //     storeResultDigest(solver_exitcode, simplified_dynopt_result_digest_);

    //     getchar();
    // }

    if(int(solver_exitcode) <= 1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool OptimizationInterface::dynamicsOptimization(float& dynamics_cost)
{
    this->initializeDynamicsOptimizer();
    this->fillInitialRobotState();
    this->fillContactSequence(dynamics_optimizer_.dynamicsSequence());

    // std::cout << "TimeHorizon: " << optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_TimeHorizon) << std::endl;
    // std::cout << "TimeStep: " << optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_TimeStep) << std::endl;
    // std::cout << "NumTimeSteps: " << optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps) << std::endl;

    // optimize a motion
    solver::ExitCode solver_exitcode = dynamics_optimizer_.optimize(initial_state_, reference_dynamics_sequence_);

    dynamics_cost = dynamics_optimizer_.problemInfo().get(solver::SolverDoubleParam_PrimalCost);

    int final_time_id = optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps) - 1;
    // std::cout << "Total Timesteps: " << optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps) << std::endl;
    // std::cout << "solver code: " << int(solver_exitcode) << ", dynamics_cost: " << dynamics_optimizer_.problemInfo().get(solver::SolverDoubleParam_DualCost) << ", duality gap: " << dynamics_optimizer_.problemInfo().get(solver::SolverDoubleParam_DualityGap) << std::endl;

    // storeResultDigest(solver_exitcode, dynopt_result_digest_);

    // getchar();

    if(int(solver_exitcode) <= 1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void OptimizationInterface::dynamicsSequenceConcatenation(std::vector<momentumopt::DynamicsSequence>& dynamics_sequence_vector)
{
    this->initializeDynamicsOptimizer();
    this->fillInitialRobotState();
    this->fillContactSequence(dynamics_optimizer_.dynamicsSequence());

    double current_time = 0.0;
    double time_step = optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_TimeStep);

    dynamics_optimizer_.dynamicsSequence().clean();
    int dynamics_sequence_total_size = 0;
    for(auto & dynamics_sequence : dynamics_sequence_vector)
    {
        dynamics_sequence_total_size += dynamics_sequence.size();
    }
    dynamics_optimizer_.dynamicsSequence().resize(dynamics_sequence_total_size);
    dynamics_optimizer_.dynamicsSequence().activeEndeffectorSteps() = Eigen::MatrixXi::Zero(dynamics_optimizer_.dynamicsSequence().activeEndeffectorSteps().rows(),
                                                                                  dynamics_optimizer_.dynamicsSequence().activeEndeffectorSteps().cols());

    // std::cout << "Dynamics Sequence Total Size: " << dynamics_sequence_total_size << std::endl;

    int time_id = 0;
    for(auto & dynamics_sequence : dynamics_sequence_vector)
    {
        dynamics_optimizer_.dynamicsSequence().activeEndeffectorSteps() += dynamics_sequence.activeEndeffectorSteps();
        if(time_id != 0)
        {
            for(int eff_id; eff_id < dynamics_optimizer_.dynamicsSequence().activeEndeffectorSteps().rows(); eff_id++)
            {
                if(dynamics_sequence.dynamicsState(0).endeffectorActivation(eff_id))
                {
                    dynamics_optimizer_.dynamicsSequence().activeEndeffectorSteps()(eff_id,1) -= 1;
                }
            }
        }

        // std::cout << "Dynamics Sequence Size: " << dynamics_sequence.dynamicsSequence().size() << " ";

        for(int state_id = 0; state_id < dynamics_sequence.dynamicsSequence().size(); state_id++)
        {
            if(time_id == 0 || state_id != 0)
            {
                dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id) = dynamics_sequence.dynamicsSequence()[state_id];
                dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).time() = time_step;
                // std::cout << "Time: " << time_step << " " << current_time << std::endl;
                current_time += time_step;
                time_id += 1;
            }
        }
    }

    optimizer_setting_.get(momentumopt::PlannerBoolParam::PlannerBoolParam_StoreData) = true;
    dynamics_optimizer_.storeExternalDynamicsSequenceToFile(initial_state_, reference_dynamics_sequence_);

}

void OptimizationInterface::updateStateCoM(std::shared_ptr<ContactState> contact_state)
{
    int final_time_id = optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps) - 1;
    double robot_mass = optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_RobotMass);

    contact_state->com_ = transformPositionFromSLToOpenrave(dynamics_optimizer_.dynamicsSequence().dynamicsState(final_time_id).centerOfMass());
    contact_state->com_dot_ = rotateVectorFromSLToOpenrave(dynamics_optimizer_.dynamicsSequence().dynamicsState(final_time_id).linearMomentum()) / robot_mass;

    // contact_state->com_ = goal_com.cast<float>();
    // contact_state->com_dot_ = goal_lmom.cast<float>() / robot_mass;
}

void OptimizationInterface::recordEdgeDynamicsSequence(std::shared_ptr<ContactState> contact_state)
{
    contact_state->parent_edge_dynamics_sequence_ = dynamics_optimizer_.dynamicsSequence();
}

void OptimizationInterface::storeResultDigest(solver::ExitCode solver_exitcode, std::ofstream& file_stream)
{
    // exit_code, Initial CoM, Goal CoM, Final CoM, contact num, pose, moving_eff_id, new_pose

    int final_time_id = optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps) - 1;
    double time_step = optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_TimeStep);

    Eigen::Vector3d initial_com_SL = initial_state_.centerOfMass();
    Translation3D initial_com_Openrave = transformPositionFromSLToOpenrave(initial_com_SL);
    Eigen::Vector3d com_motion_SL = optimizer_setting_.get(momentumopt::PlannerVectorParam::PlannerVectorParam_CenterOfMassMotion);
    Vector3D com_motion_Openrave = rotateVectorFromSLToOpenrave(com_motion_SL);
    Translation3D goal_com = initial_com_Openrave.cast<float>() + com_motion_Openrave;
    Translation3D final_com = transformPositionFromSLToOpenrave(dynamics_optimizer_.dynamicsSequence().dynamicsState(final_time_id).centerOfMass());
    Eigen::Vector3d initial_lmon = initial_state_.linearMomentum();

    // std::cout << "Initial CoM: " << initial_com_Openrave.transpose() << std::endl;
    // std::cout << "CoM Motion: " << com_motion_Openrave.transpose() << std::endl;
    // std::cout << "Goal CoM: " << goal_com.transpose() << std::endl;
    // std::cout << "Final CoM: " << final_com.transpose() << std::endl;

    // float com_objective = 0;

    // for(int i = 0; i < 3; i++)
    // {
    //     com_objective += 1000 * std::pow(goal_com(i) - final_com(i), 2);
    // }


    file_stream << int(solver_exitcode) << ", "
                << step_transition_time_ << ", " << time_step << ", "
                << initial_com_Openrave(0) << ", " << initial_com_Openrave(1) << ", " << initial_com_Openrave(2) << ", "
                << goal_com(0) << ", " << goal_com(1) << ", " << goal_com(2) << ", "
                << final_com(0) << ", " << final_com(1) << ", " << final_com(2) << ", "
                << initial_lmon(0) << ", " << initial_lmon(1) << ", " << initial_lmon(2) << ", "
                // << com_objective << ", "
                << dynamics_optimizer_.problemInfo().get(solver::SolverDoubleParam_DualCost) << ", "
                // << dynamics_optimizer_.problemInfo().get(solver::SolverDoubleParam_DualCost) - com_objective << ", "
                << dynamics_optimizer_.problemInfo().get(solver::SolverDoubleParam_DualityGap) << ", ";

    int eff_id, cnt_id;
    RPYTF eff_pose;
    int state_counter = 0;
    for(auto & contact_state : contact_state_sequence_)
    {
        std::shared_ptr<Stance> stance = contact_state->stances_vector_[0];

        if(state_counter == 0) // the initial state
        {
            for(auto & manip : ALL_MANIPULATORS)
            {
                if(stance->ee_contact_status_[manip])
                {
                    eff_id = contact_manipulator_id_map_.find(manip)->second;
                    eff_pose = stance->ee_contact_poses_[manip];

                    file_stream << eff_pose.x_ << ", " << eff_pose.y_ << ", " << eff_pose.z_ << ", " << eff_pose.roll_ << ", " << eff_pose.pitch_ << ", " << eff_pose.yaw_ << ", ";
                    // std::cout << "MANIP: " << manip << ", Initial Pose: " << eff_pose.x_ << " " << eff_pose.y_ << " " << eff_pose.z_ << " " << eff_pose.roll_ << " " << eff_pose.pitch_ << " " << eff_pose.yaw_ << std::endl;
                }
            }
        }
        else
        {
            ContactManipulator moving_manip = contact_state->prev_move_manip_;

            if(stance->ee_contact_status_[moving_manip])
            {
                eff_id = contact_manipulator_id_map_.find(moving_manip)->second;
                eff_pose = stance->ee_contact_poses_[moving_manip];

                file_stream << eff_pose.x_ << ", " << eff_pose.y_ << ", " << eff_pose.z_ << ", " << eff_pose.roll_ << ", " << eff_pose.pitch_ << ", " << eff_pose.yaw_  << ", ";
                // std::cout << "MANIP: " << moving_manip << ", New Pose: " << eff_pose.x_ << " " << eff_pose.y_ << " " << eff_pose.z_ << " " << eff_pose.roll_ << " " << eff_pose.pitch_ << " " << eff_pose.yaw_ << std::endl;
            }

            file_stream << int(moving_manip);
        }

        state_counter++;
    }

    file_stream << std::endl;

    // getchar();
}
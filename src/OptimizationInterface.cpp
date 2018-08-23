#include "Utilities.hpp"

static const std::map<ContactManipulator, int> contact_manipulator_id_map_ = {{ContactManipulator::L_LEG, 1}, {ContactManipulator::R_LEG, 0},
                                                                              {ContactManipulator::L_ARM, 3}, {ContactManipulator::R_ARM, 2}};


static std::array<TransformationMatrix,ContactManipulator::MANIP_NUM> ee_offset_transform_to_dynopt;


void ContactPlanFromContactSequence::addContact(int eff_id, RPYTF& eff_pose, bool prev_in_contact)
{
    int cnt_id = this->contacts_per_endeff_[eff_id];
    this->contactSequence().endeffectorContacts(eff_id).push_back(momentumopt::ContactState());

    this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactActivationTime() = this->timer_;
    if(prev_in_contact)
    {
        this->contactSequence().endeffectorContacts(eff_id)[cnt_id-1].contactDeactivationTime() = this->timer_ - this->step_transition_time_;
    }

    this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactPosition() = Eigen::Vector3d(eff_pose.x_, eff_pose.y_, eff_pose.z_);
    this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactType() = momentumopt::idToContactType(1);
    this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactOrientation() = Eigen::Quaternion<double>(Eigen::AngleAxisf(eff_pose.roll_ * DEG2RAD, Eigen::Vector3f::UnitX()) *
                                                                                                                 Eigen::AngleAxisf(eff_pose.pitch_ * DEG2RAD, Eigen::Vector3f::UnitY()) *
                                                                                                                 Eigen::AngleAxisf(eff_pose.yaw_ * DEG2RAD, Eigen::Vector3f::UnitZ()));

    this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactId() = this->contacts_per_endeff_.sum();
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
    std::array<bool,momentumopt::Problem::n_endeffs_> eff_final_in_contact;
    eff_final_in_contact.fill(false);

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
                    eff_pose = transformPoseFromOpenraveToSL(stance->ee_contact_poses_[manip], ee_offset_transform_to_dynopt[manip]);

                    bool prev_in_contact = this->contacts_per_endeff_[eff_id] != 0 && contact_state->parent_->stances_vector_[0]->ee_contact_status_[manip];

                    this->addContact(eff_id, eff_pose, prev_in_contact);
                    eff_final_in_contact[eff_id] = true;
                }
            }
        }
        else
        {
            this->timer_ += 0.4;

            ContactManipulator moving_manip = contact_state->prev_move_manip_;
            eff_id = contact_manipulator_id_map_.find(moving_manip)->second;

            if(stance->ee_contact_status_[moving_manip]) // if the robot makes new contact, add it
            {
                this->timer_ += this->step_transition_time_;

                eff_pose = transformPoseFromOpenraveToSL(stance->ee_contact_poses_[moving_manip], ee_offset_transform_to_dynopt[moving_manip]);

                bool prev_in_contact = this->contacts_per_endeff_[eff_id] != 0 && contact_state->parent_->stances_vector_[0]->ee_contact_status_[moving_manip];

                this->addContact(eff_id, eff_pose, prev_in_contact);
                eff_final_in_contact[eff_id] = true;

                std::shared_ptr<Stance> prev_stance = contact_state->parent_->stances_vector_[0];

                if(prev_stance->ee_contact_status_[moving_manip])
                {
                    prev_eff_pose = transformPoseFromOpenraveToSL(prev_stance->ee_contact_poses_[moving_manip], ee_offset_transform_to_dynopt[moving_manip]);
                    // this->addViapoint(eff_id, prev_eff_pose, eff_pose);
                }
            }
            else // the robot is breaking contact
            {
                cnt_id = this->contacts_per_endeff_[eff_id] - 1;
                this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactDeactivationTime() = this->timer_;
                this->timer_ += this->step_transition_time_;
                eff_final_in_contact[eff_id] = false;
            }

            // std::cout << "$$$$$$$$$$$$$" << std::endl;
            // for(auto & manip : ALL_MANIPULATORS)
            // {
            //     std::cout << contact_state->parent_->stances_vector_[0]->ee_contact_status_[manip] << " ";
            // }
            // std::cout << std::endl;
            // std::cout << "prev_move_manip: " << moving_manip << std::endl;
            // for(auto & manip : ALL_MANIPULATORS)
            // {
            //     std::cout << stance->ee_contact_status_[manip] << " ";
            // }
            // std::cout << std::endl;
        }

        // std::cout << "+++++++++++++++" << std::endl;
        // for(int eff_id = 0; eff_id < momentumopt::Problem::n_endeffs_; eff_id++)
        // {
        //     for(int cnt_id = 0; cnt_id < this->contacts_per_endeff_[eff_id]; cnt_id++)
        //     {
        //         std::cout << "eff_id: " << eff_id << ", cnt_id: " << cnt_id << "(" << this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactActivationTime()
        //         << "," << this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactDeactivationTime() << ")" << std::endl;
        //     }
        // }
        // std::cout << "==============" << std::endl;
        // getchar();

        state_counter++;
    }

    // add the deactivation time for the last contact of each end-effector
    for(int eff_id = 0; eff_id < momentumopt::Problem::n_endeffs_; eff_id++)
    {
        if(eff_final_in_contact[eff_id])
        {
            cnt_id = this->contacts_per_endeff_[eff_id];
            this->contactSequence().endeffectorContacts(eff_id)[cnt_id-1].contactDeactivationTime() = this->timer_ + 1.0;
        }
        // for(int cnt_id = 0; cnt_id < this->contacts_per_endeff_[eff_id]; cnt_id++)
        // {
        //     std::cout << "eff_id: " << eff_id << ", cnt_id: " << cnt_id << "(" << this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactActivationTime()
        //     << "," << this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactDeactivationTime() << ")" << std::endl;
        // }
    }
}


OptimizationInterface::OptimizationInterface(float _step_transition_time, std::string _cfg_file):
step_transition_time_(_step_transition_time)
// kinematics_interface_(momentumopt_sl::KinematicsInterfaceSl(5.0)),
// dynopt_result_digest_("dynopt_result_digest.txt", std::ofstream::app),
// simplified_dynopt_result_digest_("simplified_dynopt_result_digest.txt",std::ofstream::app) {loadDynamicsOptimizerSetting(_cfg_file);}
// simplified_dynopt_result_digest_("simplified_dynopt_test_result_digest.txt", std::ofstream::app)
{
    loadDynamicsOptimizerSetting(_cfg_file);
    //initializeKinematicsInterface();

    TransformationMatrix lf_offset_transform;
    lf_offset_transform << 1, 0, 0, 0,
                           0, 1, 0, 0,
                           0, 0, 1, 0,
                           0, 0, 0, 1;
    ee_offset_transform_to_dynopt[ContactManipulator::L_LEG] = lf_offset_transform;

    TransformationMatrix rf_offset_transform;
    rf_offset_transform << 1, 0, 0, 0,
                           0, 1, 0, 0,
                           0, 0, 1, 0,
                           0, 0, 0, 1;
    ee_offset_transform_to_dynopt[ContactManipulator::R_LEG] = rf_offset_transform;

    TransformationMatrix lh_offset_transform;
    lh_offset_transform << 0, 0,-1, 0,
                           1, 0, 0, 0,
                           0,-1, 0, 0,
                           0, 0, 0, 1;
    ee_offset_transform_to_dynopt[ContactManipulator::L_ARM] = lh_offset_transform;

    TransformationMatrix rh_offset_transform;
    rh_offset_transform << 0, 0,-1, 0,
                          -1, 0, 0, 0,
                           0, 1, 0, 0,
                           0, 0, 0, 1;
    ee_offset_transform_to_dynopt[ContactManipulator::R_ARM] = rh_offset_transform;
}

void OptimizationInterface::updateContactSequence(std::vector< std::shared_ptr<ContactState> > new_contact_state_sequence)
{
    this->contact_state_sequence_.resize(new_contact_state_sequence.size());
    this->contact_state_sequence_ = new_contact_state_sequence;
    this->contact_sequence_interpreter_ = ContactPlanFromContactSequence(this->contact_state_sequence_, this->step_transition_time_);
    this->updateContactSequenceRelatedDynamicsOptimizerSetting();
}

void OptimizationInterface::loadDynamicsOptimizerSetting(std::string _cfg_file)
{
    optimizer_setting_.initialize(_cfg_file);
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
            total_time += 0.4;

            total_time += this->step_transition_time_;
            active_eff_set.insert(int(contact_state->prev_move_manip_));
        }

        state_counter++;
    }

    total_time += time_step;

    // optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumActiveEndeffectors) = active_eff_set.size();
    optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_TimeHorizon) = std::floor(total_time / time_step + 0.001) * time_step;
    optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps) = int(std::floor(total_time / time_step + 0.001));
    Vector3D com_translation = this->contact_state_sequence_[state_counter-1]->nominal_com_ - this->contact_state_sequence_[0]->com_;
    optimizer_setting_.get(momentumopt::PlannerVectorParam::PlannerVectorParam_CenterOfMassMotion) = rotateVectorFromOpenraveToSL(com_translation);

    reference_dynamics_sequence_.clean();
    reference_dynamics_sequence_.resize(optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps));
}

// void OptimizationInterface::initializeKinematicsInterface()
// {
//     kinematics_interface_ = momentumopt_sl::KinematicsInterfaceSl(optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_Frequency));
// }

void OptimizationInterface::initializeKinematicsOptimizer()
{
    // kinematics_optimizer_.initialize(optimizer_setting_, &kinematics_interface_);
    kinematics_optimizer_.initialize(optimizer_setting_, &dummy_kinematics_interface_);
}

void OptimizationInterface::initializeDynamicsOptimizer()
{
    // dynamics_optimizer_.initialize(optimizer_setting_, &kinematics_interface_, &contact_sequence_interpreter_);
    dynamics_optimizer_.initialize(optimizer_setting_, &dummy_kinematics_interface_, &contact_sequence_interpreter_);
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
    int num_initial_contact = 0;
    for(auto & manip : ALL_MANIPULATORS)
    {
        eff_id = contact_manipulator_id_map_.find(manip)->second;
        eff_pose = transformPoseFromOpenraveToSL(stance->ee_contact_poses_[manip], ee_offset_transform_to_dynopt[manip]);

        if(stance->ee_contact_status_[manip]) // add the contact if it is in contact
        {
            initial_state_.endeffectorActivation(eff_id) = true;
            initial_state_.endeffectorPosition(eff_id) = Eigen::Vector3d(eff_pose.x_, eff_pose.y_, eff_pose.z_);
            initial_state_.endeffectorOrientation(eff_id) = Eigen::Quaternion<double>(Eigen::AngleAxisf(eff_pose.roll_ * DEG2RAD, Eigen::Vector3f::UnitX()) *
                                                                                      Eigen::AngleAxisf(eff_pose.pitch_ * DEG2RAD, Eigen::Vector3f::UnitY()) *
                                                                                      Eigen::AngleAxisf(eff_pose.yaw_ * DEG2RAD, Eigen::Vector3f::UnitZ()));
            // initial_state_.endeffectorForce(eff_id) = Eigen::Vector3d(0, 0, 0.5);
            num_initial_contact++;
        }
        else
        {
            initial_state_.endeffectorActivation(eff_id) = false;
            // if it is okay to not specifying poses of the end-effectors not in contact
            initial_state_.endeffectorForce(eff_id) = Eigen::Vector3d(0, 0, 0);
        }
    }

    for(auto & manip : ALL_MANIPULATORS)
    {
        eff_id = contact_manipulator_id_map_.find(manip)->second;

        if(stance->ee_contact_status_[manip]) // add the contact if it is in contact
        {
            initial_state_.endeffectorForce(eff_id) = Eigen::Vector3d(0, 0, 1.0/float(num_initial_contact));
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
    // contact_sequence_interpreter_.initialize(optimizer_setting_, &kinematics_interface_);
    contact_sequence_interpreter_.initialize(optimizer_setting_, &dummy_kinematics_interface_);
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

    dynamics_optimizer_.dynamicsSequence().clean();
    int dynamics_sequence_total_size = 0;
    for(auto & dynamics_sequence : dynamics_sequence_vector)
    {
        dynamics_sequence_total_size += dynamics_sequence.size();
    }
    dynamics_optimizer_.dynamicsSequence().resize(dynamics_sequence_total_size);

    this->fillInitialRobotState();
    this->fillContactSequence(dynamics_optimizer_.dynamicsSequence());

    const int active_eff_num = optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumActiveEndeffectors);
    double current_time = 0.0;
    double time_step = optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_TimeStep);

    // std::cout << "Dynamics Sequence Total Size: " << dynamics_sequence_total_size << std::endl;

    // dynamics_optimizer_.dynamicsSequence().activeEndeffectorSteps() = Eigen::MatrixXi::Zero(dynamics_optimizer_.dynamicsSequence().activeEndeffectorSteps().rows(),
    //                                                                                         dynamics_optimizer_.dynamicsSequence().activeEndeffectorSteps().cols());

    int time_id = 0;
    for(auto & dynamics_sequence : dynamics_sequence_vector)
    {
        // dynamics_optimizer_.dynamicsSequence().activeEndeffectorSteps() += dynamics_sequence.activeEndeffectorSteps();
        // std::cout << dynamics_optimizer_.dynamicsSequence().activeEndeffectorSteps() << std::endl;
        // if(time_id != 0)
        // {
        //     for(int eff_id = 0; eff_id < dynamics_optimizer_.dynamicsSequence().activeEndeffectorSteps().rows(); eff_id++)
        //     {
        //         if(dynamics_sequence.dynamicsState(0).endeffectorActivation(eff_id))
        //         {
        //             dynamics_optimizer_.dynamicsSequence().activeEndeffectorSteps()(eff_id,1) -= 1;
        //         }
        //     }
        // }

        // std::cout << "Dynamics Sequence Size: " << dynamics_sequence.dynamicsSequence().size() << " ";

        for(int state_id = 0; state_id < dynamics_sequence.dynamicsSequence().size(); state_id++)
        {
            if(time_id == 0 || state_id != 0)
            {
                std::vector<int> local_eff_activation_id(active_eff_num);
                for (int eff_id=0; eff_id<active_eff_num; eff_id++)
                {
                    local_eff_activation_id[eff_id] = dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).endeffectorActivationId(eff_id);
                }

                dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id) = dynamics_sequence.dynamicsSequence()[state_id];

                for (int eff_id=0; eff_id<active_eff_num; eff_id++)
                {
                    dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).endeffectorActivationId(eff_id) = local_eff_activation_id[eff_id];
                }

                dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).time() = time_step;
                // std::cout << "Time: " << time_step << " " << current_time << std::endl;
                current_time += time_step;
                time_id += 1;
            }
        }
    }

    // std::cout << "store solution" << std::endl;

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

void OptimizationInterface::storeDynamicsOptimizationResult(std::shared_ptr<ContactState> input_current_state, float& dynamics_cost, bool dynamically_feasible, int planning_id)
{
    // mirror the data if the robot is using the right side of the manipulators
    std::shared_ptr<ContactState> current_state = std::make_shared<ContactState>(*input_current_state);
    std::shared_ptr<ContactState> prev_state = std::make_shared<ContactState>(*input_current_state->parent_);

    if(current_state->prev_move_manip_ == ContactManipulator::R_LEG || current_state->prev_move_manip_ == ContactManipulator::R_ARM)
    {
        TransformationMatrix reference_frame = prev_state->getFeetMeanTransform();
        current_state = current_state->getMirrorState(reference_frame);
        prev_state = prev_state->getMirrorState(reference_frame);
    }

    current_state->parent_ = prev_state;
    std::shared_ptr<Stance> prev_stance = prev_state->stances_vector_[0];
    std::shared_ptr<Stance> current_stance = current_state->stances_vector_[0];

    TransformationMatrix inv_prev_mean_feet_transform = inverseTransformationMatrix(prev_state->getFeetMeanTransform());
    RotationMatrix inv_prev_mean_feet_rotation = inv_prev_mean_feet_transform.block(0,0,3,3);

    if(current_state->prev_move_manip_ != ContactManipulator::L_LEG && current_state->prev_move_manip_ != ContactManipulator::L_ARM)
    {
        RAVELOG_ERROR("Recording a motion which moves right hand side of the robot. This should not happen.\n");
        getchar();
    }

    auto transition_code_poses_pair = current_state->getTransitionCodeAndPoses();
    ContactTransitionCode contact_transition_code = transition_code_poses_pair.first;
    std::vector<RPYTF> contact_manip_pose_vec = transition_code_poses_pair.second;

    std::vector< std::vector<RPYTF> > possible_contact_pose_representation(contact_manip_pose_vec.size());
    float angle_duplication_range = 90;


    for(unsigned int i = 0; i < contact_manip_pose_vec.size(); i++)
    {
        RPYTF transformed_pose = SE3ToXYZRPY(inv_prev_mean_feet_transform * XYZRPYToSE3(contact_manip_pose_vec[i]));
        std::array<std::vector<float>,3> possible_rpy;

        for(int j = 3; j < 6; j++)
        {
            possible_rpy[j-3].push_back(transformed_pose.getXYZRPY()[j]);
            if(transformed_pose.getXYZRPY()[j] > 180-angle_duplication_range/2.0)
            {
                possible_rpy[j-3].push_back(transformed_pose.getXYZRPY()[j]-360);
            }
            else if(transformed_pose.getXYZRPY()[j] < -180+angle_duplication_range/2.0)
            {
                possible_rpy[j-3].push_back(transformed_pose.getXYZRPY()[j]+360);
            }
        }

        for(auto & roll : possible_rpy[0])
        {
            for(auto & pitch : possible_rpy[1])
            {
                for(auto & yaw : possible_rpy[2])
                {
                    possible_contact_pose_representation[i].push_back(RPYTF(transformed_pose.x_, transformed_pose.y_, transformed_pose.z_, roll, pitch, yaw));
                }
            }
        }
    }

    std::vector< std::vector<RPYTF> > all_contact_pose_combinations;
    std::vector<RPYTF> contact_pose_combination_placeholder(possible_contact_pose_representation.size());
    getAllContactPoseCombinations(all_contact_pose_combinations, possible_contact_pose_representation, 0, contact_pose_combination_placeholder);

    if(dynamically_feasible)
    {
        std::ofstream dynopt_result_fstream("dynopt_result/dynopt_result_" + std::to_string(planning_id) + ".txt", std::ofstream::app);

        Translation3D transformed_prev_com = (inv_prev_mean_feet_transform * prev_state->com_.homogeneous()).block(0,0,3,1);
        Vector3D transformed_prev_com_dot = inv_prev_mean_feet_rotation * prev_state->com_dot_;

        Translation3D transformed_current_com = (inv_prev_mean_feet_transform * current_state->com_.homogeneous()).block(0,0,3,1);
        Vector3D transformed_current_com_dot = inv_prev_mean_feet_rotation * current_state->com_dot_;

        for(auto & contact_pose_combination : all_contact_pose_combinations)
        {
            // contact state code
            dynopt_result_fstream << int(contact_transition_code) << " ";

            // get the contact poses
            for(auto & transformed_pose : contact_pose_combination)
            {
                dynopt_result_fstream << transformed_pose.x_ << " "
                                        << transformed_pose.y_ << " "
                                        << transformed_pose.z_ << " "
                                        << transformed_pose.roll_ * DEG2RAD << " "
                                        << transformed_pose.pitch_ * DEG2RAD << " "
                                        << transformed_pose.yaw_ * DEG2RAD << " ";
            }

            dynopt_result_fstream << transformed_prev_com[0] << " " << transformed_prev_com[1] << " " << transformed_prev_com[2] << " ";
            dynopt_result_fstream << transformed_prev_com_dot[0] << " " << transformed_prev_com_dot[1] << " " << transformed_prev_com_dot[2] << " ";

            dynopt_result_fstream << transformed_current_com[0] << " " << transformed_current_com[1] << " " << transformed_current_com[2] << " ";
            dynopt_result_fstream << transformed_current_com_dot[0] << " " << transformed_current_com_dot[1] << " " << transformed_current_com_dot[2] << " ";

            dynopt_result_fstream << dynamics_cost;

            dynopt_result_fstream << std::endl;
        }

        dynopt_result_fstream.close();
    }
    else
    {
        std::ofstream dynopt_result_fstream("dynopt_result/dynopt_result_infeasible_" + std::to_string(planning_id) + ".txt", std::ofstream::app);

        Translation3D transformed_prev_com = (inv_prev_mean_feet_transform * prev_state->com_.homogeneous()).block(0,0,3,1);
        Vector3D transformed_prev_com_dot = inv_prev_mean_feet_rotation * prev_state->com_dot_;

        for(auto & contact_pose_combination : all_contact_pose_combinations)
        {
            // contact state code
            dynopt_result_fstream << int(contact_transition_code) << " ";

            // get the contact poses
            for(auto & transformed_pose : contact_pose_combination)
            {
                dynopt_result_fstream << transformed_pose.x_ << " "
                                        << transformed_pose.y_ << " "
                                        << transformed_pose.z_ << " "
                                        << transformed_pose.roll_ * DEG2RAD << " "
                                        << transformed_pose.pitch_ * DEG2RAD << " "
                                        << transformed_pose.yaw_ * DEG2RAD << " ";
            }

            dynopt_result_fstream << transformed_prev_com[0] << " " << transformed_prev_com[1] << " " << transformed_prev_com[2] << " ";
            dynopt_result_fstream << transformed_prev_com_dot[0] << " " << transformed_prev_com_dot[1] << " " << transformed_prev_com_dot[2] << " ";

            dynopt_result_fstream << std::endl;
        }

        dynopt_result_fstream.close();
    }
}
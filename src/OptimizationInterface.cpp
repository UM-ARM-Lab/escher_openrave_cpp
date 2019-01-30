#include <yaml-cpp/yaml.h>
#include <yaml-cpp/yaml_eigen.h>
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

            if(dynamics_optimizer_application_ == DynOptApplication::ZERO_STEP_CAPTURABILITY_DYNOPT)
            {
                this->timer_ += this->support_phase_time_;
            }
        }
        else
        {
            if(dynamics_optimizer_application_ == DynOptApplication::CONTACT_TRANSITION_DYNOPT)
            {
                this->timer_ += this->support_phase_time_;
            }

            // 0_6_0_4 legacy code
            // this->timer_ += 0.6;

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
                if(dynamics_optimizer_application_ == DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT)
                {
                    RAVELOG_ERROR("The dynamics optimization application is testing one step capturability, but the second state breaks contact.");
                    getchar();
                }
                cnt_id = this->contacts_per_endeff_[eff_id] - 1;
                this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactDeactivationTime() = this->timer_;
                this->timer_ += this->step_transition_time_;
                eff_final_in_contact[eff_id] = false;
            }

            if(dynamics_optimizer_application_ == DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT)
            {
                this->timer_ += this->support_phase_time_;
            }

            // 0_6_0_4 legacy code
            // this->timer_ += 0.4;

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
            this->contactSequence().endeffectorContacts(eff_id)[cnt_id-1].contactDeactivationTime() = this->timer_ + 3.0;
        }
        // for(int cnt_id = 0; cnt_id < this->contacts_per_endeff_[eff_id]; cnt_id++)
        // {
        //     std::cout << "eff_id: " << eff_id << ", cnt_id: " << cnt_id << ", (" << this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactActivationTime()
        //     << "," << this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactDeactivationTime() << "), Position: (" <<
        //     this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactPosition()[0] << ", " <<
        //     this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactPosition()[1] << ", " <<
        //     this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactPosition()[2] << ")" << std::endl;
        // }
    }

    // std::vector<std::string> eff_name = {"effcnt_rf", "effcnt_lf", "effcnt_rh", "effcnt_lh"};
    // std::ofstream eff_cnt_fstream("eff_cnt_list.txt", std::ofstream::out);
    // for(int eff_id = 0; eff_id < momentumopt::Problem::n_endeffs_; eff_id++)
    // {
    //     eff_cnt_fstream << eff_name[eff_id] << ":" << std::endl;
    //     for(int cnt_id = 0; cnt_id < this->contacts_per_endeff_[eff_id]; cnt_id++)
    //     {
    //         eff_cnt_fstream << "cnt" << cnt_id << ":" << "["
    //                         << this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactActivationTime() << ", "
    //                         << this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactDeactivationTime() << ", "
    //                         << this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactPosition()[0] << ", "
    //                         << this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactPosition()[1] << ", "
    //                         << this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactPosition()[2] << ", "
    //                         << this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactOrientation().w() << ", "
    //                         << this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactOrientation().x() << ", "
    //                         << this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactOrientation().y() << ", "
    //                         << this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactOrientation().z() << ", "
    //                         << float(this->contactSequence().endeffectorContacts(eff_id)[cnt_id].contactType()) << ", "
    //                         << -1.0
    //                         << "]" << std::endl;
    //     }
    // }
}


OptimizationInterface::OptimizationInterface(float _step_transition_time, float _support_phase_time,
                                             std::string _cfg_file, DynOptApplication _dynamics_optimizer_application):
step_transition_time_(_step_transition_time),
support_phase_time_(_support_phase_time),
dynamics_optimizer_application_(_dynamics_optimizer_application)
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
    this->contact_sequence_interpreter_ = ContactPlanFromContactSequence(this->contact_state_sequence_,
                                                                         this->step_transition_time_,
                                                                         this->support_phase_time_,
                                                                         dynamics_optimizer_application_);
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

            if(dynamics_optimizer_application_ == DynOptApplication::ZERO_STEP_CAPTURABILITY_DYNOPT)
            {
                total_time += this->support_phase_time_;
            }
        }
        else
        {
            // the result is the same, but we keep them separated for highlighting the underlying motion sequence
            if(dynamics_optimizer_application_ == DynOptApplication::CONTACT_TRANSITION_DYNOPT)
            {
                total_time += this->support_phase_time_;
                total_time += this->step_transition_time_;
            }
            else if(dynamics_optimizer_application_ == DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT)
            {
                total_time += this->step_transition_time_;
                total_time += this->support_phase_time_;
            }

            active_eff_set.insert(int(contact_state->prev_move_manip_));


            // 0_6_0_4 legacy code
            // total_time += (time_step*3);

            // total_time += this->step_transition_time_;
            // active_eff_set.insert(int(contact_state->prev_move_manip_));

            // total_time += (time_step*2);
        }

        state_counter++;
    }

    // total_time += time_step;

    // optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumActiveEndeffectors) = active_eff_set.size();
    optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_TimeHorizon) = std::floor(total_time / time_step + 0.001) * time_step;
    optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps) = int(std::floor(total_time / time_step + 0.001));

    Vector3D com_translation = this->contact_state_sequence_[state_counter-1]->nominal_com_ - this->contact_state_sequence_[0]->com_;
    optimizer_setting_.get(momentumopt::PlannerVectorParam::PlannerVectorParam_CenterOfMassMotion) = rotateVectorFromOpenraveToSL(com_translation);
    reference_dynamics_sequence_.clean();
    reference_dynamics_sequence_.resize(optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps));
}

void OptimizationInterface::updateReferenceDynamicsSequence(Translation3D com_translation, float desired_speed)
{
    reference_dynamics_sequence_.clean();
    reference_dynamics_sequence_.resize(optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps));

    double robot_mass = optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_RobotMass);

    if(com_translation.norm() != 0) // if norm == 0, the reference momentum is 0
    {
        Eigen::Vector3d desired_linear_momentum = rotateVectorFromOpenraveToSL(com_translation) * desired_speed * robot_mass / com_translation.norm();
        for(int time_id = 0; time_id < optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps); time_id++)
        {
            reference_dynamics_sequence_.dynamicsState(time_id).linearMomentum() = desired_linear_momentum;
        }
    }

    optimizer_setting_.get(momentumopt::PlannerVectorParam::PlannerVectorParam_CenterOfMassMotion) = rotateVectorFromOpenraveToSL(com_translation);
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
        // recordDynamicsMetrics();

        if(dynamics_optimizer_application_ == DynOptApplication::ZERO_STEP_CAPTURABILITY_DYNOPT ||
           dynamics_optimizer_application_ == DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT)
        {
            for(int time_id = 0; time_id < optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps); time_id++)
            {
                if(dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).angularMomentum().norm() > 0.01)
                {
                    return false;
                }
            }
        }

        return true;
    }
    else
    {
        return false;
    }
}

void OptimizationInterface::recordDynamicsMetrics()
{
    const int num_timesteps = optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps);
    double mu = optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_FrictionCoefficient);

    Eigen::VectorXd min_cop_dist_to_boundary_vec(num_timesteps);
    Eigen::VectorXd min_force_dist_to_boundary_vec(num_timesteps);
    Eigen::VectorXd max_force_angle_vec(num_timesteps);
    Eigen::VectorXd max_lateral_force_vec(num_timesteps);
    Eigen::VectorXd max_torque_vec(num_timesteps);
    Eigen::VectorXd force_rms_vec(num_timesteps);

    Eigen::VectorXd mean_cop_dist_to_boundary_vec(num_timesteps);
    Eigen::VectorXd mean_force_dist_to_boundary_vec(num_timesteps);
    Eigen::VectorXd mean_force_angle_vec(num_timesteps);
    Eigen::VectorXd mean_lateral_force_vec(num_timesteps);
    Eigen::VectorXd mean_torque_vec(num_timesteps);

    Eigen::MatrixXd lmom_vec(3, num_timesteps);
    Eigen::MatrixXd amom_vec(3, num_timesteps);
    Eigen::MatrixXd lmom_rate_vec(3, num_timesteps);
    Eigen::MatrixXd amom_rate_vec(3, num_timesteps);

    Eigen::VectorXd lmom_norm_vec(num_timesteps);
    Eigen::VectorXd amom_norm_vec(num_timesteps);
    Eigen::VectorXd lmom_rate_norm_vec(num_timesteps);
    Eigen::VectorXd amom_rate_norm_vec(num_timesteps);

    // friction cone distance & cop distance
    for(int time_id = 0; time_id < num_timesteps; time_id++)
    {
        double min_cop_dist_to_boundary = 9999.0;
        double min_force_dist_to_boundary = 9999.0;
        double max_lateral_force = -9999.0;
        double max_force_angle = -9999.0;
        double max_torque = -9999.0;

        double mean_cop_dist_to_boundary = 0;
        double mean_force_dist_to_boundary = 0;
        double mean_lateral_force = 0;
        double mean_force_angle = 0;
        double mean_torque = 0;

        double force_rms = 0;

        int num_contacts = 0;
        int num_solid_contacts = 0;

        for (int eff_id = 0; eff_id < optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumActiveEndeffectors); eff_id++)
        {
            if (dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).endeffectorContactType(eff_id) != momentumopt::ContactType::FullContact)
            {
                if (dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).endeffectorActivation(eff_id))
                {
                    num_contacts++;
                    // penalty on lateral forces of each contact
                    Eigen::Matrix3d eff_rotation = dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).endeffectorOrientation(eff_id).toRotationMatrix().transpose();
                    Eigen::Vector3d frc_local = eff_rotation * dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).endeffectorForce(eff_id);

                    force_rms += pow(dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).endeffectorForce(eff_id).norm(), 2);

                    if(frc_local[2] > 0.01)
                    {
                        num_solid_contacts += 1;

                        double lateral_force_norm = sqrt(pow(frc_local[0],2) + pow(frc_local[1],2));
                        double force_dist_to_boundary = fabs(mu*lateral_force_norm - frc_local[2]) / sqrt(pow(mu,2)+1);

                        double force_angle = atan2(lateral_force_norm, frc_local[2]);

                        if(force_dist_to_boundary < min_force_dist_to_boundary)
                        {
                            min_force_dist_to_boundary = force_dist_to_boundary;
                        }

                        if(force_angle > max_force_angle)
                        {
                            max_force_angle = force_angle;
                        }

                        if(lateral_force_norm > max_lateral_force)
                        {
                            max_lateral_force = lateral_force_norm;
                        }

                        mean_force_dist_to_boundary += force_dist_to_boundary;
                        mean_lateral_force += lateral_force_norm;
                        mean_force_angle += force_angle;
                    }

                    // penalty on cop position of each contact
                    double cop_x_lower_bound = optimizer_setting_.get(momentumopt::PlannerArrayParam::PlannerArrayParam_CenterOfPressureRange)[eff_id][0];
                    double cop_x_upper_bound = optimizer_setting_.get(momentumopt::PlannerArrayParam::PlannerArrayParam_CenterOfPressureRange)[eff_id][1];
                    double cop_y_lower_bound = optimizer_setting_.get(momentumopt::PlannerArrayParam::PlannerArrayParam_CenterOfPressureRange)[eff_id][2];
                    double cop_y_upper_bound = optimizer_setting_.get(momentumopt::PlannerArrayParam::PlannerArrayParam_CenterOfPressureRange)[eff_id][3];

                    double cop_x_lower_dist = fabs(dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).endeffectorCoP(eff_id)[0] - cop_x_lower_bound);
                    double cop_x_upper_dist = fabs(dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).endeffectorCoP(eff_id)[0] - cop_x_upper_bound);
                    double cop_y_lower_dist = fabs(dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).endeffectorCoP(eff_id)[1] - cop_y_lower_bound);
                    double cop_y_upper_dist = fabs(dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).endeffectorCoP(eff_id)[1] - cop_y_upper_bound);

                    double cop_dist_to_boundary = std::min(std::min(cop_x_lower_dist, cop_x_upper_dist), std::min(cop_y_lower_dist, cop_y_upper_dist));

                    if(cop_dist_to_boundary < min_cop_dist_to_boundary)
                    {
                        min_cop_dist_to_boundary = cop_dist_to_boundary;
                    }

                    double torque = fabs(dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).endeffectorTorque(eff_id)[2]);

                    if(torque > max_torque)
                    {
                        max_torque = torque;
                    }

                    mean_cop_dist_to_boundary += cop_dist_to_boundary;
                    mean_torque += torque;
                }
            }
        }

        if(min_cop_dist_to_boundary == 9999.0 || min_force_dist_to_boundary == 9999.0 || max_force_angle == -9999.0 || max_lateral_force == -9999.0)
        {
            std::cout << "Bug: Wrong cop and local forces." << std::endl;
            getchar();
        }

        force_rms = sqrt(force_rms/num_contacts);

        mean_cop_dist_to_boundary /= num_contacts;
        mean_torque /= num_contacts;
        mean_force_dist_to_boundary /= num_solid_contacts;
        mean_lateral_force /= num_solid_contacts;
        mean_force_angle /= num_solid_contacts;

        min_cop_dist_to_boundary_vec[time_id] = min_cop_dist_to_boundary;
        min_force_dist_to_boundary_vec[time_id] = min_force_dist_to_boundary;
        max_force_angle_vec[time_id] = max_force_angle;
        max_lateral_force_vec[time_id] = max_lateral_force;

        mean_cop_dist_to_boundary_vec[time_id] = mean_cop_dist_to_boundary;
        mean_force_dist_to_boundary_vec[time_id] = mean_force_dist_to_boundary;
        mean_force_angle_vec[time_id] = mean_force_angle;
        mean_lateral_force_vec[time_id] = mean_lateral_force;
        mean_torque_vec[time_id] = mean_torque;

        force_rms_vec[time_id] = force_rms;
        max_torque_vec[time_id] = max_torque;

        lmom_vec.block(0,time_id,3,1) = dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).linearMomentum();
        amom_vec.block(0,time_id,3,1) = dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).angularMomentum();
        lmom_rate_vec.block(0,time_id,3,1) = dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).linearMomentumRate();
        amom_rate_vec.block(0,time_id,3,1) = dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).angularMomentumRate();

        lmom_norm_vec[time_id] = dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).linearMomentum().norm();
        amom_norm_vec[time_id] = dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).angularMomentum().norm();
        lmom_rate_norm_vec[time_id] = dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).linearMomentumRate().norm();
        amom_rate_norm_vec[time_id] = dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).angularMomentumRate().norm();
    }

    mean_min_cop_dist_to_boundary_ = min_cop_dist_to_boundary_vec.mean();
    mean_min_force_dist_to_boundary_ = min_force_dist_to_boundary_vec.mean();
    mean_max_force_angle_ = max_force_angle_vec.mean();
    mean_max_lateral_force_ = max_lateral_force_vec.mean();

    mean_mean_cop_dist_to_boundary_ = mean_cop_dist_to_boundary_vec.mean();
    mean_mean_force_dist_to_boundary_ = mean_force_dist_to_boundary_vec.mean();
    mean_mean_force_angle_ = mean_force_angle_vec.mean();
    mean_mean_lateral_force_ = mean_lateral_force_vec.mean();

    mean_lmom_x_ = lmom_vec.block(0,0,1,num_timesteps).cwiseAbs().mean();
    mean_lmom_y_ = lmom_vec.block(1,0,1,num_timesteps).cwiseAbs().mean();
    mean_lmom_z_ = lmom_vec.block(2,0,1,num_timesteps).cwiseAbs().mean();
    mean_lmom_norm_ = lmom_norm_vec.mean();

    mean_amom_x_ = amom_vec.block(0,0,1,num_timesteps).cwiseAbs().mean();
    mean_amom_y_ = amom_vec.block(1,0,1,num_timesteps).cwiseAbs().mean();
    mean_amom_z_ = amom_vec.block(2,0,1,num_timesteps).cwiseAbs().mean();
    mean_amom_norm_ = amom_norm_vec.mean();

    mean_lmom_rate_x_ = lmom_rate_vec.block(0,0,1,num_timesteps).cwiseAbs().mean();
    mean_lmom_rate_y_ = lmom_rate_vec.block(1,0,1,num_timesteps).cwiseAbs().mean();
    mean_lmom_rate_z_ = lmom_rate_vec.block(2,0,1,num_timesteps).cwiseAbs().mean();
    mean_lmom_rate_norm_ = lmom_rate_norm_vec.mean();

    mean_amom_rate_x_ = amom_rate_vec.block(0,0,1,num_timesteps).cwiseAbs().mean();
    mean_amom_rate_y_ = amom_rate_vec.block(1,0,1,num_timesteps).cwiseAbs().mean();
    mean_amom_rate_z_ = amom_rate_vec.block(2,0,1,num_timesteps).cwiseAbs().mean();
    mean_amom_rate_norm_ = amom_rate_norm_vec.mean();

    mean_force_rms_ = force_rms_vec.mean();
    mean_max_torque_ = max_torque_vec.mean();
    mean_mean_torque_ = mean_torque_vec.mean();
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
    std::shared_ptr<ContactState> current_state = input_current_state->getStandardInputState(dynamics_optimizer_application_);
    std::shared_ptr<ContactState> prev_state = current_state->parent_;

    int motion_code;
    std::vector<RPYTF> contact_manip_pose_vec;
    std::string feasible_record_file_name, infeasible_record_file_name;

    if(dynamics_optimizer_application_ == DynOptApplication::CONTACT_TRANSITION_DYNOPT)
    {
        auto transition_code_poses_pair = current_state->getTransitionCodeAndPoses();
        motion_code = int(transition_code_poses_pair.first);
        contact_manip_pose_vec = transition_code_poses_pair.second;
        feasible_record_file_name = "../data/dynopt_result/contact_transition_dynopt_result_" + std::to_string(planning_id) + ".txt";
        infeasible_record_file_name = "../data/dynopt_result/contact_transition_dynopt_result_infeasible_" + std::to_string(planning_id) + ".txt";
    }
    else if(dynamics_optimizer_application_ == DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT)
    {
        auto capturability_code_poses_pair = current_state->getOneStepCapturabilityCodeAndPoses();
        motion_code = int(capturability_code_poses_pair.first);
        contact_manip_pose_vec = capturability_code_poses_pair.second;
        feasible_record_file_name = "../data/dynopt_result/one_step_capture_dynopt_result_" + std::to_string(planning_id) + ".txt";
        infeasible_record_file_name = "../data/dynopt_result/one_step_capture_dynopt_result_infeasible_" + std::to_string(planning_id) + ".txt";
    }
    else if(dynamics_optimizer_application_ == DynOptApplication::ZERO_STEP_CAPTURABILITY_DYNOPT)
    {
        auto capturability_code_poses_pair = current_state->getZeroStepCapturabilityCodeAndPoses();
        motion_code = int(capturability_code_poses_pair.first);
        contact_manip_pose_vec = capturability_code_poses_pair.second;
        feasible_record_file_name = "../data/dynopt_result/zero_step_capture_dynopt_result_" + std::to_string(planning_id) + ".txt";
        infeasible_record_file_name = "../data/dynopt_result/zero_step_capture_dynopt_result_infeasible_" + std::to_string(planning_id) + ".txt";
    }

    unsigned int contact_pose_num = contact_manip_pose_vec.size();
    std::vector< std::vector<RPYTF> > possible_contact_pose_representation(contact_pose_num);
    float angle_duplication_range = 90;

    for(unsigned int i = 0; i < contact_pose_num; i++)
    {
        RPYTF contact_pose = contact_manip_pose_vec[i];
        std::array<std::vector<float>,3> possible_rpy;

        for(int j = 3; j < 6; j++)
        {
            possible_rpy[j-3].push_back(contact_pose.getXYZRPY()[j]);
            if(contact_pose.getXYZRPY()[j] > 180-angle_duplication_range/2.0)
            {
                possible_rpy[j-3].push_back(contact_pose.getXYZRPY()[j]-360);
            }
            else if(contact_pose.getXYZRPY()[j] < -180+angle_duplication_range/2.0)
            {
                possible_rpy[j-3].push_back(contact_pose.getXYZRPY()[j]+360);
            }
        }

        for(auto & roll : possible_rpy[0])
        {
            for(auto & pitch : possible_rpy[1])
            {
                for(auto & yaw : possible_rpy[2])
                {
                    possible_contact_pose_representation[i].push_back(RPYTF(contact_pose.x_, contact_pose.y_, contact_pose.z_, roll, pitch, yaw));
                }
            }
        }
    }

    std::vector< std::vector<RPYTF> > all_contact_pose_combinations;
    std::vector<RPYTF> contact_pose_combination_placeholder(contact_pose_num);
    getAllContactPoseCombinations(all_contact_pose_combinations, possible_contact_pose_representation, 0, contact_pose_combination_placeholder);

    if(dynamically_feasible)
    {
        std::ofstream dynopt_result_fstream(feasible_record_file_name, std::ofstream::app);

        double motion_duration = 0;
        for(int i = 0; i < optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps); i++)
        {
            motion_duration += dynamics_optimizer_.dynamicsSequence().dynamicsState(i).time();
        }

        for(auto & contact_pose_combination : all_contact_pose_combinations)
        {
            // contact state code
            dynopt_result_fstream << motion_code << " ";

            // get the contact poses
            for(auto & contact_pose : contact_pose_combination)
            {
                dynopt_result_fstream << contact_pose.x_ << " "
                                      << contact_pose.y_ << " "
                                      << contact_pose.z_ << " "
                                      << contact_pose.roll_ * DEG2RAD << " "
                                      << contact_pose.pitch_ * DEG2RAD << " "
                                      << contact_pose.yaw_ * DEG2RAD << " ";
            }

            if(dynamics_optimizer_application_ == DynOptApplication::CONTACT_TRANSITION_DYNOPT || dynamics_optimizer_application_ == DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT)
            {
                dynopt_result_fstream << prev_state->com_[0] << " " << prev_state->com_[1] << " " << prev_state->com_[2] << " ";
                dynopt_result_fstream << prev_state->com_dot_[0] << " " << prev_state->com_dot_[1] << " " << prev_state->com_dot_[2] << " ";
            }

            // dynopt_result_fstream << reference_dynamics_sequence_.dynamicsState(0).linearMomentum()[0] << " "
            //                       << reference_dynamics_sequence_.dynamicsState(0).linearMomentum()[1] << " "
            //                       << reference_dynamics_sequence_.dynamicsState(0).linearMomentum()[2] << " ";

            dynopt_result_fstream << current_state->com_[0] << " " << current_state->com_[1] << " " << current_state->com_[2] << " ";
            dynopt_result_fstream << current_state->com_dot_[0] << " " << current_state->com_dot_[1] << " " << current_state->com_dot_[2] << " ";

            dynopt_result_fstream << dynamics_cost << " ";

            dynopt_result_fstream << motion_duration;

            dynopt_result_fstream << std::endl;
        }

        dynopt_result_fstream.close();
    }
    else
    {
        std::ofstream dynopt_result_fstream(infeasible_record_file_name, std::ofstream::app);

        for(auto & contact_pose_combination : all_contact_pose_combinations)
        {
            // contact state code
            dynopt_result_fstream << motion_code << " ";

            // get the contact poses
            for(auto & contact_pose : contact_pose_combination)
            {
                dynopt_result_fstream << contact_pose.x_ << " "
                                      << contact_pose.y_ << " "
                                      << contact_pose.z_ << " "
                                      << contact_pose.roll_ * DEG2RAD << " "
                                      << contact_pose.pitch_ * DEG2RAD << " "
                                      << contact_pose.yaw_ * DEG2RAD << " ";
            }

            if(dynamics_optimizer_application_ == DynOptApplication::CONTACT_TRANSITION_DYNOPT || dynamics_optimizer_application_ == DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT)
            {
                dynopt_result_fstream << prev_state->com_[0] << " " << prev_state->com_[1] << " " << prev_state->com_[2] << " ";
                dynopt_result_fstream << prev_state->com_dot_[0] << " " << prev_state->com_dot_[1] << " " << prev_state->com_dot_[2] << " ";
            }
            else if(dynamics_optimizer_application_ == DynOptApplication::ZERO_STEP_CAPTURABILITY_DYNOPT)
            {
                dynopt_result_fstream << current_state->com_[0] << " " << current_state->com_[1] << " " << current_state->com_[2] << " ";
                dynopt_result_fstream << current_state->com_dot_[0] << " " << current_state->com_dot_[1] << " " << current_state->com_dot_[2] << " ";
            }

            dynopt_result_fstream << std::endl;
        }

        dynopt_result_fstream.close();
    }
}

void OptimizationInterface::drawCoMTrajectory(std::shared_ptr<DrawingHandler> drawing_handler, Vector3D color)
{
    dynamics_optimizer_.dynamicsSequence();

    int final_time_id = optimizer_setting_.get(momentumopt::PlannerIntParam::PlannerIntParam_NumTimesteps) - 1;
    double robot_mass = optimizer_setting_.get(momentumopt::PlannerDoubleParam::PlannerDoubleParam_RobotMass);

    Translation3D prev_com, com;
    Vector3D prev_com_dot, com_dot;

    for(int time_id = 0; time_id <= final_time_id; time_id++)
    {
        com = transformPositionFromSLToOpenrave(dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).centerOfMass());
        com_dot = rotateVectorFromSLToOpenrave(dynamics_optimizer_.dynamicsSequence().dynamicsState(time_id).linearMomentum()) / robot_mass;
        drawing_handler->DrawLocation(com, color);
        drawing_handler->DrawArrow(com, com_dot, color);

        if(time_id != 0)
        {
            // drawing_handler->DrawLineSegment(prev_com, com, {color[0],color[1],color[2],1});
        }

        prev_com = com;
        prev_com_dot = com_dot;
    }
}

void OptimizationInterface::exportOptimizationConfigFile(std::string template_path, std::string output_path)
{
    YAML::Node optimization_cfg = YAML::LoadFile(template_path.c_str());

    // modify the contact poses
    std::vector<std::string> eff_name_vec = {"effcnt_rf", "effcnt_lf", "effcnt_rh", "effcnt_lh"};
    for(int eff_id = 0; eff_id < momentumopt::Problem::n_endeffs_; eff_id++)
    {
        optimization_cfg["contact_plan"]["num_contacts"][eff_id] = contact_sequence_interpreter_.contacts_per_endeff_[eff_id];
        for(int cnt_id = 0; cnt_id < contact_sequence_interpreter_.contacts_per_endeff_[eff_id]; cnt_id++)
        {
            std::string eff_name = eff_name_vec[eff_id];
            std::string cnt_name = "cnt"+std::to_string(cnt_id);
            YAML::Node cnt_node = optimization_cfg["contact_plan"][eff_name][cnt_name];
            auto cnt_obj = contact_sequence_interpreter_.contactSequence().endeffectorContacts(eff_id)[cnt_id];

            cnt_node[0] = cnt_obj.contactActivationTime();
            cnt_node[1] = cnt_obj.contactDeactivationTime();
            cnt_node[2] = cnt_obj.contactPosition()[0];
            cnt_node[3] = cnt_obj.contactPosition()[1];
            cnt_node[4] = cnt_obj.contactPosition()[2];
            cnt_node[5] = cnt_obj.contactOrientation().w();
            cnt_node[6] = cnt_obj.contactOrientation().x();
            cnt_node[7] = cnt_obj.contactOrientation().y();
            cnt_node[8] = cnt_obj.contactOrientation().z();
            cnt_node[9] = float(cnt_obj.contactType());
            cnt_node[10] = -1.0;
        }
    }

    // modify the initial com, and com dot
    for(int i = 0; i < 3; i++)
    {
        optimization_cfg["initial_robot_state"]["com"][i] = initial_state_.centerOfMass()[i];
        optimization_cfg["initial_robot_state"]["lmom"][i] = initial_state_.linearMomentum()[i];
        optimization_cfg["initial_robot_state"]["amom"][i] = initial_state_.angularMomentum()[i];
    }

    std::ofstream fout(output_path.c_str());
    fout << optimization_cfg;
    fout.close();
}

// construct the SL Objects with the contact poses
void OptimizationInterface::exportSLObjectsFile(std::string output_path, std::shared_ptr<RobotProperties> robot_properties)
{
    double thickness = 0.001;
    double inflate_scale = 1.2;

    std::ofstream dump_object_stream(output_path, std::ofstream::out);

    // wirte the comments
    dump_object_stream << "/* this file contains a list of object specifications that can be added to the " << std::endl;
    dump_object_stream << "graphics scence. Each object must contain blank or tab delimited parameters in " << std::endl;
    dump_object_stream << "the following sequence: " << std::endl;
    dump_object_stream << std::endl;
    dump_object_stream << "name  (max 20 characters)" << std::endl;
    dump_object_stream << "    object type (e.g., CUBE=1, SPHERE=2, see SL_openGL.h)" << std::endl;
    dump_object_stream << "    rgb color values (3 values)" << std::endl;
    dump_object_stream << "    pos_x pos_y pos_z" << std::endl;
    dump_object_stream << "    rot_x rot_y rot_z" << std::endl;
    dump_object_stream << "    scale_x scale_y scale_z" << std::endl;
    dump_object_stream << "    contact_model  (e.g., DAMPED_SPRING_FRICTION=1)" << std::endl;
    dump_object_stream << "    appropriate number of object parameters (N_OBJ_PARMS: see SL_openGL.h)" << std::endl;
    dump_object_stream << "    appropriate number of contact parameters (N_OBJ_PARMS: see SL_openGL.h)" << std::endl;
    dump_object_stream << "*/" << std::endl;
    dump_object_stream << std::endl;


    std::vector<std::string> eff_name_vec = {"effcnt_rf", "effcnt_lf", "effcnt_rh", "effcnt_lh"};
    std::vector< std::array<float,3> > eff_color_vec = {{0,1,0}, {1,0,0} , {1,1,0}, {0,0,1}};
    for(int eff_id = 0; eff_id < momentumopt::Problem::n_endeffs_; eff_id++)
    {
        double contact_h, contact_w;

        if(eff_id < 2)
        {
            contact_h = robot_properties->foot_h_;
            contact_w = robot_properties->foot_w_;
        }
        else
        {
            contact_h = robot_properties->hand_h_;
            contact_w = robot_properties->hand_w_;
        }

        for(int cnt_id = 0; cnt_id < contact_sequence_interpreter_.contacts_per_endeff_[eff_id]; cnt_id++)
        {
            auto cnt_obj = contact_sequence_interpreter_.contactSequence().endeffectorContacts(eff_id)[cnt_id];
            auto cnt_rpy = cnt_obj.contactOrientation().toRotationMatrix().eulerAngles(0, 1, 2);

            dump_object_stream << "box_" << eff_name_vec[eff_id] << "_cnt_" << cnt_id << " 1          /* name and ID of object type */ " << std::endl;
            dump_object_stream << eff_color_vec[eff_id][0] << " " << eff_color_vec[eff_id][1] << " " << eff_color_vec[eff_id][2] << "         /* rgb */ " << std::endl;
            dump_object_stream << cnt_obj.contactPosition()[0] << " " << cnt_obj.contactPosition()[1] << " " << cnt_obj.contactPosition()[2] << "         /* pos -1.45 */ " << std::endl;
            dump_object_stream << cnt_rpy[0] << " " << cnt_rpy[1] << " " << cnt_rpy[2] << "         /* orient */ " << std::endl;
            dump_object_stream << contact_h << " " << contact_w << " " << thickness << "         /* scale */ " << std::endl;
            dump_object_stream << "3         /* contact model */ " << std::endl;
            dump_object_stream << "10            /* object parameters */ " << std::endl;
            dump_object_stream << "7000 20 7000 20  1.5 1.5  0.2 0.2         /* contact parameters */ " << std::endl;
            dump_object_stream << std::endl;
        }
    }

    dump_object_stream.close();
}
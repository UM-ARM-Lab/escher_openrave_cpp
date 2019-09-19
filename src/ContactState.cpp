#include "Utilities.hpp"

Stance::Stance(RPYTF _left_foot_pose, RPYTF _right_foot_pose, RPYTF _left_hand_pose, RPYTF _right_hand_pose, std::array<bool,ContactManipulator::MANIP_NUM> _ee_contact_status):
               left_foot_pose_(_left_foot_pose),
               right_foot_pose_(_right_foot_pose),
               left_hand_pose_(_left_hand_pose),
               right_hand_pose_(_right_hand_pose),
               ee_contact_status_(_ee_contact_status)
{
    this->ee_contact_poses_[ContactManipulator::L_LEG] = left_foot_pose_;
    this->ee_contact_poses_[ContactManipulator::R_LEG] = right_foot_pose_;
    this->ee_contact_poses_[ContactManipulator::L_ARM] = left_hand_pose_;
    this->ee_contact_poses_[ContactManipulator::R_ARM] = right_hand_pose_;
}

bool Stance::operator==(const Stance& other) const
{
    // if there is any difference in ee contact status, return false
    for(int i = 0; i < ContactManipulator::MANIP_NUM; i++)
    {
        if(this->ee_contact_status_[i] != other.ee_contact_status_[i])
        {
            return false;
        }
    }

    // in those ee that is in contact, check if they have the same pose
    for(int i = 0; i < ContactManipulator::MANIP_NUM; i++)
    {
        if(this->ee_contact_status_[i])
        {
            if(this->ee_contact_poses_[i] != other.ee_contact_poses_[i])
            {
                return false;
            }
        }

    }

    return true;
}

bool Stance::operator!=(const Stance& other) const
{
    return !(*this == other);
}

// Constructor for the initial state
ContactState::ContactState(std::vector< std::shared_ptr<Stance> > _initial_stance_vector, Translation3D _initial_com, Vector3D _initial_com_dot, Vector3D _initial_lmom, Vector3D _initial_amom, std::vector<ContactManipulator> _future_move_manips, bool _is_root):
                           num_stance_in_state_(_initial_stance_vector.size()),
                           com_(_initial_com),
                           com_dot_(_initial_com_dot),
                           lmom_(_initial_lmom),
                           amom_(_initial_amom),
                           is_root_(_is_root),
                           future_move_manips_(_future_move_manips),
                           explore_state_(ExploreState::OPEN),
                           g_(0.0),
                           h_(0.0),
                           priority_value_(-9999.0),
                           accumulated_dynamics_cost_(0.0)
{
    this->stances_vector_.resize(num_stance_in_state_);
    this->stances_vector_ = _initial_stance_vector;

    this->nominal_com_ = Translation3D(0,0,0);
    this->mean_feet_position_ = Translation3D(0,0,0);

    float feet_contact_num = 0;
    for(auto & manip : LEG_MANIPULATORS)
    {
        if(this->manip_in_contact(manip))
        {
            this->nominal_com_[0] += this->stances_vector_[0]->ee_contact_poses_[manip].x_;
            this->nominal_com_[1] += this->stances_vector_[0]->ee_contact_poses_[manip].y_;
            this->nominal_com_[2] += this->stances_vector_[0]->ee_contact_poses_[manip].z_;

            this->mean_feet_position_[0] += this->stances_vector_[0]->ee_contact_poses_[manip].x_;
            this->mean_feet_position_[1] += this->stances_vector_[0]->ee_contact_poses_[manip].y_;
            this->mean_feet_position_[2] += this->stances_vector_[0]->ee_contact_poses_[manip].z_;

            feet_contact_num += 1.0;
        }
    }

    this->nominal_com_[0] /= feet_contact_num;
    this->nominal_com_[1] /= feet_contact_num;
    this->nominal_com_[2] /= feet_contact_num;
    this->mean_feet_position_[0] /= feet_contact_num;
    this->mean_feet_position_[1] /= feet_contact_num;
    this->mean_feet_position_[2] /= feet_contact_num;
    // this->nominal_com_[2] += 1.0;
    this->nominal_com_[2] += 0.7;

    this->max_manip_x_ = -9999.0;

    for(int i = 0; i < ContactManipulator::MANIP_NUM; i++)
    {
        if(this->stances_vector_[0]->ee_contact_status_[i])
        {
            if(this->stances_vector_[0]->ee_contact_poses_[i].x_ > this->max_manip_x_)
            {
                this->max_manip_x_ = this->stances_vector_[0]->ee_contact_poses_[i].x_;
            }
        }
    }

    this->depth_ = 0;
}

// Constructor for other states
ContactState::ContactState(std::shared_ptr<Stance> _new_stance, std::shared_ptr<ContactState> _parent, ContactManipulator _move_manip, int _num_stance_in_state, const float _robot_com_z):
                           parent_(_parent),
                           is_root_(false),
                           num_stance_in_state_(_num_stance_in_state)
{
    // updates the stances_vector_
    this->stances_vector_.resize(num_stance_in_state_);
    this->stances_vector_[num_stance_in_state_-1] = _new_stance;
    for(int i = 0; i < num_stance_in_state_-1; i++)
    {
        this->stances_vector_[i] = _parent->stances_vector_[i+1];
    }

    if(num_stance_in_state_ > 1)
    {
        prev_move_manip_ = parent_->future_move_manips_[0];
        future_move_manips_.resize(parent_->future_move_manips_.size());
        for(int i = 0; i < future_move_manips_.size()-1; i++)
        {
            future_move_manips_[i] = parent_->future_move_manips_[i+1];
        }
        future_move_manips_[future_move_manips_.size()-1] = _move_manip;
    }
    else
    {
        prev_move_manip_ = _move_manip;
    }

    this->com_ = Translation3D(0,0,0);
    this->mean_feet_position_ = Translation3D(0,0,0);

    float feet_contact_num = 0;
    for(auto & manip : LEG_MANIPULATORS)
    {
        if(this->manip_in_contact(manip))
        {
            this->com_[0] += this->stances_vector_[0]->ee_contact_poses_[manip].x_;
            this->com_[1] += this->stances_vector_[0]->ee_contact_poses_[manip].y_;
            this->com_[2] += this->stances_vector_[0]->ee_contact_poses_[manip].z_;

            this->mean_feet_position_[0] += this->stances_vector_[0]->ee_contact_poses_[manip].x_;
            this->mean_feet_position_[1] += this->stances_vector_[0]->ee_contact_poses_[manip].y_;
            this->mean_feet_position_[2] += this->stances_vector_[0]->ee_contact_poses_[manip].z_;

            feet_contact_num += 1.0;
        }
    }

    this->com_[0] /= feet_contact_num;
    this->com_[1] /= feet_contact_num;
    this->com_[2] /= feet_contact_num;
    this->mean_feet_position_[0] /= feet_contact_num;
    this->mean_feet_position_[1] /= feet_contact_num;
    this->mean_feet_position_[2] /= feet_contact_num;
    this->com_[2] += _robot_com_z;

    this->com_dot_ = Vector3D::Zero();
    this->lmom_ = Vector3D::Zero();
    this->amom_ = Vector3D::Zero();
    this->nominal_com_ = this->com_;

    // initialize the explore states
    explore_state_ = ExploreState::OPEN;

    // update the g
    this->g_ = _parent->g_;

    // update the h
    this->h_ = 0.0;

    this->max_manip_x_ = -9999.0;

    for(int i = 0; i < ContactManipulator::MANIP_NUM; i++)
    {
        if(this->stances_vector_[0]->ee_contact_status_[i])
        {
            if(this->stances_vector_[0]->ee_contact_poses_[i].x_ > this->max_manip_x_)
            {
                this->max_manip_x_ = this->stances_vector_[0]->ee_contact_poses_[i].x_;
            }
        }
    }

    this->depth_ = _parent->depth_ + 1;
}

bool ContactState::operator==(const ContactState& other) const
{
    for(int i = 0; i < num_stance_in_state_; i++)
    {
        if(*(this->stances_vector_[i]) != *(other.stances_vector_[i]))
        {
            return false;
        }
    }

    return true;
}

bool ContactState::operator!=(const ContactState& other) const
{
    return !(*this == other);
}

float ContactState::getLeftHorizontalYaw(int stance_index)
{
    RotationMatrix l_foot_rotation = RPYToSO3(this->stances_vector_[stance_index]->left_foot_pose_);
    Vector3D cy = l_foot_rotation.col(1);
    Vector3D nx = cy.cross(Vector3D(0,0,1));

    return (round(std::atan2(nx[1], nx[0]) * RAD2DEG * 10.0) / 10.0);
}

float ContactState::getRightHorizontalYaw(int stance_index)
{
    RotationMatrix r_foot_rotation = RPYToSO3(this->stances_vector_[stance_index]->right_foot_pose_);
    Vector3D cy = r_foot_rotation.col(1);
    Vector3D nx = cy.cross(Vector3D(0,0,1));

    return (round(std::atan2(nx[1], nx[0]) * RAD2DEG * 10.0) / 10.0);
}

float ContactState::getFeetMeanHorizontalYaw(int stance_index)
{
    if(stances_vector_[stance_index]->ee_contact_status_[ContactManipulator::L_LEG] && stances_vector_[stance_index]->ee_contact_status_[ContactManipulator::R_LEG])
    {
        return getAngleMean(this->getLeftHorizontalYaw(), this->getRightHorizontalYaw());
    }
    else if(stances_vector_[stance_index]->ee_contact_status_[ContactManipulator::L_LEG])
    {
        return this->getLeftHorizontalYaw();
    }
    else if(stances_vector_[stance_index]->ee_contact_status_[ContactManipulator::R_LEG])
    {
        return this->getRightHorizontalYaw();
    }
}

TransformationMatrix ContactState::getFeetMeanTransform(int stance_index)
{
    float feet_mean_x, feet_mean_y, feet_mean_z, feet_mean_roll, feet_mean_pitch, feet_mean_yaw;
    if(this->stances_vector_[stance_index]->ee_contact_status_[ContactManipulator::L_LEG] && this->stances_vector_[stance_index]->ee_contact_status_[ContactManipulator::R_LEG])
    {
        feet_mean_x = (this->stances_vector_[stance_index]->left_foot_pose_.x_ + this->stances_vector_[stance_index]->right_foot_pose_.x_) / 2.0;
        feet_mean_y = (this->stances_vector_[stance_index]->left_foot_pose_.y_ + this->stances_vector_[stance_index]->right_foot_pose_.y_) / 2.0;
        feet_mean_z = (this->stances_vector_[stance_index]->left_foot_pose_.z_ + this->stances_vector_[stance_index]->right_foot_pose_.z_) / 2.0;
        feet_mean_roll = 0;
        feet_mean_pitch = 0;
        feet_mean_yaw = this->getFeetMeanHorizontalYaw();
    }
    else if(this->stances_vector_[stance_index]->ee_contact_status_[ContactManipulator::L_LEG])
    {
        feet_mean_x = this->stances_vector_[stance_index]->left_foot_pose_.x_;
        feet_mean_y = this->stances_vector_[stance_index]->left_foot_pose_.y_;
        feet_mean_z = this->stances_vector_[stance_index]->left_foot_pose_.z_;
        feet_mean_roll = 0;
        feet_mean_pitch = 0;
        feet_mean_yaw = this->getLeftHorizontalYaw();
    }
    else if(this->stances_vector_[stance_index]->ee_contact_status_[ContactManipulator::R_LEG])
    {
        feet_mean_x = this->stances_vector_[stance_index]->right_foot_pose_.x_;
        feet_mean_y = this->stances_vector_[stance_index]->right_foot_pose_.y_;
        feet_mean_z = this->stances_vector_[stance_index]->right_foot_pose_.z_;
        feet_mean_roll = 0;
        feet_mean_pitch = 0;
        feet_mean_yaw = this->getRightHorizontalYaw();
    }
    else
    {
        RAVELOG_ERROR("Try to getFeetMeanTransform from a state without both foot contact. Error!");
        getchar();
    }

    return XYZRPYToSE3(RPYTF(feet_mean_x, feet_mean_y, feet_mean_z, feet_mean_roll, feet_mean_pitch, feet_mean_yaw));
}

std::shared_ptr<ContactState> ContactState::getNoFutureContactState()
{
    std::shared_ptr<ContactState> no_future_contact_state = std::make_shared<ContactState>(*this);
    no_future_contact_state->stances_vector_.resize(1);
    no_future_contact_state->num_stance_in_state_ = 1;
    no_future_contact_state->future_move_manips_.clear();
    return no_future_contact_state;
}

std::shared_ptr<ContactState> ContactState::getMirrorState(TransformationMatrix& reference_frame)
{
    // mirror the left and right end-effector poses
    TransformationMatrix inv_reference_frame =  inverseTransformationMatrix(reference_frame);
    RotationMatrix reference_frame_rotation = reference_frame.block(0,0,3,3);
    RotationMatrix inv_reference_frame_rotation = inv_reference_frame.block(0,0,3,3);

    RotationMatrix mirror_matrix;
    mirror_matrix << 1,  0, 0,
                     0, -1, 0,
                     0,  0, 1;

    std::vector<ContactManipulator> mirror_manip_vec = {ContactManipulator::R_LEG, ContactManipulator::L_LEG, ContactManipulator::R_ARM, ContactManipulator::L_ARM};

    std::vector< std::shared_ptr<Stance> > mirror_stance_vector(stances_vector_.size());
    std::vector<ContactManipulator> mirror_future_move_manips(future_move_manips_.size());

    int stance_id = 0;
    for(auto & stance : stances_vector_)
    {
        std::array<RPYTF,ContactManipulator::MANIP_NUM> mirror_ee_contact_pose;
        std::array<bool,ContactManipulator::MANIP_NUM> mirror_ee_contact_status;

        // mirror the end-effector poses
        for(auto & manip : ALL_MANIPULATORS)
        {
            RPYTF mirrored_contact_pose_rpy;

            if(stance->ee_contact_status_[int(manip)])
            {
                TransformationMatrix transformed_contact_pose = inv_reference_frame * XYZRPYToSE3(stance->ee_contact_poses_[int(manip)]);
                TransformationMatrix mirrored_contact_pose = transformed_contact_pose;
                mirrored_contact_pose.block(0,0,3,3) = mirror_matrix * transformed_contact_pose.block(0,0,3,3) * mirror_matrix;
                mirrored_contact_pose(1,3) = -transformed_contact_pose(1,3);
                mirrored_contact_pose_rpy = SE3ToXYZRPY(reference_frame * mirrored_contact_pose);
            }
            else
            {
                mirrored_contact_pose_rpy = stance->ee_contact_poses_[int(manip)];
            }

            mirror_ee_contact_pose[int(mirror_manip_vec[int(manip)])] = mirrored_contact_pose_rpy;
            mirror_ee_contact_status[int(mirror_manip_vec[int(manip)])] = stance->ee_contact_status_[int(manip)];
        }

        mirror_stance_vector[stance_id] = std::make_shared<Stance>(mirror_ee_contact_pose[0], mirror_ee_contact_pose[1], mirror_ee_contact_pose[2], mirror_ee_contact_pose[3], mirror_ee_contact_status);
        stance_id++;
    }

    for(int i = 0; i < future_move_manips_.size(); i++)
    {
        mirror_future_move_manips[i] = mirror_manip_vec[int(future_move_manips_[i])];
    }

    // mirror the com and com dot
    Translation3D transformed_com = (inv_reference_frame * com_.homogeneous()).block(0,0,3,1);
    transformed_com[1] = -transformed_com[1];
    Translation3D mirror_com = (reference_frame * transformed_com.homogeneous()).block(0,0,3,1);

    Vector3D transformed_com_dot = inv_reference_frame_rotation * com_dot_;
    transformed_com_dot[1] = -transformed_com_dot[1];
    Vector3D mirror_com_dot = reference_frame_rotation * transformed_com_dot;

    Vector3D transformed_lmom = inv_reference_frame_rotation * lmom_;
    transformed_lmom[1] = -transformed_lmom[1];
    Vector3D mirror_lmom = reference_frame_rotation * transformed_lmom;

    Vector3D transformed_amom = inv_reference_frame_rotation * amom_;
    transformed_amom[0] = -transformed_amom[0];
    transformed_amom[2] = -transformed_amom[2];
    Vector3D mirror_amom = reference_frame_rotation * transformed_amom;

    std::shared_ptr<ContactState> mirror_state = std::make_shared<ContactState>(mirror_stance_vector, mirror_com, mirror_com_dot, mirror_lmom, mirror_amom, mirror_future_move_manips, is_root_);
    if(!is_root_)
    {
        mirror_state->prev_move_manip_ = mirror_manip_vec[int(prev_move_manip_)];
    }

    return mirror_state;
}

std::shared_ptr<ContactState> ContactState::getCenteredState(TransformationMatrix& reference_frame)
{
    // mirror the left and right end-effector poses
    TransformationMatrix inv_reference_frame =  inverseTransformationMatrix(reference_frame);

    RotationMatrix reference_frame_rotation = reference_frame.block(0,0,3,3);
    RotationMatrix inv_reference_frame_rotation = inv_reference_frame.block(0,0,3,3);

    std::vector< std::shared_ptr<Stance> > centered_stance_vector;
    centered_stance_vector.reserve(stances_vector_.size());

    for(auto & stance : stances_vector_)
    {
        std::array<RPYTF,ContactManipulator::MANIP_NUM> centered_ee_contact_pose;

        // mirror the end-effector poses
        for(auto & manip : ALL_MANIPULATORS)
        {
            if(stance->ee_contact_status_[int(manip)])
            {
                centered_ee_contact_pose[int(manip)] = SE3ToXYZRPY(inv_reference_frame * XYZRPYToSE3(stance->ee_contact_poses_[int(manip)]));
            }
            else
            {
                centered_ee_contact_pose[int(manip)] = stance->ee_contact_poses_[int(manip)];
            }
        }

        centered_stance_vector.push_back( std::make_shared<Stance>(centered_ee_contact_pose[0],
                                                                   centered_ee_contact_pose[1],
                                                                   centered_ee_contact_pose[2],
                                                                   centered_ee_contact_pose[3],
                                                                   stance->ee_contact_status_));
    }

    // mirror the com and com dot
    Translation3D centered_com = (inv_reference_frame * com_.homogeneous()).block(0,0,3,1);
    Vector3D centered_com_dot = inv_reference_frame_rotation * com_dot_;
    Vector3D centered_lmom = inv_reference_frame_rotation * lmom_;
    Vector3D centered_amom = inv_reference_frame_rotation * amom_;

    std::shared_ptr<ContactState> centered_state = std::make_shared<ContactState>(centered_stance_vector, centered_com, centered_com_dot, centered_lmom, centered_amom, future_move_manips_, is_root_);
    if(!is_root_)
    {
        centered_state->prev_move_manip_ = prev_move_manip_;
    }

    return centered_state;
}

std::shared_ptr<ContactState> ContactState::getStandardInputState(DynOptApplication dynamics_optimizer_application)
{
    // get the standard state for storing in a training data, and as input to the networks.
    std::shared_ptr<ContactState> standard_state, prev_state;
    TransformationMatrix reference_frame;

    if(dynamics_optimizer_application == DynOptApplication::CONTACT_TRANSITION_DYNOPT ||
       dynamics_optimizer_application == DynOptApplication::ONE_STEP_CAPTURABILITY_DYNOPT)
    {
        // standard_state = std::make_shared<ContactState>(*this);
        // prev_state = std::make_shared<ContactState>(*parent_);
        standard_state = shared_from_this();
        prev_state = parent_;
        reference_frame = prev_state->getFeetMeanTransform();

        // mirror the states if necessary
        if(standard_state->prev_move_manip_ == ContactManipulator::R_LEG || standard_state->prev_move_manip_ == ContactManipulator::R_ARM)
        {
            standard_state = standard_state->getMirrorState(reference_frame);
            prev_state = prev_state->getMirrorState(reference_frame);
        }

        standard_state = standard_state->getCenteredState(reference_frame);
        prev_state = prev_state->getCenteredState(reference_frame);

        standard_state->parent_ = prev_state;

        if(standard_state->prev_move_manip_ != ContactManipulator::L_LEG && standard_state->prev_move_manip_ != ContactManipulator::L_ARM)
        {
            RAVELOG_ERROR("A standard state which moves right hand side of the robot. This should not happen.\n");
            getchar();
        }
    }
    else if(dynamics_optimizer_application == DynOptApplication::ZERO_STEP_CAPTURABILITY_DYNOPT)
    {
        // standard_state = std::make_shared<ContactState>(*this);
        standard_state = shared_from_this();
        reference_frame = standard_state->getFeetMeanTransform();

        if((standard_state->manip_in_contact(ContactManipulator::L_LEG) && !standard_state->manip_in_contact(ContactManipulator::R_LEG)) ||
           (standard_state->manip_in_contact(ContactManipulator::L_LEG) && standard_state->manip_in_contact(ContactManipulator::R_LEG) && standard_state->manip_in_contact(ContactManipulator::L_ARM)))
        {
            standard_state = standard_state->getMirrorState(reference_frame);
        }

        standard_state = standard_state->getCenteredState(reference_frame);
    }

    return standard_state;
}

std::pair<ContactTransitionCode, std::vector<RPYTF> > ContactState::getTransitionCodeAndPoses()
{
    std::shared_ptr<ContactState> prev_state = parent_;
    std::shared_ptr<Stance> current_stance = stances_vector_[0];
    std::shared_ptr<Stance> prev_stance = prev_state->stances_vector_[0];

    ContactTransitionCode contact_transition_code;

    std::vector<RPYTF> contact_manip_pose_vec = {prev_stance->left_foot_pose_, prev_stance->right_foot_pose_};

    if(!prev_stance->ee_contact_status_[ContactManipulator::L_ARM] && !prev_stance->ee_contact_status_[ContactManipulator::R_ARM])
    {
        if(prev_move_manip_ == ContactManipulator::L_LEG)
        {
            contact_transition_code = ContactTransitionCode::FEET_ONLY_MOVE_FOOT;
            contact_manip_pose_vec.push_back(current_stance->left_foot_pose_);
        }
        else if(prev_move_manip_ == ContactManipulator::L_ARM)
        {
            contact_transition_code = ContactTransitionCode::FEET_ONLY_ADD_HAND;
            contact_manip_pose_vec.push_back(current_stance->left_hand_pose_);
        }
        else
        {
            RAVELOG_ERROR("Unknown Contact Transition.\n");
            getchar();
        }
    }
    else if(prev_stance->ee_contact_status_[ContactManipulator::L_ARM] && prev_stance->ee_contact_status_[ContactManipulator::R_ARM])
    {
        if(prev_move_manip_ == ContactManipulator::L_LEG)
        {
            contact_transition_code = ContactTransitionCode::FEET_AND_TWO_HANDS_MOVE_FOOT;
            contact_manip_pose_vec.push_back(prev_stance->left_hand_pose_);
            contact_manip_pose_vec.push_back(prev_stance->right_hand_pose_);
            contact_manip_pose_vec.push_back(current_stance->left_foot_pose_);
        }
        else if(prev_move_manip_ == ContactManipulator::L_ARM && !current_stance->ee_contact_status_[ContactManipulator::L_ARM])
        {
            contact_transition_code = ContactTransitionCode::FEET_AND_TWO_HANDS_BREAK_HAND;
            contact_manip_pose_vec.push_back(prev_stance->left_hand_pose_);
            contact_manip_pose_vec.push_back(prev_stance->right_hand_pose_);
        }
        else if(prev_move_manip_ == ContactManipulator::L_ARM && current_stance->ee_contact_status_[ContactManipulator::L_ARM])
        {
            contact_transition_code = ContactTransitionCode::FEET_AND_TWO_HANDS_MOVE_HAND;
            contact_manip_pose_vec.push_back(prev_stance->left_hand_pose_);
            contact_manip_pose_vec.push_back(prev_stance->right_hand_pose_);
            contact_manip_pose_vec.push_back(current_stance->left_hand_pose_);
        }
        else
        {
            RAVELOG_ERROR("Unknown Contact Transition.\n");
            getchar();
        }
    }
    else
    {
        if(prev_move_manip_ == ContactManipulator::L_LEG && current_stance->ee_contact_status_[ContactManipulator::L_ARM])
        {
            contact_transition_code = ContactTransitionCode::FEET_AND_ONE_HAND_MOVE_INNER_FOOT;
            contact_manip_pose_vec.push_back(prev_stance->left_hand_pose_);
            contact_manip_pose_vec.push_back(current_stance->left_foot_pose_);
        }
        else if(prev_move_manip_ == ContactManipulator::L_LEG && current_stance->ee_contact_status_[ContactManipulator::R_ARM])
        {
            contact_transition_code = ContactTransitionCode::FEET_AND_ONE_HAND_MOVE_OUTER_FOOT;
            contact_manip_pose_vec.push_back(prev_stance->right_hand_pose_);
            contact_manip_pose_vec.push_back(current_stance->left_foot_pose_);
        }
        else if(prev_move_manip_ == ContactManipulator::L_ARM && !current_stance->ee_contact_status_[ContactManipulator::L_ARM])
        {
            contact_transition_code = ContactTransitionCode::FEET_AND_ONE_HAND_BREAK_HAND;
            contact_manip_pose_vec.push_back(prev_stance->left_hand_pose_);
        }
        else if(prev_move_manip_ == ContactManipulator::L_ARM && current_stance->ee_contact_status_[ContactManipulator::L_ARM] && !current_stance->ee_contact_status_[ContactManipulator::R_ARM])
        {
            contact_transition_code = ContactTransitionCode::FEET_AND_ONE_HAND_MOVE_HAND;
            contact_manip_pose_vec.push_back(prev_stance->left_hand_pose_);
            contact_manip_pose_vec.push_back(current_stance->left_hand_pose_);
        }
        else if(prev_move_manip_ == ContactManipulator::L_ARM && current_stance->ee_contact_status_[ContactManipulator::L_ARM] && current_stance->ee_contact_status_[ContactManipulator::R_ARM])
        {
            contact_transition_code = ContactTransitionCode::FEET_AND_ONE_HAND_ADD_HAND;
            contact_manip_pose_vec.push_back(prev_stance->right_hand_pose_);
            contact_manip_pose_vec.push_back(current_stance->left_hand_pose_);
        }
        else
        {
            RAVELOG_ERROR("Unknown Contact Transition.\n");
            getchar();
        }
    }

    return std::make_pair(contact_transition_code, contact_manip_pose_vec);
}

std::pair<OneStepCaptureCode, std::vector<RPYTF> > ContactState::getOneStepCapturabilityCodeAndPoses()
{
    std::shared_ptr<ContactState> prev_state = parent_;
    std::shared_ptr<Stance> current_stance = stances_vector_[0];
    std::shared_ptr<Stance> prev_stance = prev_state->stances_vector_[0];

    OneStepCaptureCode one_step_capture_code;

    std::vector<RPYTF> contact_manip_pose_vec;

    // {L_LEG, R_LEG, L_ARM, R_ARM}
    // enum OneStepCaptureCode
    // {
    //     ONE_FOOT_ADD_FOOT,                  // 0
    //     ONE_FOOT_ADD_INNER_HAND,            // 1
    //     ONE_FOOT_ADD_OUTER_HAND,            // 2
    //     TWO_FEET_ADD_HAND,                  // 3
    //     ONE_FOOT_AND_INNER_HAND_ADD_FOOT,   // 4
    //     ONE_FOOT_AND_INNER_HAND_ADD_HAND,   // 5
    //     ONE_FOOT_AND_OUTER_HAND_ADD_FOOT,   // 6
    //     ONE_FOOT_AND_OUTER_HAND_ADD_HAND,   // 7
    //     ONE_FOOT_AND_TWO_HANDS_ADD_FOOT,    // 8
    //     TWO_FEET_AND_ONE_HAND_ADD_HAND      // 9
    // };

    const std::array<bool,ContactManipulator::MANIP_NUM> only_left_foot = {true,false,false,false};
    const std::array<bool,ContactManipulator::MANIP_NUM> only_right_foot = {false,true,false,false};
    const std::array<bool,ContactManipulator::MANIP_NUM> both_feet = {true,true,false,false};
    const std::array<bool,ContactManipulator::MANIP_NUM> right_foot_and_right_hand = {false,true,false,true};
    const std::array<bool,ContactManipulator::MANIP_NUM> left_foot_and_right_hand = {true,false,false,true};
    const std::array<bool,ContactManipulator::MANIP_NUM> right_foot_and_left_hand = {false,true,true,false};
    const std::array<bool,ContactManipulator::MANIP_NUM> right_foot_and_both_hands = {false,true,true,true};
    const std::array<bool,ContactManipulator::MANIP_NUM> both_feet_and_right_hand = {true,true,false,true};

    // one foot contact
    if(prev_stance->ee_contact_status_ == only_left_foot || prev_stance->ee_contact_status_ == only_right_foot)
    {
        if(prev_stance->ee_contact_status_[ContactManipulator::R_LEG] && prev_move_manip_ == ContactManipulator::L_LEG)
        {
            one_step_capture_code = OneStepCaptureCode::ONE_FOOT_ADD_FOOT;
        }
        else if(prev_stance->ee_contact_status_[ContactManipulator::L_LEG] && prev_move_manip_ == ContactManipulator::L_ARM)
        {
            one_step_capture_code = OneStepCaptureCode::ONE_FOOT_ADD_INNER_HAND;
        }
        else if(prev_stance->ee_contact_status_[ContactManipulator::R_LEG] && prev_move_manip_ == ContactManipulator::L_ARM)
        {
            one_step_capture_code = OneStepCaptureCode::ONE_FOOT_ADD_OUTER_HAND;
        }
        else
        {
            RAVELOG_ERROR("Unknown One Step Capture Case.\n");
            getchar();
        }
    }
    else if(prev_stance->ee_contact_status_ == both_feet)
    {
        if(prev_move_manip_ == ContactManipulator::L_ARM)
        {
            one_step_capture_code = OneStepCaptureCode::TWO_FEET_ADD_HAND;
        }
        else
        {
            RAVELOG_ERROR("Unknown One Step Capture Case.\n");
            getchar();
        }
    }
    else if(prev_stance->ee_contact_status_ == right_foot_and_right_hand)
    {
        if(prev_move_manip_ == ContactManipulator::L_LEG)
        {
            one_step_capture_code = OneStepCaptureCode::ONE_FOOT_AND_INNER_HAND_ADD_FOOT;
        }
        else if(prev_move_manip_ == ContactManipulator::L_ARM)
        {
            one_step_capture_code = OneStepCaptureCode::ONE_FOOT_AND_INNER_HAND_ADD_HAND;
        }
        else
        {
            RAVELOG_ERROR("Unknown One Step Capture Case.\n");
            getchar();
        }
    }
    else if(prev_stance->ee_contact_status_ == left_foot_and_right_hand || prev_stance->ee_contact_status_ == right_foot_and_left_hand)
    {
        if(prev_move_manip_ == ContactManipulator::L_LEG)
        {
            one_step_capture_code = OneStepCaptureCode::ONE_FOOT_AND_OUTER_HAND_ADD_FOOT;
        }
        else if(prev_move_manip_ == ContactManipulator::L_ARM)
        {
            one_step_capture_code = OneStepCaptureCode::ONE_FOOT_AND_OUTER_HAND_ADD_HAND;
        }
        else
        {
            RAVELOG_ERROR("Unknown One Step Capture Case.\n");
            getchar();
        }
    }
    else if(prev_stance->ee_contact_status_ == right_foot_and_both_hands)
    {
        if(prev_move_manip_ == ContactManipulator::L_LEG)
        {
            one_step_capture_code = OneStepCaptureCode::ONE_FOOT_AND_TWO_HANDS_ADD_FOOT;
        }
        else
        {
            RAVELOG_ERROR("Unknown One Step Capture Case.\n");
            getchar();
        }
    }
    else if(prev_stance->ee_contact_status_ == both_feet_and_right_hand)
    {
        if(prev_move_manip_ == ContactManipulator::L_ARM)
        {
            one_step_capture_code = OneStepCaptureCode::TWO_FEET_AND_ONE_HAND_ADD_HAND;
        }
        else
        {
            RAVELOG_ERROR("Unknown One Step Capture Case.\n");
            getchar();
        }
    }
    else
    {
        RAVELOG_ERROR("Unknown One Step Capture Case.\n");
        getchar();
    }

    for(auto & manip : ALL_MANIPULATORS)
    {
        if(prev_stance->ee_contact_status_[manip])
        {
            contact_manip_pose_vec.push_back(prev_stance->ee_contact_poses_[manip]);
        }
    }
    contact_manip_pose_vec.push_back(current_stance->ee_contact_poses_[prev_move_manip_]);

    return std::make_pair(one_step_capture_code, contact_manip_pose_vec);
}

std::pair<ZeroStepCaptureCode, std::vector<RPYTF> > ContactState::getZeroStepCapturabilityCodeAndPoses()
{
    std::shared_ptr<Stance> current_stance = stances_vector_[0];

    ZeroStepCaptureCode zero_step_capture_code;

    std::vector<RPYTF> contact_manip_pose_vec;

    // {L_LEG, R_LEG, L_ARM, R_ARM}
    // enum ZeroStepCaptureCode
    // {
    //     ONE_FOOT,                   // 0
    //     TWO_FEET,                   // 1
    //     ONE_FOOT_AND_INNER_HAND,    // 2
    //     ONE_FOOT_AND_OUTER_HAND,    // 3
    //     ONE_FOOT_AND_TWO_HANDS,     // 4
    //     FEET_AND_ONE_HAND           // 5
    // };

    const std::array<bool,ContactManipulator::MANIP_NUM> only_right_foot = {false,true,false,false};
    const std::array<bool,ContactManipulator::MANIP_NUM> both_feet = {true,true,false,false};
    const std::array<bool,ContactManipulator::MANIP_NUM> right_foot_and_right_hand = {false,true,false,true};
    const std::array<bool,ContactManipulator::MANIP_NUM> right_foot_and_left_hand = {false,true,true,false};
    const std::array<bool,ContactManipulator::MANIP_NUM> right_foot_and_both_hands = {false,true,true,true};
    const std::array<bool,ContactManipulator::MANIP_NUM> both_feet_and_right_hand = {true,true,false,true};

    // one foot contact
    if(current_stance->ee_contact_status_ == only_right_foot)
    {
        zero_step_capture_code = ZeroStepCaptureCode::ONE_FOOT;
    }
    else if(current_stance->ee_contact_status_ == both_feet)
    {
        zero_step_capture_code = ZeroStepCaptureCode::TWO_FEET;
    }
    else if(current_stance->ee_contact_status_ == right_foot_and_right_hand)
    {
        zero_step_capture_code = ZeroStepCaptureCode::ONE_FOOT_AND_INNER_HAND;
    }
    else if(current_stance->ee_contact_status_ == right_foot_and_left_hand)
    {
        zero_step_capture_code = ZeroStepCaptureCode::ONE_FOOT_AND_OUTER_HAND;
    }
    else if(current_stance->ee_contact_status_ == right_foot_and_both_hands)
    {
        zero_step_capture_code = ZeroStepCaptureCode::ONE_FOOT_AND_TWO_HANDS;
    }
    else if(current_stance->ee_contact_status_ == both_feet_and_right_hand)
    {
        zero_step_capture_code = ZeroStepCaptureCode::FEET_AND_ONE_HAND;
    }
    else
    {
        RAVELOG_ERROR("Unknown Zero Step Capture Case.\n");
        getchar();
    }

    for(auto & manip : ALL_MANIPULATORS)
    {
        if(current_stance->ee_contact_status_[manip])
        {
            contact_manip_pose_vec.push_back(current_stance->ee_contact_poses_[manip]);
        }
    }

    return std::make_pair(zero_step_capture_code, contact_manip_pose_vec);
}

void ContactState::printStateInfo()
{
    std::vector<std::string> contact_manip_name_vector = {"L_LEG", "R_LEG", "L_ARM", "R_ARM"};
    std::cout << "contact manipulator: " << std::endl;

    for(int i = 0; i < stances_vector_[0]->ee_contact_poses_.size(); i++)
    {
        if(stances_vector_[0]->ee_contact_status_[i])
        {
            std::cout << contact_manip_name_vector[i] << ": ";
            stances_vector_[0]->ee_contact_poses_[i].printPose();
        }
    }

    std::cout << "com: " << com_.transpose() << std::endl;
    std::cout << "com dot: " << com_dot_.transpose() << std::endl;
    std::cout << "is root: " << is_root_ << std::endl;

    if(!is_root_)
    {
        std::cout << "prev move manip: " << prev_move_manip_ << std::endl;
    }

}

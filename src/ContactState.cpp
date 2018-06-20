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


ContactState::ContactState(std::shared_ptr<Stance> new_stance, std::shared_ptr<ContactState> _parent, ContactManipulator _move_manip, bool _is_root):
                           parent_(_parent),
                           prev_move_manip_(_move_manip),
                           is_root_(_is_root)
{
    // updates the stances_array_
    this->stances_array_[0] = new_stance;
    for(int i = 0; i < NUM_STANCE_IN_STATE-1; i++)
    {
        this->stances_array_[i] = _parent->stances_array_[i+1];
    }

    // updates the com_ and com_dot_ by optimization (will be time consuming) probably need to parallelize the construction of states
    this->com_ = {0,0,0};
    this->com_dot_ = {0,0,0};

    // run the dynamics optimization here to get the dynamics edge cost (maybe outside the contact state)

    // initialize the explore states
    explore_state_ = ExploreState::OPEN;

    // update the g
    this->g_ = _parent->g_;

    // update the h
    this->h_ = 0.0;
}

void ContactState::UpdateCostsAndCoM(float edge_cost, float heuristics, std::array<float,3> com, std::array<float,3> com_dot)
{
    this->g_ += edge_cost;
    this->h_ = heuristics;
    this->com_ = com;
    this->com_dot_ = com_dot;
}

bool ContactState::operator==(const ContactState& other) const
{
    for(int i = 0; i < NUM_STANCE_IN_STATE; i++)
    {
        if(*(this->stances_array_[i]) != *(other.stances_array_[i]))
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

float ContactState::getLeftHorizontalYaw()
{
    RotationMatrix l_foot_rotation = RPYToSO3(this->stances_array_[0]->left_foot_pose_);
    Vector3D cy = l_foot_rotation.col(1);
    Vector3D nx = cy.cross(Vector3D(0,0,1));

    return (round(std::atan2(nx[1], nx[0]) * RAD2DEG * 10.0) / 10.0);
}

float ContactState::getRightHorizontalYaw()
{
    RotationMatrix r_foot_rotation = RPYToSO3(this->stances_array_[0]->right_foot_pose_);
    Vector3D cy = r_foot_rotation.col(1);
    Vector3D nx = cy.cross(Vector3D(0,0,1));

    return (round(std::atan2(nx[1], nx[0]) * RAD2DEG * 10.0) / 10.0);
}

float ContactState::getFeetMeanHorizontalYaw()
{
    return getAngleMean(this->getLeftHorizontalYaw(), this->getRightHorizontalYaw());
}

TransformationMatrix ContactState::getFeetMeanTransform()
{
    float feet_mean_x = (this->stances_array_[0]->left_foot_pose_.x_ + this->stances_array_[0]->right_foot_pose_.x_) / 2.0;
    float feet_mean_y = (this->stances_array_[0]->left_foot_pose_.y_ + this->stances_array_[0]->right_foot_pose_.y_) / 2.0;
    float feet_mean_z = (this->stances_array_[0]->left_foot_pose_.z_ + this->stances_array_[0]->right_foot_pose_.z_) / 2.0;
    float feet_mean_roll = 0;
    float feet_mean_pitch = 0;
    float feet_mean_yaw = this->getFeetMeanHorizontalYaw();
    
    return XYZRPYToSE3(RPYTF(feet_mean_x, feet_mean_y, feet_mean_z, feet_mean_roll, feet_mean_pitch, feet_mean_yaw));
}
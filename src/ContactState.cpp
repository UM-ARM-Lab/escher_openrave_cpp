#include "Utilities.hpp"

bool Stance::operator==(const ContactState& other) const
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

bool Stance::operator!=(const ContactState& other) const 
{
    return !(*this == other);
}


ContactState::ContactState(Stance new_stance, std::shared_ptr<ContactState> _parent, ContactManipulator _move_manip, bool _is_root):
                           parent_(_parent),
                           prev_move_manip_(_move_manip),
                           is_root_(_is_root)
{
    // updates the stance_array_
    this->stance_array_[0] = new_stance;
    for(int i = 0; i < NUM_STANCE_IN_STATE-1; i++)
    {
        this->stance_array_[i] = _parent->stances_array_[i+1];
    }

    // update the g
    this->g_ = _parent.g_ + _parent.getEdgeCost(_move_manip, new_stance.ee_contact_poses_[_move_manip]);

    // update the h
    this->h_ = this->getHeusitics();
}

bool ContactState::operator==(const ContactState& other) const
{
    for(int i = 0; i < NUM_STANCE_IN_STATE; i++)
    {
        if(this->stance_array_[i] != other.stance_array_[i])
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

float ContactState::getEdgeCost(ContactManipulator _move_manip, RPYTF _new_pose)
{
    return 0.0;
}

float ContactState::getHeusitics()
{
    return 0.0;
}
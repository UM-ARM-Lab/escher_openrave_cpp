#ifndef CONTACT_STATE_H
#define CONTACT_STATE_H

class Stance(RPYTF _left_foot_pose, RPYTF _right_foot_pose, RPYTF _left_hand_pose, RPYTF _right_hand_pose)
{
    public:
        Stance(RPYTF _left_foot_pose, RPYTF _right_foot_pose, RPYTF _left_hand_pose, RPYTF _right_hand_pose, std::map<string,bool> _ee_contact_status);

        const RPYTF left_foot_pose_;
        const RPYTF right_foot_pose_;
        const RPYTF left_hand_pose_;
        const RPYTF right_hand_pose_;

        const std::array<bool,ContactManipulator::MANIP_NUM> ee_contact_status_;
        std::array<RPYTF,ContactManipulator::MANIP_NUM> ee_contact_poses_;

        bool operator==(const ContactState& other) const;
        bool operator!=(const ContactState& other) const;

    private:
}

Stance::Stance(RPYTF _left_foot_pose, RPYTF _right_foot_pose, RPYTF _left_hand_pose, RPYTF _right_hand_pose, std::array<bool,ContactManipulator::MANIP_NUM> _ee_contact_status)
               left_foot_pose_(_left_foot_pose),
               right_foot_pose_(_right_foot_pose),
               left_hand_pose_(_left_hand_pose),
               right_hand_pose_(_right_hand_pose),
               ee_contact_status_(_ee_contact_status)
{
    ee_contact_poses_[ContactManipulator::L_LEG] = left_foot_pose_;
    ee_contact_poses_[ContactManipulator::R_LEG] = right_foot_pose_;
    ee_contact_poses_[ContactManipulator::L_ARM] = left_hand_pose_;
    ee_contact_poses_[ContactManipulator::R_ARM] = rght_hand_pose_;
}

class ContactState
{
    public:
        ContactState(Stance new_stance, std::shared_ptr<ContactState> _parent, ContactManipulator _move_manip, bool _is_root);

        float getEdgeCost(ContactManipulator _move_manip, RPYTF _new_pose);
        float getHeusitics();

        // foot orientation projected to flat gruond
        // float get_left_horizontal_yaw() const;
        // float get_right_horizontal_yaw() const;

        std::array<Stance,NUM_STANCE_IN_STATE> stances_array_;

        enum explore_state{OPEN, EXPLORED, CLOSED, REOPEN};

        const bool is_root_;

        const ContactManipulator prev_move_manip_;
        float g_;
        float h_;

        explore_state explore_state_;

        bool operator==(const ContactState& other) const;
        bool operator!=(const ContactState& other) const;

        inline bool operator<(const ContactState& other) const {return (this->getF() < other.getF());}

        inline std::shared_ptr<ContactState> getParent() {return parent_;}
        inline const float getF() const {return (g_+h_);}

    private:
        std::shared_ptr<ContactState> parent_;
};

#endif
#ifndef CONTACT_STATE_H
#define CONTACT_STATE_H

class ContactState;

class Stance
{
    public:
        Stance(RPYTF _left_foot_pose, RPYTF _right_foot_pose, RPYTF _left_hand_pose, RPYTF _right_hand_pose, std::array<bool,ContactManipulator::MANIP_NUM> _ee_contact_status);

        const RPYTF left_foot_pose_;
        const RPYTF right_foot_pose_;
        const RPYTF left_hand_pose_;
        const RPYTF right_hand_pose_;

        const std::array<bool,ContactManipulator::MANIP_NUM> ee_contact_status_;
        std::array<RPYTF,ContactManipulator::MANIP_NUM> ee_contact_poses_;

        bool operator==(const Stance& other) const;
        bool operator!=(const Stance& other) const;

    private:
};

class ContactState
{
    public:
        ContactState(std::shared_ptr<Stance> new_stance, std::shared_ptr<ContactState> _parent, ContactManipulator _move_manip, bool _is_root, float edge_cost, float heuristics_cost);

        float getEdgeCost(ContactManipulator _move_manip, RPYTF _new_pose);
        float getHeusitics();

        // foot orientation projected to flat gruond
        float getLeftHorizontalYaw();
        float getRightHorizontalYaw();
        float getFeetMeanHorizontalYaw();
        TransformationMatrix getFeetMeanTransform();

        std::array<std::shared_ptr<Stance>,NUM_STANCE_IN_STATE> stances_array_;
        std::array<float,3> com_;
        std::array<float,3> com_dot_;

        const bool is_root_;

        const ContactManipulator prev_move_manip_;
        float g_;
        float h_;
        float priority_value_;

        ExploreState explore_state_;

        bool operator==(const ContactState& other) const;
        bool operator!=(const ContactState& other) const;

        inline bool operator<(const ContactState& other) const {return (this->priority_value_ < other.priority_value_);}

        struct pointer_less
        {
            template <typename T>
            bool operator()(const T& lhs, const T& rhs) const 
            {
                return *lhs < *rhs;
            }
        };

        inline std::shared_ptr<ContactState> getParent() {return parent_;}
        inline const float getF() const {return (g_ + h_);}

    private:
        std::shared_ptr<ContactState> parent_;
};

#endif
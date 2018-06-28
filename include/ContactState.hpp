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
        ContactState(std::shared_ptr<Stance> _initial_stance, Translation3D _initial_com, Vector3D _initial_com_dot, int _num_stance_in_state);
        ContactState(std::shared_ptr<Stance> new_stance, std::shared_ptr<ContactState> _parent, ContactManipulator _move_manip, int _num_stance_in_state);

        // foot orientation projected to flat gruond
        float getLeftHorizontalYaw();
        float getRightHorizontalYaw();
        float getFeetMeanHorizontalYaw();
        TransformationMatrix getFeetMeanTransform();

        const int num_stance_in_state_;

        std::vector<std::shared_ptr<Stance> > stances_vector_;
        Translation3D com_;
        Vector3D com_dot_;

        const bool is_root_;

        ContactManipulator prev_move_manip_;
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

        std::shared_ptr<ContactState> parent_;

        // inline std::shared_ptr<ContactState> getParent() {return parent_;}
        inline const float getF() const {return (g_ + h_);}

    private:

};


namespace std
{
    template <>
    class hash<ContactState>{
        public:
            size_t operator()(const ContactState &contact_state) const
            {
                size_t hash_number;
                for(auto & stance : contact_state.stances_vector_)
                {
                    for(auto & status : stance->ee_contact_status_)
                    {
                        hash_number = hash_number ^ hash<bool>()(status);
                    }

                    for(auto & pose : stance->ee_contact_poses_)
                    {
                        hash_number = hash_number ^ hash<float>()(pose.x_) ^ hash<float>()(pose.y_) ^ hash<float>()(pose.z_)
                                                ^ hash<float>()(pose.roll_) ^ hash<float>()(pose.pitch_) ^ hash<float>()(pose.yaw_);
                    }
                }

                hash_number = hash_number ^ hash<float>()(contact_state.com_(0)) ^ hash<float>()(contact_state.com_(1)) ^ hash<float>()(contact_state.com_(2))
                                        ^ hash<float>()(contact_state.com_dot_(0)) ^ hash<float>()(contact_state.com_dot_(1)) ^ hash<float>()(contact_state.com_dot_(2));

                return hash_number;
            }
    };
}

#endif
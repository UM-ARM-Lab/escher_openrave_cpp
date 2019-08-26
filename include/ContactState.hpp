#ifndef CONTACT_STATE_H
#define CONTACT_STATE_H

#include <momentumopt/dynopt/DynamicsState.hpp>

class CapturePose;
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
        // ContactState():num_stance_in_state_(0), is_root_(false){}
        ContactState(std::vector< std::shared_ptr<Stance> > _initial_stance_vector, Translation3D _initial_com, Vector3D _initial_com_dot, Vector3D _initial_lmom, Vector3D _initial_amom, std::vector<ContactManipulator> _future_move_manips, bool _is_root=true);
        ContactState(std::shared_ptr<Stance> _new_stance, std::shared_ptr<ContactState> _parent, ContactManipulator _move_manip, int _num_stance_in_state, const float _robot_com_z);
        // ~ContactState(){}

        // foot orientation projected to flat gruond
        float getLeftHorizontalYaw(int stance_index=0);
        float getRightHorizontalYaw(int stance_index=0);
        float getFeetMeanHorizontalYaw(int stance_index=0);
        TransformationMatrix getFeetMeanTransform(int stance_index=0);

        void printStateInfo();

        int num_stance_in_state_;
        int depth_;

        std::vector<std::shared_ptr<Stance> > stances_vector_; // 1st stance: current state, 2nd stance: next state
        Translation3D nominal_com_;
        Translation3D com_;
        Vector3D com_dot_; // probably will switch to use lmom and amom to track the linear momentum and angular momentum of the state. right now keep copies of both the lmom and com_dot
        Vector3D lmom_;
        Vector3D amom_;

        Translation3D mean_feet_position_;

        const bool is_root_;

        std::vector<ContactManipulator> future_move_manips_;
        ContactManipulator prev_move_manip_;
        float g_;
        float h_;
        float priority_value_;
        float accumulated_dynamics_cost_;
        float max_manip_x_;

        float prev_disturbance_cost_;

        ExploreState explore_state_;

        momentumopt::DynamicsSequence parent_edge_dynamics_sequence_;

        std::shared_ptr<ContactState> parent_;

        bool operator==(const ContactState& other) const;
        bool operator!=(const ContactState& other) const;

        inline bool operator<(const ContactState& other) const {return (this->priority_value_ < other.priority_value_);}

        // struct pointer_less
        // {
        //     template <typename T>
        //     bool operator()(const T& lhs, const T& rhs) const
        //     {
        //         return *lhs < *rhs;
        //     }
        // };

        // inline std::shared_ptr<ContactState> getParent() {return parent_;}
        inline const float getF() const {return (g_ + h_);}

        bool manip_in_contact(ContactManipulator manip) {return stances_vector_[0]->ee_contact_status_[manip];}

        std::shared_ptr<ContactState> getNoFutureContactState();
        std::shared_ptr<ContactState> getMirrorState(TransformationMatrix& reference_frame);
        std::shared_ptr<ContactState> getCenteredState(TransformationMatrix& reference_frame);
        std::shared_ptr<ContactState> getStandardInputState(DynOptApplication dynamics_optimizer_application);
        std::pair<ContactTransitionCode, std::vector<RPYTF> > getTransitionCodeAndPoses();
        std::pair<OneStepCaptureCode, std::vector<RPYTF> > getOneStepCapturabilityCodeAndPoses();
        std::pair<ZeroStepCaptureCode, std::vector<RPYTF> > getZeroStepCapturabilityCodeAndPoses();

        // std::vector<CapturePose> capture_poses_vector_; // a list of possible capture states during the motion from prev_state to current_state

        std::vector<CapturePose> support_phase_capture_poses_vector_; // a list of possible capture states during the motion from prev_state to current_state
        std::vector<CapturePose> transition_phase_capture_poses_vector_; // a list of possible capture states in current_state
        std::vector<int> transition_phase_capture_poses_prediction_vector_; // a list of capture states prediction

    private:

};

class CapturePose
{
    public:
        CapturePose(ContactManipulator _contact_manip, RPYTF _capture_pose) :
        contact_manip_(_contact_manip),
        capture_pose_(_capture_pose) {};
        ContactManipulator contact_manip_;
        RPYTF capture_pose_;
};


namespace std
{
    template <>
    class hash<ContactState>{
        public:
            size_t operator()(const ContactState &contact_state) const
            {
                size_t hash_number = 0;
                for(auto & stance : contact_state.stances_vector_)
                {
                    for(auto & status : stance->ee_contact_status_)
                    {
                        hash_number ^= hash<bool>()(status) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                    }

                    for(auto & pose : stance->ee_contact_poses_)
                    {
                        hash_number ^= hash<float>()(pose.x_) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                        hash_number ^= hash<float>()(pose.y_) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                        hash_number ^= hash<float>()(pose.z_) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                        hash_number ^= hash<float>()(pose.roll_) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                        hash_number ^= hash<float>()(pose.pitch_) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                        hash_number ^= hash<float>()(pose.yaw_) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                    }
                }

                hash_number ^= hash<float>()(contact_state.com_(0)) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(contact_state.com_(1)) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(contact_state.com_(2)) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(contact_state.com_dot_(0)) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(contact_state.com_dot_(1)) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(contact_state.com_dot_(2)) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(contact_state.lmom_(0)) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(contact_state.lmom_(1)) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(contact_state.lmom_(2)) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(contact_state.amom_(0)) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(contact_state.amom_(1)) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);
                hash_number ^= hash<float>()(contact_state.amom_(2)) + 0x9e3779b9 + (hash_number<<6) + (hash_number>>2);

                return hash_number;
            }
    };
}

#endif
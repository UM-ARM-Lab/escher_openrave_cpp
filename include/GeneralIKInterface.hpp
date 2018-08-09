#ifndef GENERALIKINTERFACE_HPP
#define GENERALIKINTERFACE_HPP

namespace OpenRAVE
{
    using namespace std;
    #include <GeneralIK.h>
}

class GeneralIKInterface
{
    public:
        GeneralIKInterface(OpenRAVE::EnvironmentBasePtr _env, OpenRAVE::RobotBasePtr _robot);
        // ~GeneralIKInterface() {};

        std::pair<bool, std::vector<OpenRAVE::dReal> > solve();
        std::pair<bool, std::vector<OpenRAVE::dReal> > solve(std::map<std::string, OpenRAVE::Transform> _manip_poses, std::vector<OpenRAVE::dReal> _q0,
                                                             std::vector< std::pair<std::string, double> > _support_manips, std::array<OpenRAVE::dReal,3> _com);

        std::array<OpenRAVE::dReal,3>& CenterOfMass() {return com_;}
        std::array<OpenRAVE::dReal,3>& Gravity() {return gravity_;}
        std::map<std::string, OpenRAVE::Transform>& ManipPoses() {return manip_poses_;}
        std::vector< std::pair<std::string, OpenRAVE::dReal> >& SupportManips() {return support_manips_;}
        std::vector<OpenRAVE::dReal>& q0(){return q0_;}
        bool& returnClosest() {return return_closest_;}
        bool& executeMotion() {return execute_motion_;}
        bool& noRotation() {return no_rotation_;}
        bool& exactCoM() {return exact_com_;}
        bool& reuseGIWC() {return reuse_giwc_;}
        OpenRAVE::BalanceMode& balanceMode() {return balance_mode_;}

        void updateNewInput(std::map<std::string, OpenRAVE::Transform> _manip_poses, std::vector<OpenRAVE::dReal> _q0,
                            std::vector< std::pair<std::string, double> > _support_manips, std::array<OpenRAVE::dReal,3> _com);
        void addNewManipPose(std::string _manip_name, OpenRAVE::Transform _manip_transform);
        void addNewContactManip(std::string _manip_name, double _mu);
        void preComputeGIWC();
        void shareGIWC(std::vector<OpenRAVE::dReal>& shared_giwc);
        void resetContactStateRelatedParameters();

        // OpenRAVE objects
        OpenRAVE::EnvironmentBasePtr env_;
        OpenRAVE::RobotBasePtr robot_;
        OpenRAVE::IkSolverBasePtr iksolver_;

    private:
        // GeneralIK parameters
        std::array<OpenRAVE::dReal,3> com_;
        std::array<OpenRAVE::dReal,3> gravity_;
        std::map<std::string, OpenRAVE::Transform> manip_poses_;
        std::vector< std::pair<std::string, OpenRAVE::dReal> > support_manips_;
        std::vector<OpenRAVE::dReal> q0_;
        bool return_closest_;
        bool execute_motion_;
        bool no_rotation_;
        bool exact_com_;
        OpenRAVE::BalanceMode balance_mode_;

        bool reuse_giwc_;
        std::vector<OpenRAVE::dReal> giwc_;

        std::map<std::string, int> manip_name_index_map_;
};

#endif
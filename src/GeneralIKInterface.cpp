
#include "Utilities.hpp"

GeneralIKInterface::GeneralIKInterface(OpenRAVE::EnvironmentBasePtr _env, OpenRAVE::RobotBasePtr _robot) :
env_(_env),
robot_(_robot),
gravity_({0,0,-9.8}),
reuse_giwc_(false),
return_closest_(false),
execute_motion_(false),
no_rotation_(false),
exact_com_(false),
balance_mode_(OpenRAVE::BalanceMode::BALANCE_NONE)
{
    iksolver_ = OpenRAVE::RaveCreateIkSolver(env_,"GeneralIK");
    iksolver_->Init(robot_->GetActiveManipulator());

    for(int i = 0; i < robot_->GetManipulators().size(); i++)
    {
        manip_name_index_map_.insert(std::make_pair(robot_->GetManipulators()[i]->GetName(),i));
    }
}


std::pair<bool, std::vector<OpenRAVE::dReal> > GeneralIKInterface::solve()
{
    robot_->SetActiveDOFValues(q0_);
    std::vector<OpenRAVE::dReal> ik_parameters;

    // number of manipulators whose transforms are specified
    ik_parameters.push_back(manip_poses_.size());

    // the manipulator transforms
    for(auto & manip_pose : manip_poses_)
    {
        ik_parameters.push_back(float(manip_name_index_map_[manip_pose.first])); // manipulator index

        ik_parameters.push_back(manip_pose.second.rot.x);
        ik_parameters.push_back(manip_pose.second.rot.y);
        ik_parameters.push_back(manip_pose.second.rot.z);
        ik_parameters.push_back(manip_pose.second.rot.w);
        ik_parameters.push_back(manip_pose.second.trans.x);
        ik_parameters.push_back(manip_pose.second.trans.y);
        ik_parameters.push_back(manip_pose.second.trans.z);
    }

    // no obstacle avoidance
    ik_parameters.push_back(0);

    // balance mode (0: no balance, 2: GIWC)
    ik_parameters.push_back(balance_mode_);

    if(balance_mode_ == OpenRAVE::BalanceMode::BALANCE_SUPPORT_POLYGON)
    {
        RAVELOG_ERROR("Support Polygon balance mode has not been implemented yet.\n");
        getchar();
    }
    else if(balance_mode_ == OpenRAVE::BalanceMode::BALANCE_GIWC)
    {
        // gravity
        ik_parameters.push_back(gravity_[0]);
        ik_parameters.push_back(gravity_[1]);
        ik_parameters.push_back(gravity_[2]);

        // push_back 0
        ik_parameters.push_back(0); // com specified (set always false)

        // com (will be ignored)
        ik_parameters.push_back(com_[0]);
        ik_parameters.push_back(com_[1]);
        ik_parameters.push_back(com_[2]);

        // will need another way to extract this function from CBiRRT problem
        // // Calculate GIWC
        if(!reuse_giwc_)
        {
            giwc_.clear();
            std::vector<std::string> support_manip_names;
            std::vector<OpenRAVE::dReal> support_mus;
            for(auto & support_manip : support_manips_)
            {
                support_manip_names.push_back(support_manip.first);
                support_mus.push_back(support_manip.second);
            }

            GetGIWC(robot_, support_manip_names, support_mus, giwc_);
        }

        ik_parameters.insert(ik_parameters.end(), giwc_.begin(), giwc_.end());
    }

    // push_back 0
    ik_parameters.push_back(0); // junk, ignored

    // consider contact rotation
    if(no_rotation_)
    {
        ik_parameters.push_back(1);
    }
    else
    {
        ik_parameters.push_back(0);
    }

    // converge to exact COM
    if(exact_com_)
    {
        ik_parameters.push_back(1);
        ik_parameters.push_back(com_[0]);
        ik_parameters.push_back(com_[1]);
        ik_parameters.push_back(com_[2]);
    }
    else
    {
        ik_parameters.push_back(0);
    }

    std::vector<OpenRAVE::dReal> q_final(robot_->GetActiveDOF());
    boost::shared_ptr<std::vector<OpenRAVE::dReal> > q_final_ptr(new std::vector<OpenRAVE::dReal> );

    // auto time_before_ik = std::chrono::high_resolution_clock::now();

    bool found_solution = iksolver_->Solve(OpenRAVE::IkParameterization(), q0_, ik_parameters, false, q_final_ptr);

    // auto time_after_ik = std::chrono::high_resolution_clock::now();

    // float timetaken = std::chrono::duration_cast<std::chrono::microseconds>(time_after_ik - time_before_ik).count()/1000.0;

    if(found_solution)
    {
        q_final = *q_final_ptr.get();

        // RAVELOG_INFO("Solution Found! (%5.1f ms)\n",timetaken);
        if(execute_motion_)
        {
            robot_->SetActiveDOFValues(q_final);
        }
    }
    else
    {
        if(return_closest_)
        {
            q_final = *q_final_ptr.get();
        }

        // RAVELOG_INFO("No IK Solution Found (%5.1f ms)\n",timetaken);
    }

    reuse_giwc_ = true;

    return std::make_pair(found_solution, q_final);
}

std::pair<bool, std::vector<OpenRAVE::dReal> > GeneralIKInterface::solve(std::map<std::string, OpenRAVE::Transform> _manip_poses, std::vector<OpenRAVE::dReal> _q0,
                                                       std::vector< std::pair<std::string, double> > _support_manips, std::array<OpenRAVE::dReal,3> _com)
{
    updateNewInput(_manip_poses, _q0, _support_manips, _com);

    return solve();
}

void GeneralIKInterface::updateNewInput(std::map<std::string, OpenRAVE::Transform> _manip_poses, std::vector<OpenRAVE::dReal> _q0,
                                        std::vector< std::pair<std::string, double> > _support_manips, std::array<OpenRAVE::dReal,3> _com)
{
    manip_poses_ = _manip_poses;
    q0_ = _q0;
    support_manips_ = _support_manips;
    com_ = _com;
}

void GeneralIKInterface::addNewManipPose(std::string _manip_name, OpenRAVE::Transform _manip_transform)
{
    manip_poses_.insert(std::make_pair(_manip_name, _manip_transform));
}


void GeneralIKInterface::addNewContactManip(std::string _manip_name, double _mu)
{
    support_manips_.push_back(std::make_pair(_manip_name, _mu));
}

void GeneralIKInterface::preComputeGIWC()
{
    giwc_.clear();
    std::vector<std::string> support_manip_names;
    std::vector<OpenRAVE::dReal> support_mus;
    for(auto & support_manip : support_manips_)
    {
        support_manip_names.push_back(support_manip.first);
        support_mus.push_back(support_manip.second);
    }

    GetGIWC(robot_, support_manip_names, support_mus, giwc_);
}

void GeneralIKInterface::shareGIWC(std::vector<OpenRAVE::dReal>& shared_giwc)
{
    giwc_ = shared_giwc;
}

void GeneralIKInterface::resetContactStateRelatedParameters()
{
    manip_poses_.clear();
    support_manips_.clear();
}
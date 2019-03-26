#ifndef ESCHERMOTIONPLANNING_HPP
#define ESCHERMOTIONPLANNING_HPP

// #include "Utilities.hpp"

// // OpenRAVE
// #include <openrave/plugin.h>

// using namespace OpenRAVE;

class EscherMotionPlanning : public OpenRAVE::ModuleBase
{
    public:
        EscherMotionPlanning(OpenRAVE::EnvironmentBasePtr penv, std::istream& ss);
        virtual ~EscherMotionPlanning() {}

        bool calculateTraversability(std::ostream& sout, std::istream& sinput);

        bool constructContactRegions(std::ostream& sout, std::istream& sinput);

        bool startTestingTransitionDynamicsOptimization(std::ostream& sout, std::istream& sinput);

        bool startPlanningFromScratch(std::ostream& sout, std::istream& sinput);

        bool startCollectDynamicsOptimizationData(std::ostream& sout, std::istream& sinput);

    private:
        void constructContactPointGrid();
        void constructGroundContactPointGrid();

        void SetActiveRobot(std::string robot_name);

        void parseStructuresCommand(std::istream& sinput);
        void parseFootTransitionModelCommand(std::istream& sinput);
        void parseHandTransitionModelCommand(std::istream& sinput);
        void parseMapGridDimCommand(std::istream& sinput);
        void parseMapGridCommand(std::istream& sinput);
        void parseRobotPropertiesCommand(std::istream& sinput);
        void parseDisturbanceSamplesCommand(std::istream& sinput);
        std::shared_ptr<ContactState> parseContactStateCommand(std::istream& sinput);

        std::map<std::array<int,5>,std::array<float,3> > calculateFootstepTransitionTraversability(std::vector<std::array<int,5>> torso_transitions, std::string motion_mode);
        std::array<float,3> sumFootstepTransitionTraversability(std::array<int,4> correspondence, std::vector< std::array<std::array<int,2>,3> > footstep_window, GridIndices2D torso_indices);

        std::map< std::array<int,3>, std::array<std::array<float,4>,3> > calculateHandTransitionTraversability(std::vector< GridIndices3D > torso_poses);


        OpenRAVE::RobotBasePtr probot_; // Robot object using in the plugin
        OpenRAVE::EnvironmentBasePtr penv_; // Environment object using in the plugin

        std::vector<double>goal_; // Goal coordinate for the planning. [x,y,theta]

        std::vector< std::shared_ptr<TrimeshSurface> > structures_;
        std::map<int, std::shared_ptr<TrimeshSurface> > structures_dict_;

        std::shared_ptr<MapGrid> map_grid_;

        std::shared_ptr<GroundContactPointGrid> feet_contact_point_grid_;

        std::map< std::array<int,3>, std::vector<std::array<std::array<int,2>,3> > > footstep_transition_checking_cells_;
        std::map< std::array<int,3>, std::vector<std::array<std::array<int,2>,3> > > footstep_transition_checking_cells_legs_only_;

        std::vector< std::array<float,3> > foot_transition_model_;
        std::vector< std::array<float,2> > hand_transition_model_;

        std::shared_ptr< DrawingHandler > drawing_handler_;

        std::shared_ptr< RobotProperties > robot_properties_;

        // the general_ik interface
        std::shared_ptr<GeneralIKInterface> general_ik_interface_;

        std::vector< std::pair<Vector6D, float> > disturbance_samples_;

        bool is_parallel_ = false; // a flag to turn or off parallelization. (just for example)
        bool printing_ = false;
};

#endif
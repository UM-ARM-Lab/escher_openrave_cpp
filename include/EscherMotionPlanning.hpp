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

        bool CalculatingTraversability(std::ostream& sout, std::istream& sinput);
        

        bool Planning(std::ostream& sout, std::istream& sinput);

    private:
        void constructContactPointGrid();
        void constructGroundContactPointGrid();

        void SetActiveRobots(std::string robot_name, const std::vector<OpenRAVE::RobotBasePtr>& robots);

        std::map<std::array<int,5>,float> calculateFootstepTransitionTraversability(std::array<float,3> torso_grid_dimensions, std::vector<std::array<int,5>> torso_transitions);
        float sumFootstepTransitionTraversability(std::array<int,4> correspondence, std::vector< std::array<std::array<int,2>,3> > footstep_window, GridIndices2D torso_indices);

        std::map< std::array<int,3>, std::array<float,4> > calculateHandTransitionTraversability(std::array<float,3> torso_grid_dimensions, std::vector< std::array<int,3> > torso_poses);


        OpenRAVE::RobotBasePtr probot_; // Robot object using in the plugin
        OpenRAVE::EnvironmentBasePtr penv_; // Environment object using in the plugin
        
        std::vector<double>goal_; // Goal coordinate for the planning. [x,y,theta]
        
        std::vector<TrimeshSurface> structures_;
        std::shared_ptr<GroundContactPointGrid> feet_contact_point_grid_;

        std::map< std::array<int,3>, std::vector<std::array<std::array<int,2>,3>> > footstep_transition_checking_cells_;

        std::vector< std::array<float,2> > hand_transition_model_;

        std::shared_ptr< DrawingHandler > drawing_handler_;
        
        bool is_parallel_ = false; // a flag to turn or off parallelization. (just for example)
        bool printing_ = false;
};

#endif
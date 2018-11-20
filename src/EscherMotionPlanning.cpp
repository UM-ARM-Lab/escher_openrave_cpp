// #include "EscherMotionPlanning.hpp"
#include "Utilities.hpp"

#include <openrave/plugin.h>
#include <omp.h>

EscherMotionPlanning::EscherMotionPlanning(OpenRAVE::EnvironmentBasePtr penv, std::istream& ss) : OpenRAVE::ModuleBase(penv)
{
    RegisterCommand("StartPlanningFromScratch",boost::bind(&EscherMotionPlanning::startPlanningFromScratch,this,_1,_2),
                    "Start the A* planning process.");

    RegisterCommand("StartTestingTransitionDynamicsOptimization",boost::bind(&EscherMotionPlanning::startTestingTransitionDynamicsOptimization,this,_1,_2),
                    "Start the transition optimization process.");

    RegisterCommand("StartCalculatingTraversability",boost::bind(&EscherMotionPlanning::calculateTraversability,this,_1,_2),
                    "Start calculating traversability.");

    RegisterCommand("StartConstructingContactRegions",boost::bind(&EscherMotionPlanning::constructContactRegions,this,_1,_2),
                    "Start constructing contact regions.");
}

void EscherMotionPlanning::parseStructuresCommand(std::istream& sinput)
{
    int structures_num;

    structures_.clear();
    structures_dict_.clear();

    sinput >> structures_num;

    if(printing_)
    {
        RAVELOG_INFO("Input %d structures:\n",structures_num);
    }

    for(int i = 0; i < structures_num; i++)
    {
        std::string geometry;
        sinput >> geometry;

        // get the kinbody name
        std::string kinbody_name;
        sinput >> kinbody_name;

        int id;
        sinput >> id;

        if(strcmp(geometry.c_str(), "trimesh") == 0)
        {
            Eigen::Vector4f plane_parameters;

            // get the plane parameters
            for(int j = 0; j < 4; j++)
            {
                sinput >> plane_parameters[j];
            }

            // get the vertices
            int vertices_num;
            sinput >> vertices_num;

            Translation3D vertex;
            std::vector<Translation3D> vertices(vertices_num);

            for(int j = 0; j < vertices_num; j++)
            {
                for(int k = 0; k < 3; k++)
                {
                    sinput >> vertex[k];
                }

                vertices[j] = vertex;
            }

            // get the edges
            int edges_num;
            sinput >> edges_num;

            std::vector<std::pair<int, int> > edges(edges_num);
            std::pair<int, int> edge;
            for(int j = 0; j < edges_num; j++)
            {
                sinput >> edge.first;
                sinput >> edge.second;
                edges[j] = edge;
            }

            TrimeshType type;

            std::string tmp_type;
            sinput >> tmp_type;

            if(strcmp(tmp_type.c_str(), "ground") == 0)
            {
                type = TrimeshType::GROUND;
            }
            else
            {
                type = TrimeshType::OTHERS;
            }

            // std::cout<< kinbody_name << std::endl;
            // std::vector< OpenRAVE::KinBodyPtr > bodies;
            // penv_->GetBodies(bodies);
            // std::cout << bodies.size() << std::endl;
            // std::cout << penv_->GetKinBody(kinbody_name)->GetName() << std::endl;
            // std::cout << penv_->GetKinBody(kinbody_name)->GetName().c_str() << std::endl;
            std::shared_ptr<TrimeshSurface> new_surface = std::make_shared<TrimeshSurface>(penv_, kinbody_name, plane_parameters, edges, vertices, type, id);
            structures_.push_back(new_surface);
            structures_dict_.insert(std::make_pair(new_surface->getId(), structures_[structures_.size()-1]));
            Translation3D surface_center = new_surface->getCenter();
            Translation3D surface_normal = new_surface->getNormal();

            if(printing_)
            {
                RAVELOG_INFO("Structure #%d: Trimesh: Center:(%3.2f,%3.2f,%3.2f), Normal:(%3.2f,%3.2f,%3.2f), KinBody Name: %s \n",
                                new_surface->getId(),surface_center[0],surface_center[1],surface_center[2],surface_normal[0],surface_normal[1],surface_normal[2],new_surface->getKinbody()->GetName().c_str());
            }

        }
        else if(strcmp(geometry.c_str(), "box") == 0)
        {
            RAVELOG_WARN("WARNING: Box is not implemented yet.\n");
        }
    }
}

void EscherMotionPlanning::parseFootTransitionModelCommand(std::istream& sinput)
{
    int foot_transition_num;
    sinput >> foot_transition_num;

    foot_transition_model_.resize(foot_transition_num);

    if(printing_)
    {
        RAVELOG_INFO("Load %d foot transitions.\n", foot_transition_num);
    }

    float dx, dy, dtheta;

    for(int i = 0; i < foot_transition_num; i++)
    {
        sinput >> dx;
        sinput >> dy;
        sinput >> dtheta;

        foot_transition_model_[i] = {dx, dy, dtheta};
    }
}

void EscherMotionPlanning::parseHandTransitionModelCommand(std::istream& sinput)
{
    int hand_transition_num;
    sinput >> hand_transition_num;

    hand_transition_model_.resize(hand_transition_num);

    if(printing_)
    {
        RAVELOG_INFO("Load %d hand transitions.\n",hand_transition_num);
    }

    float hand_pitch, hand_yaw;

    for(int i = 0; i < hand_transition_num; i++)
    {
        sinput >> hand_pitch;
        sinput >> hand_yaw;

        hand_transition_model_[i] = {hand_pitch,hand_yaw};
    }
}

void EscherMotionPlanning::parseMapGridDimCommand(std::istream& sinput)
{
    float map_grid_min_x, map_grid_max_x, map_grid_min_y, map_grid_max_y;
    float map_grid_xy_resolution;

    sinput >> map_grid_min_x;
    sinput >> map_grid_max_x;
    sinput >> map_grid_min_y;
    sinput >> map_grid_max_y;
    sinput >> map_grid_xy_resolution;

    map_grid_ = std::make_shared<MapGrid>(map_grid_min_x, map_grid_max_x, map_grid_min_y, map_grid_max_y, map_grid_xy_resolution, TORSO_GRID_ANGULAR_RESOLUTION);
}

void EscherMotionPlanning::parseMapGridCommand(std::istream& sinput)
{
    float map_grid_min_x, map_grid_max_x, map_grid_min_y, map_grid_max_y;
    float map_grid_xy_resolution;

    sinput >> map_grid_min_x;
    sinput >> map_grid_max_x;
    sinput >> map_grid_min_y;
    sinput >> map_grid_max_y;
    sinput >> map_grid_xy_resolution;

    map_grid_ = std::make_shared<MapGrid>(map_grid_min_x, map_grid_max_x, map_grid_min_y, map_grid_max_y, map_grid_xy_resolution, TORSO_GRID_ANGULAR_RESOLUTION);

    for(int i = 0; i < map_grid_->dim_x_; i++)
    {
        for(int j = 0; j < map_grid_->dim_y_; j++)
        {
            sinput >> map_grid_->cell_2D_list_[i][j].height_;

            int foot_ground_projection_surface_id;
            int safe_projection;
            sinput >> safe_projection;
            sinput >> foot_ground_projection_surface_id;

            if(foot_ground_projection_surface_id == -99)
            {
                map_grid_->cell_2D_list_[i][j].foot_ground_projection_ =
                std::make_pair(safe_projection != 0, nullptr);
            }
            else
            {
                map_grid_->cell_2D_list_[i][j].foot_ground_projection_ =
                std::make_pair(safe_projection != 0, structures_dict_[foot_ground_projection_surface_id]);
            }

            int cell_ground_surfaces_num, cell_ground_surface_id;
            sinput >> cell_ground_surfaces_num;

            for(int n = 0; n < cell_ground_surfaces_num; n++)
            {
                sinput >> cell_ground_surface_id;
                map_grid_->cell_2D_list_[i][j].all_ground_structures_.push_back(structures_dict_[cell_ground_surface_id]);
            }

            for(int k = 0; k < map_grid_->dim_theta_; k++)
            {
                std::array<int,3> parent_indices;
                sinput >> map_grid_->cell_3D_list_[i][j][k].parent_indices_[0];
                sinput >> map_grid_->cell_3D_list_[i][j][k].parent_indices_[1];
                sinput >> map_grid_->cell_3D_list_[i][j][k].parent_indices_[2];

                sinput >> map_grid_->cell_3D_list_[i][j][k].g_;
                sinput >> map_grid_->cell_3D_list_[i][j][k].h_;

                int left_hand_checking_surfaces_num, right_hand_checking_surfaces_num;
                int surface_id;

                sinput >> left_hand_checking_surfaces_num;
                for(int n = 0; n < left_hand_checking_surfaces_num; n++)
                {
                    sinput >> surface_id;
                    map_grid_->cell_3D_list_[i][j][k].left_hand_checking_surfaces_.push_back(structures_dict_[surface_id]);
                }

                sinput >> right_hand_checking_surfaces_num;
                for(int n = 0; n < right_hand_checking_surfaces_num; n++)
                {
                    sinput >> surface_id;
                    map_grid_->cell_3D_list_[i][j][k].right_hand_checking_surfaces_.push_back(structures_dict_[surface_id]);
                }
            }
        }
    }


    // the neighbor windows
    for(int i = 0; i < map_grid_->dim_theta_; i++)
    {
        int cell_num;
        sinput >> cell_num;

        int ix, iy;
        std::vector< GridIndices2D > neighbor_window_vector(cell_num);

        for(int j = 0; j < cell_num; j++)
        {
            sinput >> ix;
            sinput >> iy;
            neighbor_window_vector[j] = GridIndices2D({ix,iy});
        }

        map_grid_->left_foot_neighbor_window_.insert(std::make_pair(i,neighbor_window_vector));
    }

    for(int i = 0; i < map_grid_->dim_theta_; i++)
    {
        int cell_num;
        sinput >> cell_num;

        int ix, iy;
        std::vector< GridIndices2D > neighbor_window_vector(cell_num);

        for(int j = 0; j < cell_num; j++)
        {
            sinput >> ix;
            sinput >> iy;
            neighbor_window_vector[j] = GridIndices2D({ix,iy});
        }

        map_grid_->right_foot_neighbor_window_.insert(std::make_pair(i,neighbor_window_vector));
    }

    for(int i = 0; i < map_grid_->dim_theta_; i++)
    {
        int cell_num;
        sinput >> cell_num;

        int ix, iy;
        std::vector< GridIndices2D > neighbor_window_vector(cell_num);

        for(int j = 0; j < cell_num; j++)
        {
            sinput >> ix;
            sinput >> iy;
            neighbor_window_vector[j] = GridIndices2D({ix,iy});
        }

        map_grid_->torso_neighbor_window_.insert(std::make_pair(i,neighbor_window_vector));
    }
}

void EscherMotionPlanning::parseRobotPropertiesCommand(std::istream& sinput)
{
    if(printing_)
    {
        RAVELOG_INFO("Load robot properties.\n");
    }

    int dof_num = probot_->GetDOF();
    std::vector<OpenRAVE::dReal> IK_init_DOF_Values(dof_num);
    std::vector<OpenRAVE::dReal> default_DOF_Values(dof_num);

    float foot_h, foot_w, hand_h, hand_w, robot_z, top_z, shoulder_z, shoulder_w, max_arm_length, min_arm_length, max_stride;

    for(int i = 0; i < dof_num; i++)
    {
        sinput >> IK_init_DOF_Values[i];
    }

    for(int i = 0; i < dof_num; i++)
    {
        sinput >> default_DOF_Values[i];
    }

    sinput >> foot_h;
    sinput >> foot_w;
    sinput >> hand_h;
    sinput >> hand_w;
    sinput >> robot_z;
    sinput >> top_z;
    sinput >> shoulder_z;
    sinput >> shoulder_w;
    sinput >> max_arm_length;
    sinput >> min_arm_length;
    sinput >> max_stride;

    robot_properties_ = std::make_shared<RobotProperties>(probot_, IK_init_DOF_Values, default_DOF_Values,
                                                          foot_h, foot_w, hand_h, hand_w, robot_z, top_z,
                                                          shoulder_z, shoulder_w, max_arm_length, min_arm_length, max_stride);
}

std::shared_ptr<ContactState> EscherMotionPlanning::parseContactStateCommand(std::istream& sinput)
{
    std::vector<RPYTF> ee_poses(ContactManipulator::MANIP_NUM);
    std::array<bool,ContactManipulator::MANIP_NUM> ee_contact_status;

    Translation3D com;
    Vector3D com_dot;

    std::string param;

    for(int i = 0; i < ContactManipulator::MANIP_NUM; i++)
    {
        float x, y, z, roll, pitch, yaw;
        sinput >> x;
        sinput >> y;
        sinput >> z;
        sinput >> roll;
        sinput >> pitch;
        sinput >> yaw;

        ee_poses[i] = RPYTF(x, y, z, roll, pitch, yaw);
        // std::cout << ee_poses[i].x_ << " " << ee_poses[i].y_ << " " << ee_poses[i].z_ << " " << ee_poses[i].roll_ << " " << ee_poses[i].pitch_ << " " << ee_poses[i].yaw_ << " " << std::endl;
    }

    for(int i = 0; i < ContactManipulator::MANIP_NUM; i++)
    {
        sinput >> param;
        if(strcmp(param.c_str(), "0") == 0)
        {
            ee_contact_status[i] = false;
        }
        else
        {
            ee_contact_status[i] = true;
        }
        // std::cout << ee_contact_status[i] << " ";
    }
    // std::cout << std::endl;

    for(int i = 0; i < 3; i++)
    {
        sinput >> com(i);
    }

    for(int i = 0; i < 3; i++)
    {
        sinput >> com_dot(i);
    }

    std::shared_ptr<Stance> stance = std::make_shared<Stance>(ee_poses[0], ee_poses[1], ee_poses[2], ee_poses[3], ee_contact_status);
    auto state = std::make_shared<ContactState>(stance, com, com_dot, 1);

    return state;
}

bool EscherMotionPlanning::calculateTraversability(std::ostream& sout, std::istream& sinput)
{
    penv_ = GetEnv();
    drawing_handler_ = std::make_shared<DrawingHandler>(penv_);
    std::string robot_name;
    std::string param;
    std::vector< std::array<int,5> > torso_transitions;
    float torso_grid_min_x, torso_grid_min_y, torso_grid_resolution;
    float ground_grid_resolution;
    // pass the structures
    // pass the windows
    // pass the grid information

    auto start = std::chrono::high_resolution_clock::now();

    while(!sinput.eof())
    {
        sinput >> param;
        if(!sinput)
        {
            break;
        }

        // if(strcmp(param.c_str(), "robotname") == 0)
        // {
        //     sinput >> robot_name;
        // }

        if(strcmp(param.c_str(), "structures") == 0)
        {
            parseStructuresCommand(sinput);
        }

        if(strcmp(param.c_str(), "map_grid") == 0)
        {
            parseMapGridCommand(sinput);
        }

        if(strcmp(param.c_str(), "transition_footstep_window_cells_legs_only") == 0)
        {
            int footstep_transition_num;
            sinput >> footstep_transition_num;

            if(printing_)
            {
                RAVELOG_INFO("Input %d footstep transitions for legs only case:\n",footstep_transition_num);
            }

            for(int i = 0; i < footstep_transition_num; i++)
            {
                std::array<int,3> torso_transition;
                int footstep_transition_cell_tuple_num;

                sinput >> torso_transition[0];
                sinput >> torso_transition[1];
                sinput >> torso_transition[2];
                sinput >> footstep_transition_cell_tuple_num;

                if(printing_)
                {
                    RAVELOG_INFO("Torso Transition:(%d,%d,%d): %d footstep tuples.\n",torso_transition[0],torso_transition[1],torso_transition[2],footstep_transition_cell_tuple_num);
                }

                std::vector< std::array<std::array<int,2>,3> > footstep_window_cell_tuples(footstep_transition_cell_tuple_num);
                std::array<std::array<int,2>,3> cell_tuple;

                for(int j = 0; j < footstep_transition_cell_tuple_num; j++)
                {
                    sinput >> cell_tuple[0][0];
                    sinput >> cell_tuple[0][1];
                    sinput >> cell_tuple[1][0];
                    sinput >> cell_tuple[1][1];
                    sinput >> cell_tuple[2][0];
                    sinput >> cell_tuple[2][1];

                    footstep_window_cell_tuples[j] = cell_tuple;
                }

                footstep_transition_checking_cells_legs_only_.insert(std::pair< std::array<int,3>, std::vector< std::array<std::array<int,2>,3> > >(torso_transition,footstep_window_cell_tuples));
            }
        }

        if(strcmp(param.c_str(), "transition_footstep_window_cells") == 0)
        {
            int footstep_transition_num;
            sinput >> footstep_transition_num;

            if(printing_)
            {
                RAVELOG_INFO("Input %d footstep transitions:\n",footstep_transition_num);
            }

            for(int i = 0; i < footstep_transition_num; i++)
            {
                std::array<int,3> torso_transition;
                int footstep_transition_cell_tuple_num;

                sinput >> torso_transition[0];
                sinput >> torso_transition[1];
                sinput >> torso_transition[2];
                sinput >> footstep_transition_cell_tuple_num;

                if(printing_)
                {
                    RAVELOG_INFO("Torso Transition:(%d,%d,%d): %d footstep tuples.\n",torso_transition[0],torso_transition[1],torso_transition[2],footstep_transition_cell_tuple_num);
                }

                std::vector< std::array<std::array<int,2>,3> > footstep_window_cell_tuples(footstep_transition_cell_tuple_num);
                std::array<std::array<int,2>,3> cell_tuple;

                for(int j = 0; j < footstep_transition_cell_tuple_num; j++)
                {
                    sinput >> cell_tuple[0][0];
                    sinput >> cell_tuple[0][1];
                    sinput >> cell_tuple[1][0];
                    sinput >> cell_tuple[1][1];
                    sinput >> cell_tuple[2][0];
                    sinput >> cell_tuple[2][1];

                    footstep_window_cell_tuples[j] = cell_tuple;
                }

                footstep_transition_checking_cells_.insert(std::pair< std::array<int,3>, std::vector< std::array<std::array<int,2>,3> > >(torso_transition,footstep_window_cell_tuples));
            }
        }

        if(strcmp(param.c_str(), "torso_transitions") == 0)
        {
            int torso_transition_num;
            sinput >> torso_transition_num;

            if(printing_)
            {
                RAVELOG_INFO("%d torso transitions queried.\n",torso_transition_num);
            }

            for(int i = 0; i < torso_transition_num; i++)
            {
                std::array<int,5> torso_transition;

                sinput >> torso_transition[0];
                sinput >> torso_transition[1];
                sinput >> torso_transition[2];
                sinput >> torso_transition[3];
                sinput >> torso_transition[4];

                torso_transitions.push_back(torso_transition);
            }
        }

        if(strcmp(param.c_str(), "footstep_window_grid_resolution") == 0)
        {
            sinput >> ground_grid_resolution;

            if(printing_)
            {
                RAVELOG_INFO("Footstep window grid resolution=%5.3f.\n",ground_grid_resolution);
            }
        }

        if(strcmp(param.c_str(), "hand_transition_model") == 0)
        {
            parseHandTransitionModelCommand(sinput);
        }

        if(strcmp(param.c_str(), "parallelization") == 0)
        {
            sinput >> param;
            if(strcmp(param.c_str(), "0") == 0)
            {
                is_parallel_ = false;
                if(printing_)
                {
                    RAVELOG_INFO("Don't do parallelization.\n");
                }
            }
            else
            {
                is_parallel_ = true;
                if(printing_)
                {
                    RAVELOG_INFO("Do parallelization.\n");
                }
            }
        }

        if(strcmp(param.c_str(), "printing") == 0)
        {
            printing_ = true;
        }

    }

    feet_contact_point_grid_ = std::make_shared<GroundContactPointGrid>(map_grid_->min_x_,map_grid_->max_x_,
                                                                        map_grid_->min_y_,map_grid_->max_y_,ground_grid_resolution);

    auto after_receving_command = std::chrono::high_resolution_clock::now();

    if(printing_)
    {
        RAVELOG_INFO("Command parsed; now start calculating traversability...\n");
    }

    // calculate the clearance on each surface
    if(printing_)
    {
        RAVELOG_INFO("Now construct the contact point grid on each surface...\n");
    }
    constructContactPointGrid();

    auto after_constructing_point_grid = std::chrono::high_resolution_clock::now();

    // project the ground surface contact points onto the 2D grid
    if(printing_)
    {
        RAVELOG_INFO("Now construct the contact point grid on the 2D ground surface...\n");
    }
    constructGroundContactPointGrid();

    auto after_constructing_ground_point_grid = std::chrono::high_resolution_clock::now();

    // batch calculation of every transition traversability of footsteps
    if(printing_)
    {
        RAVELOG_INFO("Now calculate the footstep contact transition traversability...\n");
    }
    std::map<std::array<int,5>,std::array<float,3> > footstep_traversability;
    footstep_traversability = calculateFootstepTransitionTraversability(torso_transitions,"others");

    std::map<std::array<int,5>,std::array<float,3> > footstep_traversability_legs_only;
    footstep_traversability_legs_only = calculateFootstepTransitionTraversability(torso_transitions,"legs_only");

    auto after_calculating_footstep_transition_traversability = std::chrono::high_resolution_clock::now();

    // batch calculation of every transition traversability of hands
    if(printing_)
    {
        RAVELOG_INFO("Now construct the hand contact transition traversability...\n");
    }
    std::map< std::array<int,3>, std::array<std::array<float,4>,3> > hand_traversability;

    std::set< std::array<int,3> > torso_poses_set;

    for(int i = 0; i < torso_transitions.size(); i++)
    {
        // std::array<int,3> tmp_pose = {torso_transitions[i][0],torso_transitions[i][1],torso_transitions[i][2]};
        torso_poses_set.insert({torso_transitions[i][0],torso_transitions[i][1],torso_transitions[i][2]});
    }

    std::vector< std::array<int,3> > torso_poses(torso_poses_set.begin(), torso_poses_set.end());

    hand_traversability = calculateHandTransitionTraversability(torso_poses);

    auto after_calculating_hand_transition_traversability = std::chrono::high_resolution_clock::now();

    // int a;
    // std::cin >> a;

    sout << footstep_traversability_legs_only.size() << " ";
    for(auto & traversability_pair : footstep_traversability_legs_only)
    {
        sout << traversability_pair.first.at(0) << " "
             << traversability_pair.first.at(1) << " "
             << traversability_pair.first.at(2) << " "
             << traversability_pair.first.at(3) << " "
             << traversability_pair.first.at(4) << " "
             << traversability_pair.second.at(0) << " "
             << traversability_pair.second.at(1) << " "
             << traversability_pair.second.at(2) << " ";
    }

    sout << footstep_traversability.size() << " ";
    for(auto & traversability_pair : footstep_traversability)
    {
        sout << traversability_pair.first.at(0) << " "
             << traversability_pair.first.at(1) << " "
             << traversability_pair.first.at(2) << " "
             << traversability_pair.first.at(3) << " "
             << traversability_pair.first.at(4) << " "
             << traversability_pair.second.at(0) << " "
             << traversability_pair.second.at(1) << " "
             << traversability_pair.second.at(2) << " ";
    }

    sout << hand_traversability.size() << " ";
    for(auto & traversability_pair : hand_traversability)
    {
        sout << traversability_pair.first.at(0) << " "
             << traversability_pair.first.at(1) << " "
             << traversability_pair.first.at(2) << " ";

        for(int i = 0; i < 3; i++)
        {
            sout << traversability_pair.second.at(i).at(0) << " "
                 << traversability_pair.second.at(i).at(1) << " "
                 << traversability_pair.second.at(i).at(2) << " "
                 << traversability_pair.second.at(i).at(3) << " ";
        }
    }

    auto after_output = std::chrono::high_resolution_clock::now();

    std::cout << "Receiving Command: " << std::chrono::duration_cast<std::chrono::milliseconds>(after_receving_command - start).count() << " miliseconds." << std::endl;
    std::cout << "Constructing Contact Point Grid: " << std::chrono::duration_cast<std::chrono::milliseconds>(after_constructing_point_grid - after_receving_command).count() << " miliseconds." << std::endl;
    std::cout << "Constructing Ground Contact Point Grid: " << std::chrono::duration_cast<std::chrono::milliseconds>(after_constructing_ground_point_grid - after_constructing_point_grid).count() << " miliseconds." << std::endl;
    std::cout << "Calculating Footstep Traversability: " << std::chrono::duration_cast<std::chrono::milliseconds>(after_calculating_footstep_transition_traversability - after_constructing_ground_point_grid).count() << " miliseconds." << std::endl;
    std::cout << "Calculating Hand Traversability: " << std::chrono::duration_cast<std::chrono::milliseconds>(after_calculating_hand_transition_traversability - after_calculating_footstep_transition_traversability).count() << " miliseconds." << std::endl;
    std::cout << "Output: " << std::chrono::duration_cast<std::chrono::milliseconds>(after_output - after_calculating_hand_transition_traversability).count() << " miliseconds." << std::endl;

    return true;
}


bool EscherMotionPlanning::constructContactRegions(std::ostream& sout, std::istream& sinput)
{
    penv_ = GetEnv();
    drawing_handler_ = std::make_shared<DrawingHandler>(penv_);

    std::string param;

    bool structures_exist = structures_.size() != 0;

    int s_id, s_id_num;
    std::set<int> structures_id;

    // parsing the inputs
    while(!sinput.eof())
    {
        sinput >> param;
        if(!sinput)
        {
            break;
        }

        if(strcmp(param.c_str(), "structures") == 0)
        {
            if(structures_exist)
            {
                continue;
            }

            parseStructuresCommand(sinput);
        }

        if(strcmp(param.c_str(), "structures_id") == 0)
        {
            sinput >> s_id_num;
            for(int i = 0; i < s_id_num; i++)
            {
                sinput >> s_id;
                structures_id.insert(s_id);
            }
        }

        if(strcmp(param.c_str(), "printing") == 0)
        {
            printing_ = true;
        }

    }

    if(printing_)
    {
        RAVELOG_INFO("Initialize the contact point grid.");
    }

    // initialize the contact point grid
    for(int st_index = 0; st_index < structures_.size(); st_index++)
    {
        std::vector< std::shared_ptr<TrimeshSurface> >::iterator st_it = structures_.begin() + st_index;

        // Initialize contact point grid
        (*st_it)->contact_point_grid_->initializeParameters((*st_it)->getMinProjX(), (*st_it)->getMaxProjX(), (*st_it)->getMinProjY(),
                                                            (*st_it)->getMaxProjY(), 0.03);

        for(int i = 0; i < (*st_it)->contact_point_grid_->contact_point_list_.size(); i++)
        {
            (*st_it)->contact_point_grid_->contact_point_list_[i].clear();
        }
        (*st_it)->contact_point_grid_->contact_point_list_.clear();
    }

    if(printing_)
    {
        RAVELOG_INFO("Construct the contact point grid.");
    }

    // Construct the contact_point_list for each contact point grid of each surface.
    constructContactPointGrid();

    const int num_structures = structures_.size();
    std::vector< std::vector<ContactPoint> > collision_free_contact_points(num_structures);
    std::vector< std::vector<ContactRegion> > contact_regions(num_structures);

    if(printing_)
    {
        RAVELOG_INFO("Extract the projected boundaries from each structure.");
    }

    // Extract the proj boundaries of each structure.
    // #pragma omp parallel for schedule(dynamic) num_threads(16)
    for(std::vector< std::shared_ptr<TrimeshSurface> >::iterator st_it = structures_.begin(); st_it != structures_.end(); st_it++)
    {
        if(structures_id.size() != 0 && structures_id.count((*st_it)->getId()) == 0)
        {
            continue;
        }
        // update projected boundaries
        if(printing_)
        {
            RAVELOG_INFO("Update the projected boundaries.");
        }

        (*st_it)->updateProjBoundaries(structures_);

        if(printing_)
        {
            RAVELOG_INFO("Visualize the projected boundaries.");
        }

        // for(std::vector<Boundary>::iterator bdry_it = (*st_it)->proj_boundaries_.begin(); bdry_it != (*st_it)->proj_boundaries_.end(); bdry_it++)
        // {
        //     std::pair <Translation2D,Translation2D> end_points = bdry_it->getEndPoints();
        //     drawing_handler_->DrawLineSegment((*st_it)->getGlobalPosition(end_points.first)+0.1*(*st_it)->getNormal(), (*st_it)->getGlobalPosition(end_points.second)+0.1*(*st_it)->getNormal(), {1,0,0,1});
        // }

        int k = st_it - structures_.begin();
        Translation2D proj_position;
        float clearance, tmp_clearance, clearance_limit;
        TrimeshType st_type = (*st_it)->getType();

        if(st_type == TrimeshType::GROUND)
        {
            clearance_limit = FOOT_RADIUS;
        }
        else if(st_type == TrimeshType::OTHERS)
        {
            clearance_limit = HAND_RADIUS;
        }

        // Find the clearance of each contact point, and extract points that are far enough from obstacles and boundaries
        std::shared_ptr<SurfaceContactPointGrid> contact_point_grid = (*st_it)->contact_point_grid_;

        std::array<int,2> grid_dimensions = contact_point_grid->getDimensions();
        const int grid_dim_x = grid_dimensions[0];
        const int grid_dim_y = grid_dimensions[1];

        for(int i = 0; i < grid_dim_x; i++)
        {
            for(int j = 0; j < grid_dim_y; j++)
            {
                if(contact_point_grid->contact_point_list_[i][j].isFeasible())
                {
                    // get the position of the grid cell
                    proj_position = contact_point_grid->contact_point_list_[i][j].getProjectedPosition();

                    // get the closest distance from the proj_boundareis
                    clearance = 9999.0;
                    for(std::vector<Boundary>::iterator pb_it = (*st_it)->proj_boundaries_.begin(); pb_it != (*st_it)->proj_boundaries_.end(); pb_it++)
                    {
                        tmp_clearance = pb_it->getDistance(proj_position);
                        if(tmp_clearance < clearance)
                        {
                            clearance = tmp_clearance;
                        }
                    }

                    if(clearance > clearance_limit)
                    {
                        contact_point_grid->contact_point_list_[i][j].setClearance(clearance-clearance_limit);
                        collision_free_contact_points[k].push_back(contact_point_grid->contact_point_list_[i][j]);
                    }
                    else
                    {
                        contact_point_grid->contact_point_list_[i][j].feasible_ = false;
                    }
                }
            }
        }

        std::sort(collision_free_contact_points[k].rbegin(), collision_free_contact_points[k].rend());
        int collision_free_contact_points_num = collision_free_contact_points[k].size();

        // Grow circular contact regions based on the clearance
        for(int i = 0; i < collision_free_contact_points_num; i++)
        {
            if(!collision_free_contact_points[k][i].convered_)
            {
                ContactRegion new_contact_region(collision_free_contact_points[k][i]);

                for(int j = 0; j < collision_free_contact_points_num; j++)
                {
                    if(!collision_free_contact_points[k][j].convered_)
                    {
                        if(new_contact_region.isInsideContactRegion(collision_free_contact_points[k][j].getPosition()))
                        {
                            collision_free_contact_points[k][j].convered_ = true;
                        }
                    }
                }

                contact_regions[k].push_back(new_contact_region);
            }
        }
    }

    int combined_contact_point_num = 0;
    int combined_contact_region_num = 0;
    for(int k = 0; k < num_structures; k++)
    {
        combined_contact_point_num += collision_free_contact_points[k].size();
        combined_contact_region_num += contact_regions[k].size();
    }


    // combine the results from multiple threads
    // std::vector<ContactPoint> combined_collision_free_contact_points(combined_contact_point_num);
    // std::vector<ContactRegion> combined_contact_regions(combined_contact_region_num);
    std::vector<ContactPoint> combined_collision_free_contact_points;
    std::vector<ContactRegion> combined_contact_regions;

    for(int k = 0; k < num_structures; k++)
    {
        combined_collision_free_contact_points.insert(combined_collision_free_contact_points.end(), collision_free_contact_points[k].begin(), collision_free_contact_points[k].end());
        combined_contact_regions.insert(combined_contact_regions.end(), contact_regions[k].begin(), contact_regions[k].end());
    }

    // output the contact regions and contact points
    Translation3D cp_position, cp_normal;
    Translation3D cr_position, cr_normal;
    float cr_radius;
    sout << combined_collision_free_contact_points.size() << " ";
    for(auto & contact_point : combined_collision_free_contact_points)
    {
        cp_position = contact_point.getPosition();
        cp_normal = contact_point.getNormal();
        sout << cp_position(0) << " "
             << cp_position(1) << " "
             << cp_position(2) << " "
             << cp_normal(0) << " "
             << cp_normal(1) << " "
             << cp_normal(2) << " ";
    }

    sout << combined_contact_regions.size() << " ";
    for(auto & contact_region : combined_contact_regions)
    {
        cr_position = contact_region.getPosition();
        cr_normal = contact_region.getNormal();
        cr_radius = contact_region.getRadius();
        sout << cr_position(0) << " "
             << cr_position(1) << " "
             << cr_position(2) << " "
             << cr_normal(0) << " "
             << cr_normal(1) << " "
             << cr_normal(2) << " "
             << cr_radius << " ";
    }

    return true;

}


void EscherMotionPlanning::constructContactPointGrid()
{
    // int dead_1 = 0;
    // int dead_2 = 0;
    // int alive = 0;

    // omp_set_num_threads(16);
    #pragma omp parallel for schedule(dynamic) num_threads(16)
    // for(std::vector<TrimeshSurface>::iterator st_it = structures_.begin(); st_it != structures_.end(); st_it++)
    // {
    for(int st_index = 0; st_index < structures_.size(); st_index++)
    {
        std::vector< std::shared_ptr<TrimeshSurface> >::iterator st_it = structures_.begin() + st_index;

        std::shared_ptr<SurfaceContactPointGrid> surface_contact_point_grid = (*st_it)->contact_point_grid_;

        std::array<int,2> grid_dimensions = surface_contact_point_grid->getDimensions();
        const int grid_dim_x = grid_dimensions[0];
        const int grid_dim_y = grid_dimensions[1];

        Translation3D project_ray = -(*st_it)->getNormal();
        float project_dist = 0.5;

        std::vector< std::pair<Translation2D,std::vector< std::shared_ptr<TrimeshSurface> >::iterator> > checking_structures;

        for(std::vector< std::shared_ptr<TrimeshSurface> >::iterator st_it2 = structures_.begin(); st_it2 != structures_.end(); st_it2++)
        {
            if((*st_it)->getId() != (*st_it2)->getId() && (*st_it)->getType() == (*st_it2)->getType())
            {
                Translation2D struct2_center_in_struct_frame = (*st_it)->projectionPlaneFrame((*st_it2)->getCenter());

                if(struct2_center_in_struct_frame.norm() < (*st_it)->getCircumRadius() + (*st_it2)->getCircumRadius())
                {
                    checking_structures.push_back(std::make_pair(struct2_center_in_struct_frame,st_it2));
                }
            }
        }

        // setting up the contact point grid
        (*st_it)->contact_point_grid_->contact_point_list_.clear();
        for(int i = 0; i < grid_dim_x; i++)
        {
            std::vector<ContactPoint> tmp_contact_point_list;

            for(int j = 0; j < grid_dim_y; j++)
            {
                GridPositions2D cell_center_position = surface_contact_point_grid->indicesToPositions({i,j});
                Translation2D sample_p_2D = gridPositions2DToTranslation2D(cell_center_position);
                Translation3D sample_p_3D = (*st_it)->getGlobalPosition(sample_p_2D);

                bool collision_free = true;

                if((*st_it)->insidePolygonPlaneFrame(sample_p_2D))
                {
                    for(auto cs_it = checking_structures.begin(); cs_it != checking_structures.end(); cs_it++)
                    {
                        Translation2D struct2_center_in_struct_frame = cs_it->first;
                        std::vector< std::shared_ptr<TrimeshSurface> >::iterator st_it2 = cs_it->second;

                        if((sample_p_2D-struct2_center_in_struct_frame).norm() < (*st_it2)->getCircumRadius())
                        {
                            Translation3D proj_origin_p = sample_p_3D + project_dist * (*st_it)->getNormal();
                            Translation3D struct2_proj_p = (*st_it2)->projectionGlobalFrame(proj_origin_p,project_ray);

                            if(isValidPosition(struct2_proj_p) && (*st_it2)->insidePolygon(struct2_proj_p))
                            {
                                float struct2_project_dist = (proj_origin_p-struct2_proj_p).norm();

                                if(struct2_project_dist < project_dist)
                                {
                                    collision_free = false;
                                    // dead_2 += 1;
                                    break;
                                }
                            }
                        }
                    }
                }
                else
                {
                    collision_free = false;
                    // dead_1 += 1;
                }

                if(collision_free)
                {
                    tmp_contact_point_list.push_back(ContactPoint(sample_p_3D,sample_p_2D,-(*st_it)->getNormal(),9999.0,true));
                    // alive += 1;
                }
                else
                {
                    tmp_contact_point_list.push_back(ContactPoint(sample_p_3D,sample_p_2D,-(*st_it)->getNormal(),9999.0,false));
                }

            }

            (*st_it)->contact_point_grid_->contact_point_list_.push_back(tmp_contact_point_list);
        }

        // find the boundaries points in the contact point grid.
        std::vector< std::array<int,2> >  boundary_contact_point_indices;

        for(int i = 0; i < grid_dim_x; i++)
        {
            for(int j = 0; j < grid_dim_y; j++)
            {
                if((*st_it)->contact_point_grid_->contact_point_list_[i][j].feasible_)
                {
                    // points in the index limit are guaranteed to be boundary points
                    if(i == 0 || i == grid_dim_x-1 || j == 0 || j == grid_dim_y-1)
                    {
                        boundary_contact_point_indices.push_back({i,j});
                        (*st_it)->contact_point_grid_->contact_point_list_[i][j].setClearance(0);
                        continue;
                    }

                    bool is_boundary_point = false;

                    for(int i2 = i-1; i2 <= i+1; i2++)
                    {
                        for(int j2 = j-1; j2 <= j+1; j2++)
                        {
                            if(!(*st_it)->contact_point_grid_->contact_point_list_[i2][j2].feasible_)
                            {
                                boundary_contact_point_indices.push_back({i,j});
                                (*st_it)->contact_point_grid_->contact_point_list_[i][j].setClearance(0);
                                is_boundary_point = true;
                                break;
                            }
                        }

                        if(is_boundary_point)
                        {
                            break;
                        }
                    }
                }
            }
        }

        // calculate the clearance for each contact point
        for(int i = 0; i < grid_dim_x; i++)
        {
            for(int j = 0; j < grid_dim_y; j++)
            {
                if((*st_it)->contact_point_grid_->contact_point_list_[i][j].isFeasible() &&
                   (*st_it)->contact_point_grid_->contact_point_list_[i][j].getClearance() != 0)
                {
                    float dist_to_boundary_point;
                    for(std::vector< std::array<int,2> >::iterator bcpi_it = boundary_contact_point_indices.begin(); bcpi_it != boundary_contact_point_indices.end(); bcpi_it++)
                    {
                        dist_to_boundary_point = hypot(float(bcpi_it->at(0)-i),float(bcpi_it->at(1)-j)) * (*st_it)->contact_point_grid_->getResolution();
                        if(dist_to_boundary_point < (*st_it)->contact_point_grid_->contact_point_list_[i][j].getClearance())
                        {
                            (*st_it)->contact_point_grid_->contact_point_list_[i][j].setClearance(dist_to_boundary_point);
                        }
                    }
                }

                // if(st_it->contact_point_grid_->contact_point_list_[i][j].isFeasible())
                // {
                //     ContactPoint cp = st_it->contact_point_grid_->contact_point_list_[i][j];
                //     std::array<float,4> color = HSVToRGB({((1-cp.getTotalScore(ContactType::FOOT, GLOBAL_NEGATIVE_Z))*2.0/3.0)*360,1,1,1});
                //     // std::array<float,4> color = HSVToRGB({((1-cp.getClearance()/0.2)*2.0/3.0)*360,1,1,1});

                //     drawing_handler_->DrawLineSegment(cp.getPosition(), cp.getPosition()-0.02*cp.getNormal(), color);
                // }
            }
        }
    }

    // std::cout<<"Dead 1:" << dead_1 <<", Dead 2:" << dead_2<<", Alive:"<< alive << std::endl;

}

void EscherMotionPlanning::constructGroundContactPointGrid()
{
    // filter out interesting structures
    std::vector< std::vector< std::shared_ptr<TrimeshSurface> >::iterator > feet_contact_structures;
    for(std::vector< std::shared_ptr<TrimeshSurface> >::iterator st_it = structures_.begin(); st_it != structures_.end(); st_it++)
    {
        std::array<int,2> surface_grid_dim = (*st_it)->contact_point_grid_->getDimensions();
        // if(st_it->getType() == TrimeshType::GROUND && st_it->getId() != 99999 && st_it->getId() != 49999 &&
        //    surface_grid_dim[0] > 1 && surface_grid_dim[1] > 1)
        if((*st_it)->getType() == TrimeshType::GROUND &&
           surface_grid_dim[0] > 1 && surface_grid_dim[1] > 1)
        {
            feet_contact_structures.push_back(st_it);
        }

    }

    std::array<int,2> feet_contact_point_grid_dim = feet_contact_point_grid_->getDimensions();
    for(int i = 0; i < feet_contact_point_grid_dim[0]; i++)
    {
        std::vector<std::array<float,3> > tmp_score_list(feet_contact_point_grid_dim[1],{0,0,0});

        for(int j = 0; j < feet_contact_point_grid_dim[1]; j++)
        {
            GridPositions2D cell_position = feet_contact_point_grid_->indicesToPositions({i,j});
            float cell_x = cell_position[0];
            float cell_y = cell_position[1];

            for(int k = 0; k < feet_contact_structures.size(); k++)
            {
                std::vector< std::shared_ptr<TrimeshSurface> >::iterator st_it = feet_contact_structures[k];
                Translation3D surface_center = (*st_it)->getCenter();

                if(hypot(surface_center[0]-cell_x,surface_center[1]-cell_y) < (*st_it)->getCircumRadius())
                {
                    GridPositions2D proj_feet_contact_point_positions = translation2DToGridPositions2D((*st_it)->projectionPlaneFrame(Translation3D(cell_x,cell_y,9999.0),GLOBAL_NEGATIVE_Z));

                    std::array<int,2> surface_contact_grid_dim = (*st_it)->contact_point_grid_->getDimensions();

                    // Check if the projection is inside the grid
                    if((*st_it)->contact_point_grid_->insideGrid(proj_feet_contact_point_positions))
                    {
                        GridIndices2D proj_feet_contact_point_indices = (*st_it)->contact_point_grid_->positionsToIndices(proj_feet_contact_point_positions);

                        if(proj_feet_contact_point_indices[0] < surface_contact_grid_dim[0]-1 &&
                           proj_feet_contact_point_indices[1] < surface_contact_grid_dim[1]-1)
                        {

                            int pfcp_ix = proj_feet_contact_point_indices[0];
                            int pfcp_iy = proj_feet_contact_point_indices[1];

                            ContactPoint p1 = (*st_it)->contact_point_grid_->contact_point_list_[pfcp_ix][pfcp_iy];
                            ContactPoint p2 = (*st_it)->contact_point_grid_->contact_point_list_[pfcp_ix+1][pfcp_iy];
                            ContactPoint p3 = (*st_it)->contact_point_grid_->contact_point_list_[pfcp_ix][pfcp_iy+1];
                            ContactPoint p4 = (*st_it)->contact_point_grid_->contact_point_list_[pfcp_ix+1][pfcp_iy+1];

                            if(p1.isFeasible() && p2.isFeasible() && p3.isFeasible() && p4.isFeasible())
                            {
                                float p1_clearance_score = p1.getClearanceScore(ContactType::FOOT);
                                float p2_clearance_score = p2.getClearanceScore(ContactType::FOOT);
                                float p3_clearance_score = p3.getClearanceScore(ContactType::FOOT);
                                float p4_clearance_score = p4.getClearanceScore(ContactType::FOOT);

                                float p1_orientation_score = p1.getOrientationScore(GLOBAL_NEGATIVE_Z);
                                float p2_orientation_score = p2.getOrientationScore(GLOBAL_NEGATIVE_Z);
                                float p3_orientation_score = p3.getOrientationScore(GLOBAL_NEGATIVE_Z);
                                float p4_orientation_score = p4.getOrientationScore(GLOBAL_NEGATIVE_Z);

                                float p1_total_score = p1.getTotalScore(ContactType::FOOT, GLOBAL_NEGATIVE_Z);
                                float p2_total_score = p2.getTotalScore(ContactType::FOOT, GLOBAL_NEGATIVE_Z);
                                float p3_total_score = p3.getTotalScore(ContactType::FOOT, GLOBAL_NEGATIVE_Z);
                                float p4_total_score = p4.getTotalScore(ContactType::FOOT, GLOBAL_NEGATIVE_Z);

                                GridPositions2D cell_center_positions = (*st_it)->contact_point_grid_->indicesToPositions(proj_feet_contact_point_indices);

                                float lx1 = proj_feet_contact_point_positions[0] - (cell_center_positions[0] - 0.5*(*st_it)->contact_point_grid_->getResolution());
                                float lx2 = (*st_it)->contact_point_grid_->getResolution() - lx1;
                                float ly1 = proj_feet_contact_point_positions[1] - (cell_center_positions[1] - 0.5*(*st_it)->contact_point_grid_->getResolution());
                                float ly2 = (*st_it)->contact_point_grid_->getResolution() - ly1;

                                tmp_score_list[j][0] = (*st_it)->contact_point_grid_->getInterpolatedScore({p1_clearance_score,p2_clearance_score,p3_clearance_score,p4_clearance_score}, {lx1,lx2,ly1,ly2});
                                tmp_score_list[j][1] = (*st_it)->contact_point_grid_->getInterpolatedScore({p1_orientation_score,p2_orientation_score,p3_orientation_score,p4_orientation_score}, {lx1,lx2,ly1,ly2});
                                tmp_score_list[j][2] = (*st_it)->contact_point_grid_->getInterpolatedScore({p1_total_score,p2_total_score,p3_total_score,p4_total_score}, {lx1,lx2,ly1,ly2});

                                break; // the contact points on the ground structures does not overlap
                            }
                        }
                    }
                }
            }
        }

        feet_contact_point_grid_->score_cell_list_.push_back(tmp_score_list);

    }

    // for(int i = 0; i < feet_contact_point_grid_dim[0]; i++)
    // {
    //     for(int j = 0; j < feet_contact_point_grid_dim[1]; j++)
    //     {
    //         std::array<float,4> color = HSVToRGB({((1-feet_contact_point_grid_->score_cell_list_[i][j])*2.0/3.0)*360,1,1,1});
    //         GridPositions2D cell_position = feet_contact_point_grid_->indicesToPositions({i,j});
    //         Translation3D drawing_cell_position(cell_position[0],cell_position[1],0.1);

    //         drawing_handler_->DrawLineSegment(drawing_cell_position, drawing_cell_position-0.02*GLOBAL_NEGATIVE_Z, color);
    //     }
    // }
}

std::map<std::array<int,5>,std::array<float,3> > EscherMotionPlanning::calculateFootstepTransitionTraversability(std::vector< std::array<int,5> > transitions, std::string motion_mode)
{
    std::map<std::array<int,5>,std::array<float,3> > traversability_map;

	// int ix1, iy1, itheta1;
	// int ix2, iy2;

    // float x1, y1;
	// int theta1;

	// printf("torso grid dimension: (%5.3f,%5.3f).\n",torso_grid_min_x,torso_grid_min_y);
    // printf("footstep grid dimension: (%5.3f,%5.3f).\n",footstep_window_grid_min_x,footstep_window_grid_min_y);
    std::array<float,3> default_traversability = {0,0,0};

    for(int i = 0; i < transitions.size(); i++)
    {
        traversability_map.insert(std::make_pair(transitions[i],default_traversability));
    }

    // omp_set_num_threads(16);
    #pragma omp parallel for schedule(dynamic) num_threads(16)
	for(int i = 0; i < transitions.size(); i++)
	{
		int ix1 = transitions[i][0];
		int iy1 = transitions[i][1];
		int itheta1 = transitions[i][2];

		int ix2 = transitions[i][3];
		int iy2 = transitions[i][4];

        // printf("From (%d,%d,%d) to (%d,%d): \n",ix1,iy1,itheta1,ix2,iy2);

        GridPositions3D torso_grid_position1 = map_grid_->indicesToPositions({ix1,iy1,itheta1});

        float x1 = torso_grid_position1[0];
        float y1 = torso_grid_position1[1];
        int theta1 = int(torso_grid_position1[2]);

        // printf("(x1,y1,theta1)=(%5.5f,%5.5f,%d)\n",x1,y1,theta1);

        std::array<int,4> correspondence;
        int window_theta, window_dix, window_diy;

		window_theta = (theta1+360) % 90;
		window_dix = 0;
		window_diy = 0;

		if(theta1 >= 0 && theta1 < 90)
		{
			window_dix = ix2-ix1;
			window_diy = iy2-iy1;
			correspondence[0] = 1; correspondence[1] = 0; correspondence[2] = 1; correspondence[3] = 1;
		}
		else if(theta1 >= 90 && theta1 < 180)
		{
			window_dix = iy2-iy1;
			window_diy = -(ix2-ix1);
			correspondence[0] = -1; correspondence[1] = 1; correspondence[2] = 1; correspondence[3] = 0;
		}
		else if(theta1 >= -180 && theta1 < -90)
		{
			window_dix = -(ix2-ix1);
			window_diy = -(iy2-iy1);
			correspondence[0] = -1; correspondence[1] = 0; correspondence[2] = -1; correspondence[3] = 1;
		}
		else if(theta1 >= -90 && theta1 < 0)
		{
			window_dix = -(iy2-iy1);
			window_diy = ix2-ix1;
			correspondence[0] = 1; correspondence[1] = 1; correspondence[2] = -1; correspondence[3] = 0;
		}

        GridIndices2D torso_indices = feet_contact_point_grid_->positionsToIndices({x1,y1});

		// torso_ix = int((x1-footstep_window_grid_min_x)/FOOTSTEP_WINDOW_GRID_RESOLUTION);
        // torso_iy = int((y1-footstep_window_grid_min_y)/FOOTSTEP_WINDOW_GRID_RESOLUTION);

		// printf("Window key:(%d,%d,%d), correspondence = [%d,%d,%d,%d].\n",window_dix,window_diy,window_theta,correspondence[0],correspondence[1],correspondence[2],correspondence[3]);
		// printf("footstep_window_grid_min_xy:(%5.3f,%5.3f).\n",footstep_window_grid_min_x,footstep_window_grid_min_y);
		// printf("Torso:(%d,%d)\n",torso_ix,torso_iy);

        std::array<int,3> window_key = {window_dix,window_diy,window_theta};

        std::vector< std::array<std::array<int,2>,3> > footstep_window;

        if(strcmp(motion_mode.c_str(),"legs_only") == 0)
        {
            footstep_window = footstep_transition_checking_cells_legs_only_.find(window_key)->second;
        }
        else
        {
            footstep_window = footstep_transition_checking_cells_.find(window_key)->second;
        }

		// printf("Now entering score calculation.\n");
        // float traversability = sumFootstepTransitionTraversability(correspondence, footstep_window, torso_indices);
        std::array<float,3> traversability = sumFootstepTransitionTraversability(correspondence, footstep_window, torso_indices);
        // traversability_map.insert(std::make_pair(transitions[i],traversability));

        std::map<std::array<int,5>,std::array<float,3> >::iterator tm_it = traversability_map.find(transitions[i]);
        tm_it->second = traversability;

        // getchar();
    }

	return traversability_map;
}

std::array<float,3> EscherMotionPlanning::sumFootstepTransitionTraversability(std::array<int,4> correspondence, std::vector< std::array<std::array<int,2>,3> > footstep_window, GridIndices2D torso_indices)
{
	int x_sign = correspondence[0];
	int x_addition_index = correspondence[1];
	int y_sign = correspondence[2];
	int y_addition_index = correspondence[3];

	// printf("[%d,%d,%d,%d].%d.\n",x_sign,x_addition_index,y_sign,y_addition_index,footstep_num);

    std::array<float,3> footstep_window_score = {0,0,0};

	for(int i = 0; i < footstep_window.size(); i++)
	{
		int left_cell_x_global = torso_indices[0] + x_sign * footstep_window[i][0][x_addition_index];
		int left_cell_y_global = torso_indices[1] + y_sign * footstep_window[i][0][y_addition_index];
		int right_cell_x_global = torso_indices[0] + x_sign * footstep_window[i][1][x_addition_index];
		int right_cell_y_global = torso_indices[1] + y_sign * footstep_window[i][1][y_addition_index];
		int footstep_cell_x_global = torso_indices[0] + x_sign * footstep_window[i][2][x_addition_index];
		int footstep_cell_y_global = torso_indices[1] + y_sign * footstep_window[i][2][y_addition_index];

        GridIndices2D left_cell_indices = {left_cell_x_global,left_cell_y_global};
        GridIndices2D right_cell_indices = {right_cell_x_global,right_cell_y_global};
        GridIndices2D footstep_cell_indices = {footstep_cell_x_global,footstep_cell_y_global};

        if(feet_contact_point_grid_->insideGrid(left_cell_indices) &&
           feet_contact_point_grid_->insideGrid(right_cell_indices) &&
           feet_contact_point_grid_->insideGrid(footstep_cell_indices))
        {
            for(int j = 0; j < 3; j++)
            {
                footstep_window_score[j] += (feet_contact_point_grid_->score_cell_list_[footstep_cell_x_global][footstep_cell_y_global][j]
                                           * feet_contact_point_grid_->score_cell_list_[left_cell_x_global][left_cell_y_global][j]
                                           * feet_contact_point_grid_->score_cell_list_[right_cell_x_global][right_cell_y_global][j]);
            }
        }
	}

	return footstep_window_score;
}

std::map< std::array<int,3>, std::array<std::array<float,4>,3> > EscherMotionPlanning::calculateHandTransitionTraversability(std::vector< GridIndices3D > torso_poses)
{
    // given torso pose, and the environment structures find projection score of each hand transition model
    std::map< std::array<int,3>, std::array<std::array<float,4>,3> > hand_transition_traversability;

    // std::array<float,4> default_traversability = {0,0,0,0}
    // std::array<std::array<float,4>,3> combined_default_traversability = {{default_traversability,default_traversability,default_traversability}};
    std::array<std::array<float,4>,3> default_traversability = {{{0,0,0,0},{0,0,0,0},{0,0,0,0}}};
    for(int i = 0; i < torso_poses.size(); i++)
    {
        // std::cout << torso_poses[i][0] << ' ' << torso_poses[i][1] << ' ' << torso_poses[i][2] << std::endl;
        hand_transition_traversability.insert(std::make_pair(torso_poses[i],default_traversability));
    }

    std::vector<std::string> feature_types = {"clearance","orientation","total"};

	// omp_set_num_threads(16);
	#pragma omp parallel for schedule(dynamic) num_threads(16)
    for(int i = 0; i < torso_poses.size(); i++)
    {
        int ix = torso_poses[i][0];
        int iy = torso_poses[i][1];
        int itheta = torso_poses[i][2];

        GridPositions3D torso_position = map_grid_->indicesToPositions({ix,iy,itheta});

        float x = torso_position[0];
		float y = torso_position[1];
        float z = 0.0; // assume the robot height is 0, which is clearly not precise
		float theta = torso_position[2];
        float theta_rad = theta * DEG2RAD;

        std::array<std::array<float,4>,3> traversability_scores = {{{0,0,0,0},{0,0,0,0},{0,0,0,0}}};

        // determine the height of the robot
        int left_counted_cell_num = 0;
        float left_mean_height = 0;
        int right_counted_cell_num = 0;
        float right_mean_height = 0;

        for(std::vector< GridIndices2D >::iterator lfn_it = map_grid_->left_foot_neighbor_window_[itheta].begin();
            lfn_it != map_grid_->left_foot_neighbor_window_[itheta].end(); lfn_it++)
        {
            int lfix = ix + lfn_it->at(0);
            int lfiy = iy + lfn_it->at(1);

            if(map_grid_->insideGrid(GridIndices3D({lfix,lfiy,itheta})) &&
               map_grid_->cell_2D_list_[lfix][lfiy].height_ > -0.5 &&
               map_grid_->cell_2D_list_[lfix][lfiy].height_ < 0.5)
            {
                left_mean_height = left_mean_height + map_grid_->cell_2D_list_[lfix][lfiy].height_;
                left_counted_cell_num = left_counted_cell_num + 1;
            }
        }

        if(left_counted_cell_num == 0)
        {
            // RAVELOG_WARN("WARNING: env_transition_feature_calculation receive a query with no standing position for left foot.");
            std::map< std::array<int,3>, std::array<std::array<float,4>,3> >::iterator htt_it = hand_transition_traversability.find(torso_poses[i]);
            htt_it->second = {{{0,0,0,0},{0,0,0,0},{0,0,0,0}}};
            continue;
        }

        for(std::vector< GridIndices2D >::iterator rfn_it = map_grid_->right_foot_neighbor_window_[itheta].begin();
            rfn_it != map_grid_->right_foot_neighbor_window_[itheta].end(); rfn_it++)
        {
            int rfix = ix + rfn_it->at(0);
            int rfiy = iy + rfn_it->at(1);

            if(map_grid_->insideGrid(GridIndices3D({rfix,rfiy,itheta})) &&
               map_grid_->cell_2D_list_[rfix][rfiy].height_ > -0.5 &&
               map_grid_->cell_2D_list_[rfix][rfiy].height_ < 0.5)
            {
                right_mean_height = right_mean_height + map_grid_->cell_2D_list_[rfix][rfiy].height_;
                right_counted_cell_num = right_counted_cell_num + 1;
            }
        }

        if(right_counted_cell_num == 0)
        {
            // RAVELOG_WARN("WARNING: env_transition_feature_calculation receive a query with no standing position for left foot.");
            std::map< std::array<int,3>, std::array<std::array<float,4>,3> >::iterator htt_it = hand_transition_traversability.find(torso_poses[i]);
            htt_it->second = {{{0,0,0,0},{0,0,0,0},{0,0,0,0}}};
            continue;
        }

        z = (left_mean_height + right_mean_height)/2.0;

        // hand_contact_structures = left_contact_structures
        // hand_contact_structures = right_contact_structures

        for(int j = 0; j < ARM_MANIPULATORS.size(); j++)
        {
            ContactManipulator manip = ARM_MANIPULATORS[j];
            std::array<float,3> relative_shoulder_position;

            if(manip == ContactManipulator::L_ARM)
            {
                relative_shoulder_position = {0,SHOULDER_W/2.0,SHOULDER_Z};
            }
            else if(manip == ContactManipulator::R_ARM)
            {
                relative_shoulder_position = {0,-SHOULDER_W/2.0,SHOULDER_Z};
            }

            float current_shoulder_x = x + cos(theta_rad) * relative_shoulder_position[0] - sin(theta_rad) * relative_shoulder_position[1];
            float current_shoulder_y = y + sin(theta_rad) * relative_shoulder_position[0] + cos(theta_rad) * relative_shoulder_position[1];
            float current_shoulder_z = z + relative_shoulder_position[2];

            Translation3D current_shoulder_position(current_shoulder_x,current_shoulder_y,current_shoulder_z);

            for(std::vector< std::array<float,2> >::iterator ht_it = hand_transition_model_.begin(); ht_it != hand_transition_model_.end(); ht_it++)
            {
                std::array<float,2> current_arm_orientation = {0,0};

                if(manip == ContactManipulator::L_ARM)
                {
                    current_arm_orientation[0] = theta + 90.0 - ht_it->at(0);
                }
                else if(manip == ContactManipulator::R_ARM)
                {
                    current_arm_orientation[0] = theta - 90.0 + ht_it->at(0);
                }
                current_arm_orientation[1] = ht_it->at(1);

                float cos_pitch = cos(current_arm_orientation[0]*DEG2RAD);
                float sin_pitch = sin(current_arm_orientation[0]*DEG2RAD);
                float cos_yaw = cos(current_arm_orientation[1]*DEG2RAD);
                float sin_yaw = sin(current_arm_orientation[1]*DEG2RAD);

                Translation3D contact_direction(cos_pitch*cos_yaw,sin_pitch*cos_yaw,sin_yaw);

                float proj_dist = 9999.0;
                Translation3D proj_point;
                std::vector< std::shared_ptr<TrimeshSurface> >::iterator contact_st;

                for(std::vector< std::shared_ptr<TrimeshSurface> >::iterator st_it = structures_.begin(); st_it != structures_.end(); st_it++)
                {
                    float shoulder_struct_center_dist = ((*st_it)->getCenter() - current_shoulder_position).norm();

                    if((*st_it)->getType() == TrimeshType::OTHERS && shoulder_struct_center_dist <= MAX_ARM_LENGTH + (*st_it)->getCircumRadius())
                    {
                        Translation3D tmp_proj_point = (*st_it)->projectionGlobalFrame(current_shoulder_position, contact_direction);
                        float tmp_proj_dist = (tmp_proj_point-current_shoulder_position).norm();

                        if(isValidPosition(tmp_proj_point) && (*st_it)->insidePolygon(tmp_proj_point) && tmp_proj_dist < proj_dist)
                        {
                            proj_dist = tmp_proj_dist;
                            proj_point = tmp_proj_point;
                            contact_st = st_it;
                        }
                    }
                }

                if(proj_dist > MIN_ARM_LENGTH && proj_dist < MAX_ARM_LENGTH)
                {
                    GridPositions2D proj_point_plane_frame = translation2DToGridPositions2D((*contact_st)->projectionPlaneFrame(proj_point));

                    for(int k = 0; k < 3; k++)
                    {
                        float score = (*contact_st)->contact_point_grid_->getScore(proj_point_plane_frame,ContactType::HAND,contact_direction,feature_types[k]);

                        if(manip == ContactManipulator::L_ARM && ht_it->at(0) > -20)
                        {
                            traversability_scores[k][0] += score;
                        }

                        if(manip == ContactManipulator::L_ARM && ht_it->at(0) < 20)
                        {
                            traversability_scores[k][1] += score;
                        }

                        if(manip == ContactManipulator::R_ARM && ht_it->at(0) > -20)
                        {
                            traversability_scores[k][2] += score;
                        }

                        if(manip == ContactManipulator::R_ARM && ht_it->at(0) < 20)
                        {
                            traversability_scores[k][3] += score;
                        }

                    }
                }
            }
        }

        // hand_transition_traversability.insert(std::make_pair(torso_poses[i],traversability_scores));

        std::map< std::array<int,3>, std::array<std::array<float,4>,3> >::iterator htt_it = hand_transition_traversability.find(torso_poses[i]);
        htt_it->second = traversability_scores;
    }



    // left_contact_structures_id = self.grid_to_left_hand_checking_surface_indices((ix1,iy1,itheta1))
    // right_contact_structures_id = self.grid_to_right_hand_checking_surface_indices((ix1,iy1,itheta1))

    // left_contact_structures = []
    // right_contact_structures = []

    // for i in left_contact_structures_id:
    //     left_contact_structures.append(structures_dict[i])

    // for i in right_contact_structures_id:
    //     right_contact_structures.append(structures_dict[i])

    return hand_transition_traversability;

}

bool EscherMotionPlanning::startTestingTransitionDynamicsOptimization(std::ostream& sout, std::istream& sinput)
{
    penv_ = GetEnv();

    std::string param;

    std::shared_ptr<ContactState> initial_state;
    std::shared_ptr<ContactState> second_state;
    std::shared_ptr<ContactState> dummy_second_state;

    // RAVELOG_INFO("Parse commands and initialize variables...\n");

    while(!sinput.eof())
    {
        sinput >> param;
        if(!sinput)
        {
            break;
        }

        if(strcmp(param.c_str(), "initial_state") == 0)
        {
            initial_state = parseContactStateCommand(sinput);
        }
        else if(strcmp(param.c_str(), "second_state") == 0)
        {
            dummy_second_state = parseContactStateCommand(sinput);
        }
    }

    // if(initial_state->stances_vector_[0]->ee_contact_status_[int(ContactManipulator::L_ARM)])
    // {
    //     second_state = std::make_shared<ContactState>(dummy_second_state->stances_vector_[0], initial_state, ContactManipulator::L_ARM, 1, 1.0);
    // }
    // else if(initial_state->stances_vector_[0]->ee_contact_status_[int(ContactManipulator::R_ARM)])
    // {
    //     second_state = std::make_shared<ContactState>(dummy_second_state->stances_vector_[0], initial_state, ContactManipulator::R_ARM, 1, 1.0);
    // }

    second_state = std::make_shared<ContactState>(dummy_second_state->stances_vector_[0], initial_state, ContactManipulator::L_ARM, 1, 1.0);

    srand (time(NULL));

    // TransformationMatrix rand_transform = XYZRPYToSE3(RPYTF((std::rand()%100)/100.0, (std::rand()%100)/100.0, (std::rand()%100)/100.0, 0, 0, float(std::rand()%180)));
    TransformationMatrix rand_transform  = Eigen::Matrix4f::Identity();
    // TransformationMatrix rand_transform;

    // rand_transform << 0.882948, -0.469472,         0,      0.98,
    //                   0.469472,  0.882948,        -0,      0.97,
    //                          0,         0,         1,      0.98,
    //                          0,         0,         0,         1;
    RotationMatrix rand_rotation = rand_transform.block(0,0,3,3);

    // std::cout << rand_transform << std::endl;

    for(unsigned int i = 0; i < ContactManipulator::MANIP_NUM; i++)
    {
        if(initial_state->stances_vector_[0]->ee_contact_status_[i])
        {
            RPYTF original_pose = initial_state->stances_vector_[0]->ee_contact_poses_[i];
            RPYTF transformed_pose = SE3ToXYZRPY(rand_transform * XYZRPYToSE3(original_pose));
            initial_state->stances_vector_[0]->ee_contact_poses_[i] = transformed_pose;
        }

        if(second_state->stances_vector_[0]->ee_contact_status_[i])
        {
            RPYTF original_pose = second_state->stances_vector_[0]->ee_contact_poses_[i];
            RPYTF transformed_pose = SE3ToXYZRPY(rand_transform * XYZRPYToSE3(original_pose));
            second_state->stances_vector_[0]->ee_contact_poses_[i] = transformed_pose;
        }
    }

    initial_state->com_ = (rand_transform * initial_state->com_.homogeneous()).block(0,0,3,1);
    initial_state->com_dot_ = rand_rotation * initial_state->com_dot_;

    // std::cout << "=====initial state=====" << std::endl;
    // for(unsigned int i = 0; i < ContactManipulator::MANIP_NUM; i++)
    // {
    //     if(initial_state->stances_vector_[0]->ee_contact_status_[i])
    //     {
    //         RPYTF transformed_pose = initial_state->stances_vector_[0]->ee_contact_poses_[i];
    //         std::cout << i << ": " << transformed_pose.x_ << " " << transformed_pose.y_ << " " << transformed_pose.z_ << " " << transformed_pose.roll_ << " " << transformed_pose.pitch_ << " " << transformed_pose.yaw_ << " " << std::endl;
    //     }
    // }

    // std::cout << "com: " << initial_state->com_.transpose() << std::endl;
    // std::cout << "com_dot: " << initial_state->com_dot_.transpose() << std::endl;

    // std::cout << "=====second state=====" << std::endl;
    // for(unsigned int i = 0; i < ContactManipulator::MANIP_NUM; i++)
    // {
    //     if(second_state->stances_vector_[0]->ee_contact_status_[i])
    //     {
    //         RPYTF transformed_pose = second_state->stances_vector_[0]->ee_contact_poses_[i];
    //         std::cout << i << ": " << transformed_pose.x_ << " " << transformed_pose.y_ << " " << transformed_pose.z_ << " " << transformed_pose.roll_ << " " << transformed_pose.pitch_ << " " << transformed_pose.yaw_ << " " << std::endl;
    //     }
    // }

    auto dynamics_optimizer_interface = std::make_shared<OptimizationInterface>(STEP_TRANSITION_TIME, SUPPORT_PHASE_TIME, "SL_optim_config_template/cfg_kdopt_demo.yaml");

    float dynamics_cost = 0.0;
    bool dynamically_feasible;

    // update the state cost and CoM

    std::vector< std::shared_ptr<ContactState> > contact_state_sequence = {second_state->parent_, second_state};

    dynamics_optimizer_interface->updateContactSequence(contact_state_sequence);

    dynamically_feasible = dynamics_optimizer_interface->dynamicsOptimization(dynamics_cost);

    if(dynamically_feasible)
    {
        // RAVELOG_INFO("Dynamically feasible.\n");
        // update com, com_dot, and parent edge dynamics sequence of the current_state
        dynamics_optimizer_interface->updateStateCoM(second_state);
        dynamics_optimizer_interface->recordEdgeDynamicsSequence(second_state);

        // std::cout << "com: " << second_state->com_.transpose() << std::endl;
        // std::cout << "com_dot: " << second_state->com_dot_.transpose() << std::endl;
        // std::cout << "Dynamics Cost: " << dynamics_cost << std::endl;

        dynamics_optimizer_interface->storeDynamicsOptimizationResult(second_state, dynamics_cost, dynamically_feasible, 10005);
    }
    else
    {
        // RAVELOG_ERROR("Dynamically infeasible.\n");
        dynamics_optimizer_interface->storeDynamicsOptimizationResult(second_state, dynamics_cost, dynamically_feasible, 10005);
    }

    // TransformationMatrix initial_state_feet_mean_transform = initial_state->getFeetMeanTransform();
    // std::shared_ptr<ContactState> mirror_initial_state = initial_state->getMirrorState(initial_state_feet_mean_transform);
    // std::shared_ptr<ContactState> mirror_second_state = second_state->getMirrorState(initial_state_feet_mean_transform);
    // mirror_second_state->parent_ = mirror_initial_state;

    // std::vector< std::shared_ptr<ContactState> > mirror_contact_state_sequence = {mirror_initial_state, mirror_second_state};

    // dynamics_optimizer_interface->updateContactSequence(mirror_contact_state_sequence);

    // dynamically_feasible = dynamics_optimizer_interface->dynamicsOptimization(dynamics_cost);

    // std::cout << "=====initial state=====" << std::endl;
    // for(unsigned int i = 0; i < ContactManipulator::MANIP_NUM; i++)
    // {
    //     if(mirror_initial_state->stances_vector_[0]->ee_contact_status_[i])
    //     {
    //         RPYTF transformed_pose = mirror_initial_state->stances_vector_[0]->ee_contact_poses_[i];
    //         std::cout << i << ": " << transformed_pose.x_ << " " << transformed_pose.y_ << " " << transformed_pose.z_ << " " << transformed_pose.roll_ << " " << transformed_pose.pitch_ << " " << transformed_pose.yaw_ << " " << std::endl;
    //     }
    // }

    // std::cout << "com: " << mirror_initial_state->com_.transpose() << std::endl;
    // std::cout << "com_dot: " << mirror_initial_state->com_dot_.transpose() << std::endl;

    // std::cout << "=====second state=====" << std::endl;
    // for(unsigned int i = 0; i < ContactManipulator::MANIP_NUM; i++)
    // {
    //     if(mirror_second_state->stances_vector_[0]->ee_contact_status_[i])
    //     {
    //         RPYTF transformed_pose = mirror_second_state->stances_vector_[0]->ee_contact_poses_[i];
    //         std::cout << i << ": " << transformed_pose.x_ << " " << transformed_pose.y_ << " " << transformed_pose.z_ << " " << transformed_pose.roll_ << " " << transformed_pose.pitch_ << " " << transformed_pose.yaw_ << " " << std::endl;
    //     }
    // }

    // if(dynamically_feasible)
    // {
    //     RAVELOG_INFO("Dynamically feasible.\n");
    //     // update com, com_dot, and parent edge dynamics sequence of the current_state
    //     dynamics_optimizer_interface->updateStateCoM(mirror_second_state);
    //     dynamics_optimizer_interface->recordEdgeDynamicsSequence(mirror_second_state);

    //     std::cout << "com: " << mirror_second_state->com_.transpose() << std::endl;
    //     std::cout << "com_dot: " << mirror_second_state->com_dot_.transpose() << std::endl;
    //     std::cout << "Dynamics Cost: " << dynamics_cost << std::endl;
    // }
    // else
    // {
    //     RAVELOG_ERROR("Dynamically infeasible.\n");
    // }


}

bool EscherMotionPlanning::startPlanningFromScratch(std::ostream& sout, std::istream& sinput)
{
    penv_ = GetEnv();

    std::string robot_name;
    std::string param;

    RAVELOG_INFO("Parse commands and initialize variables...\n");

    float goal_radius;
    float time_limit;
    float epsilon;
    bool output_first_solution;
    bool goal_as_exact_poses;
    bool use_dynamics_planning;
    bool use_learned_dynamics_model;
    bool enforce_stop_in_the_end;
    PlanningHeuristicsType heuristics_type;
    BranchingMethod branching_method = BranchingMethod::CONTACT_PROJECTION;

    int thread_num = 1;
    int planning_id = 0;

    std::shared_ptr<ContactState> initial_state;

    while(!sinput.eof())
    {
        sinput >> param;
        if(!sinput)
        {
            break;
        }

        if(strcmp(param.c_str(), "structures") == 0)
        {
            parseStructuresCommand(sinput);
        }

        else if(strcmp(param.c_str(), "robot_name") == 0)
        {
            sinput >> robot_name;
            SetActiveRobot(robot_name);
        }

        else if(strcmp(param.c_str(), "robot_properties") == 0)
        {
            parseRobotPropertiesCommand(sinput);
        }

        else if(strcmp(param.c_str(), "initial_state") == 0)
        {
            initial_state = parseContactStateCommand(sinput);
            if(printing_)
            {
                RAVELOG_INFO("Initial state is initialized.\n");
            }
        }

        else if(strcmp(param.c_str(), "goal") == 0)
        {
            goal_.resize(3);
            for(int i = 0; i < 3; i++)
            {
                sinput >> goal_[i];
            }
            std::cout << "The goal is: (x,y,theta) = (" << goal_[0] << "," << goal_[1] << "," << goal_[2] << ")" << std::endl;
        }

        else if(strcmp(param.c_str(), "foot_transition_model") == 0)
        {
            parseFootTransitionModelCommand(sinput);
        }

        else if(strcmp(param.c_str(), "hand_transition_model") == 0)
        {
            parseHandTransitionModelCommand(sinput);
        }

        else if(strcmp(param.c_str(), "planning_parameters") == 0)
        {
            sinput >> goal_radius;
            sinput >> time_limit;
            sinput >> epsilon;

            sinput >> param;
            if(strcmp(param.c_str(), "euclidean") == 0)
            {
                heuristics_type = PlanningHeuristicsType::EUCLIDEAN;
                RAVELOG_INFO("Use Euclidean heuristics.\n");
            }
            else if(strcmp(param.c_str(), "dijkstra") == 0)
            {
                heuristics_type = PlanningHeuristicsType::DIJKSTRA;
                RAVELOG_INFO("Use Dijkstra heuristics.\n");
                // RAVELOG_WARNA("Dijkstra heuristics have not been implemented. Use Euclidean heuristics.\n");
            }

            sinput >> param;
            if(strcmp(param.c_str(), "0") == 0)
            {
                output_first_solution = false;
                RAVELOG_INFO("Will improve the solution after the first one is found.\n");
            }
            else
            {
                output_first_solution = true;
                RAVELOG_INFO("Will terminate the planning with the first solution.\n");
            }

            sinput >> param;
            if(strcmp(param.c_str(), "0") == 0)
            {
                goal_as_exact_poses = false;
                RAVELOG_INFO("The goal is expressed as a radius.\n");
            }
            else
            {
                goal_as_exact_poses = true;
                RAVELOG_INFO("The goal is set as a set of exact poses.\n");
            }

            sinput >> param;
            if(strcmp(param.c_str(), "0") == 0)
            {
                use_dynamics_planning = false;
                RAVELOG_INFO("Planning with quasi-static end-point kinematics.\n");
            }
            else
            {
                use_dynamics_planning = true;
                RAVELOG_INFO("Planning with dynamics optimizer.\n");
            }

            sinput >> param;
            if(strcmp(param.c_str(), "0") == 0)
            {
                use_learned_dynamics_model = false;
                RAVELOG_INFO("Will use optimization to check dynamical feasibility.\n");
            }
            else
            {
                use_learned_dynamics_model = true;
                RAVELOG_INFO("Will use neural network to check dynamical feasibility.\n");
            }

            sinput >> param;
            if(strcmp(param.c_str(), "0") == 0)
            {
                enforce_stop_in_the_end = false;
                RAVELOG_INFO("Will not enforce the robot to stop in the end.\n");
            }
            else
            {
                enforce_stop_in_the_end = true;
                RAVELOG_INFO("Will enforce the robot to stop in the end.\n");
            }
        }

        else if(strcmp(param.c_str(), "thread_num") == 0)
        {
            sinput >> thread_num;
        }

        else if(strcmp(param.c_str(), "branching_method") == 0)
        {
            sinput >> param;

            if(strcmp(param.c_str(), "contact_projection") == 0)
            {
                branching_method = BranchingMethod::CONTACT_PROJECTION;
            }
            else if(strcmp(param.c_str(), "contact_optimization") == 0)
            {
                branching_method = BranchingMethod::CONTACT_OPTIMIZATION;
            }
        }

        else if(strcmp(param.c_str(), "planning_id") == 0)
        {
            sinput >> planning_id;
        }

        // else if(strcmp(param.c_str(), "parallelization") == 0) // maybe used for evaluating the feasibility of the states
        // {
        //     sinput >> param;
        //     if(strcmp(param.c_str(), "0") == 0)
        //     {
        //         is_parallel_ = false;
        //         RAVELOG_INFO("Don't do parallelization.\n");
        //     }
        //     else
        //     {
        //         is_parallel_ = true;
        //         RAVELOG_INFO("Do parallelization.\n");
        //     }
        // }

        else if(strcmp(param.c_str(), "map_grid") == 0)
        {
            parseMapGridDimCommand(sinput);
        }

        else if(strcmp(param.c_str(), "printing") == 0)
        {
            printing_ = true;
            RAVELOG_INFO("Will print commands.\n");
        }

        else
        {
            RAVELOG_ERROR("Unknown command: %s.\n",param.c_str());
            return false;
        }
    }

    RAVELOG_INFO("Thread Number = %d.\n",thread_num);

    drawing_handler_ = std::make_shared<DrawingHandler>(penv_, robot_properties_);

    RAVELOG_INFO("Command Parsing Done. \n");

    map_grid_->obstacleAndGapMapping(penv_, structures_);
    RAVELOG_INFO("Obstacle and ground mapping is Done. \n");

    GridIndices3D goal_cell_indices = map_grid_->positionsToIndices({goal_[0], goal_[1], goal_[2]});
    // std::cout << goal_[0] << " " << goal_[1] << " " << goal_[2] << std::endl;
    map_grid_->generateDijkstrHeuristics(map_grid_->cell_3D_list_[goal_cell_indices[0]][goal_cell_indices[1]][goal_cell_indices[2]]);

    general_ik_interface_ = std::make_shared<GeneralIKInterface>(penv_, probot_);

    ContactSpacePlanning contact_space_planner(robot_properties_, foot_transition_model_, hand_transition_model_, structures_, structures_dict_, map_grid_,
                                               general_ik_interface_, 1, thread_num, drawing_handler_, planning_id, use_dynamics_planning);

    RAVELOG_INFO("Start ANA* Planning \n");

    contact_space_planner.storeSLEnvironment();

    // getchar();

    std::vector< std::shared_ptr<ContactState> > contact_state_path = contact_space_planner.ANAStarPlanning(initial_state, {goal_[0], goal_[1], goal_[2]}, goal_radius, heuristics_type,
                                                                                                            branching_method, time_limit, epsilon, output_first_solution,
                                                                                                            goal_as_exact_poses, use_learned_dynamics_model, enforce_stop_in_the_end);

}

void EscherMotionPlanning::SetActiveRobot(std::string robot_name)
{
    std::vector<OpenRAVE::RobotBasePtr> robots;
    penv_->GetRobots(robots);

    if( robots.size() == 0 )
    {
        RAVELOG_WARNA("No robots to plan for\n");
        return;
    }

    for(std::vector<OpenRAVE::RobotBasePtr>::const_iterator it = robots.begin(); it != robots.end(); it++)
    {
        if( strcmp((*it)->GetName().c_str(), robot_name.c_str() ) == 0 )
        {
            probot_ = *it;
            break;
        }
    }

    if( probot_ == NULL )
    {
        RAVELOG_ERRORA("Failed to find %s.\n", robot_name.c_str());
        return;
    }
}

// called to create a new plugin
OpenRAVE::InterfaceBasePtr CreateInterfaceValidated(OpenRAVE::InterfaceType type, const std::string& interfacename, std::istream& sinput, OpenRAVE::EnvironmentBasePtr penv)
{
    if( type == OpenRAVE::PT_Module && interfacename == "eschermotionplanning" )
    {
        std::cout<<"Interface created."<<std::endl;
        return OpenRAVE::InterfaceBasePtr(new EscherMotionPlanning(penv,sinput));
    }

    return OpenRAVE::InterfaceBasePtr();
}

// called to query available plugins
void GetPluginAttributesValidated(OpenRAVE::PLUGININFO& info)
{
    info.interfacenames[OpenRAVE::PT_Module].push_back("EscherMotionPlanning");
}

// called before plugin is terminated
OPENRAVE_PLUGIN_API void DestroyPlugin()
{
}
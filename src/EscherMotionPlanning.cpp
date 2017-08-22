// #include "EscherMotionPlanning.hpp"
#include "Utilities.hpp"

#include <openrave/plugin.h>

EscherMotionPlanning::EscherMotionPlanning(OpenRAVE::EnvironmentBasePtr penv, std::istream& ss) : OpenRAVE::ModuleBase(penv)
{
    RegisterCommand("StartPlanning",boost::bind(&EscherMotionPlanning::Planning,this,_1,_2),
                    "Start the planning process.");

    RegisterCommand("StartCalculatingTraversability",boost::bind(&EscherMotionPlanning::CalculatingTraversability,this,_1,_2),
                    "Start calculating traversability.");
}

bool EscherMotionPlanning::CalculatingTraversability(std::ostream& sout, std::istream& sinput)
{
    penv_ = GetEnv();
    drawing_handler_ = std::make_shared<DrawingHandler>(penv_);
    std::string robot_name;
    std::string param;
    std::vector< std::array<int,5> > torso_transitions;
    std::array<float,3> torso_grid_dimensions;
    float torso_grid_min_x, torso_grid_min_y, torso_grid_resolution;
    // pass the structures
    // pass the windows
    // pass the grid information

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
            int structures_num;

            sinput >> structures_num;

            RAVELOG_INFO("Input %d structures:",structures_num);

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
                    TrimeshSurface new_surface(penv_, kinbody_name, plane_parameters, edges, vertices, type, id);
                    structures_.push_back(new_surface);
                    Translation3D surface_center = new_surface.getCenter();
                    Translation3D surface_normal = new_surface.getNormal();

                    RAVELOG_INFO("Structure #%d: Trimesh: Center:(%3.2f,%3.2f,%3.2f), Normal:(%3.2f,%3.2f,%3.2f), KinBody Name: %s",
                                 new_surface.getId(),surface_center[0],surface_center[1],surface_center[2],surface_normal[0],surface_normal[1],surface_normal[2],new_surface.getKinbody()->GetName().c_str());
                }
                else if(strcmp(geometry.c_str(), "box") == 0)
                {
                    RAVELOG_WARN("WARNING: Box is not implemented yet.\n");
                }
            }   
        }

        if(strcmp(param.c_str(), "transition_footstep_window_cells") == 0)
        {
            int footstep_transition_num;
            sinput >> footstep_transition_num;

            RAVELOG_INFO("Input %d footstep transitions:",footstep_transition_num);

            for(int i = 0; i < footstep_transition_num; i++)
            {
                std::array<int,3> torso_transition;
                int footstep_transition_cell_tuple_num;

                sinput >> torso_transition[0];
                sinput >> torso_transition[1];
                sinput >> torso_transition[2];
                sinput >> footstep_transition_cell_tuple_num;

                RAVELOG_INFO("Torso Transition:(%d,%d,%d): %d footstep tuples.",torso_transition[0],torso_transition[1],torso_transition[2],footstep_transition_cell_tuple_num);

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

            RAVELOG_INFO("%d torso transitions queried.",torso_transition_num);

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

        if(strcmp(param.c_str(), "footstep_window_grid_dimension") == 0)
        {
            float ground_grid_min_x, ground_grid_max_x, ground_grid_min_y, ground_grid_max_y, ground_grid_resolution;
            sinput >> ground_grid_min_x;
            sinput >> ground_grid_max_x;
            sinput >> ground_grid_min_y;
            sinput >> ground_grid_max_y;
            sinput >> ground_grid_resolution;

            RAVELOG_INFO("Footstep window grid dimensions : x=[%5.3f,%5.3f), y=[%5.3f,%5.3f), Resolution=%5.3f.",ground_grid_min_x,ground_grid_max_x,ground_grid_min_y,ground_grid_max_y,ground_grid_resolution);

            feet_contact_point_grid_ = std::make_shared<GroundContactPointGrid>(ground_grid_min_x,ground_grid_max_x,ground_grid_min_y,ground_grid_max_y,ground_grid_resolution);
            // feet_contact_point_grid_->initializeParameters(ground_grid_min_x,ground_grid_max_x,ground_grid_min_y,ground_grid_max_y,ground_grid_resolution);
        }

        if(strcmp(param.c_str(), "torso_grid_dimension") == 0)
        {
            sinput >> torso_grid_min_x;
            sinput >> torso_grid_min_y;
            sinput >> torso_grid_resolution;

            RAVELOG_INFO("Torso grid dimensions : min_x=%5.3f, min_y=%5.3f, Resolution=%5.3f.",torso_grid_min_x,torso_grid_min_y,torso_grid_resolution);
            
            torso_grid_dimensions = {torso_grid_min_x,torso_grid_min_y,torso_grid_resolution};
        }

        if(strcmp(param.c_str(), "hand_transition_model") == 0)
        {
            int hand_transition_num;
            sinput >> hand_transition_num;

            hand_transition_model_.resize(hand_transition_num);

            RAVELOG_INFO("Load %d hand transition models",hand_transition_num);

            float hand_pitch, hand_yaw;

            for(int i = 0; i < hand_transition_num; i++)
            {
                sinput >> hand_pitch;
                sinput >> hand_yaw;

                hand_transition_model_[i] = {hand_pitch,hand_yaw};
            }
        }

        if(strcmp(param.c_str(), "parallelization") == 0)
        {
            sinput >> param;
            if(strcmp(param.c_str(), "0") == 0)
            {
                is_parallel_ = false;
                RAVELOG_INFO("Don't do parallelization.");
            }
            else
            {
                is_parallel_ = true;
                RAVELOG_INFO("Do parallelization.");
            }
        }

    }

    RAVELOG_INFO("Command parsed; now start calculating traversability...");

    // return true;

    // calculate the clearance on each surface
    RAVELOG_INFO("Now construct the contact point grid on each surface...");
    constructContactPointGrid();

    // project the ground surface contact points onto the 2D grid
    RAVELOG_INFO("Now construct the contact point grid on the 2D ground surface...");
    constructGroundContactPointGrid();

    // batch calculation of every transition traversability of footsteps
    RAVELOG_INFO("Now calculate the footstep contact transition traversability...");
    std::map<std::array<int,5>,float> footstep_traversability;
    footstep_traversability = calculateFootstepTransitionTraversability(torso_grid_dimensions, torso_transitions);

    // batch calculation of every transition traversability of hands
    RAVELOG_INFO("Now construct the hand contact transition traversability...");
    std::map< std::array<int,3>, std::array<float,4> > hand_traversability;
    
    std::set< std::array<int,3> > torso_poses_set;

    for(int i = 0; i < torso_transitions.size(); i++)
    {
        // std::array<int,3> tmp_pose = {torso_transitions[i][0],torso_transitions[i][1],torso_transitions[i][2]};
        torso_poses_set.insert({torso_transitions[i][0],torso_transitions[i][1],torso_transitions[i][2]});
    }

    std::vector< std::array<int,3> > torso_poses(torso_poses_set.begin(), torso_poses_set.end());
    
    hand_traversability = calculateHandTransitionTraversability(torso_grid_dimensions, torso_poses);

    RAVELOG_INFO("All finished...");
    
    // int a;
    // std::cin >> a;

    return true;
}

void EscherMotionPlanning::constructContactPointGrid()
{
    int dead_1 = 0;
    int dead_2 = 0;
    int alive = 0;

    for(std::vector<TrimeshSurface>::iterator st_it = structures_.begin(); st_it != structures_.end(); st_it++)
    {
        std::shared_ptr<SurfaceContactPointGrid> surface_contact_point_grid = st_it->contact_point_grid_;

        std::array<int,2> grid_dimensions = surface_contact_point_grid->getDimensions();
        const int grid_dim_x = grid_dimensions[0];
        const int grid_dim_y = grid_dimensions[1];

        Translation3D project_ray = -st_it->getNormal();
        float project_dist = 0.5;

        std::vector< std::pair<Translation2D,std::vector<TrimeshSurface>::iterator> > checking_structures;

        for(std::vector<TrimeshSurface>::iterator st_it2 = structures_.begin(); st_it2 != structures_.end(); st_it2++)
        {                   
            if(st_it->getId() != st_it2->getId() && st_it->getType() == st_it2->getType())
            {
                Translation2D struct2_center_in_struct_frame = st_it->projectionPlaneFrame(st_it2->getCenter());

                if(struct2_center_in_struct_frame.norm() < st_it->getCircumRadius() + st_it2->getCircumRadius())
                {
                    checking_structures.push_back(std::make_pair(struct2_center_in_struct_frame,st_it2));
                }
            }
        }

        // setting up the contact point grid
        for(int i = 0; i < grid_dim_x; i++)
        {
            std::vector<ContactPoint> tmp_contact_point_list;

            for(int j = 0; j < grid_dim_y; j++)
            {
                GridPositions2D cell_center_position = surface_contact_point_grid->indicesToPositions({i,j});
                Translation2D sample_p_2D = gridPositions2DToTranslation2D(cell_center_position);
                Translation3D sample_p_3D = st_it->getGlobalPosition(sample_p_2D);
                
                bool collision_free = true;

                if(st_it->insidePolygonPlaneFrame(sample_p_2D))
                {
                    for(auto cs_it = checking_structures.begin(); cs_it != checking_structures.end(); cs_it++)
                    {
                        Translation2D struct2_center_in_struct_frame = cs_it->first;
                        std::vector<TrimeshSurface>::iterator st_it2 = cs_it->second;

                        // if((sample_p_2D-struct2_center_in_struct_frame).norm() < st_it2->getCircumRadius())
                        // {
                            Translation3D proj_origin_p = sample_p_3D + project_dist * st_it->getNormal();
                            Translation3D struct2_proj_p = st_it2->projectionGlobalFrame(proj_origin_p,project_ray);

                            if(isValidPosition(struct2_proj_p) && st_it2->insidePolygon(struct2_proj_p))
                            {
                                float struct2_project_dist = (proj_origin_p-struct2_proj_p).norm();

                                if(struct2_project_dist < project_dist)
                                {
                                    collision_free = false;
                                    dead_2 += 1;
                                    break;
                                }
                            }
                        // }
                    }
                }
                else
                {
                    collision_free = false;
                    dead_1 += 1;
                }

                if(collision_free)
                {
                    tmp_contact_point_list.push_back(ContactPoint(sample_p_3D,sample_p_2D,-st_it->getNormal(),9999.0,true));
                    alive += 1;
                }
                else
                {
                    tmp_contact_point_list.push_back(ContactPoint(sample_p_3D,sample_p_2D,-st_it->getNormal(),9999.0,false));
                }

            }

            st_it->contact_point_grid_->contact_point_list_.push_back(tmp_contact_point_list);
        }

        // find the boundaries points in the contact point grid.
        std::vector< std::array<int,2> >  boundary_contact_point_indices;

        for(int i = 0; i < grid_dim_x; i++)
        {
            for(int j = 0; j < grid_dim_y; j++)
            {
                if(st_it->contact_point_grid_->contact_point_list_[i][j].feasible_)
                {
                    // points in the index limit are guaranteed to be boundary points
                    if(i == 0 || i == grid_dim_x-1 || j == 0 || j == grid_dim_y-1)
                    {
                        boundary_contact_point_indices.push_back({i,j});
                        st_it->contact_point_grid_->contact_point_list_[i][j].setClearance(0);
                        continue;
                    }

                    bool is_boundary_point = false;

                    for(int i2 = i-1; i2 <= i+1; i2++)
                    {
                        for(int j2 = j-1; j2 <= j+1; j2++)
                        {
                            if(!st_it->contact_point_grid_->contact_point_list_[i2][j2].feasible_)
                            {
                                boundary_contact_point_indices.push_back({i,j});
                                st_it->contact_point_grid_->contact_point_list_[i][j].setClearance(0);
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
                if(st_it->contact_point_grid_->contact_point_list_[i][j].isFeasible() && 
                   st_it->contact_point_grid_->contact_point_list_[i][j].getClearance() != 0)
                {
                    float dist_to_boundary_point;
                    for(std::vector< std::array<int,2> >::iterator bcpi_it = boundary_contact_point_indices.begin(); bcpi_it != boundary_contact_point_indices.end(); bcpi_it++)
                    {
                        dist_to_boundary_point = hypot(float(bcpi_it->at(0)-i),float(bcpi_it->at(1)-j)) * st_it->contact_point_grid_->getResolution();
                        if(dist_to_boundary_point < st_it->contact_point_grid_->contact_point_list_[i][j].getClearance())
                        {
                            st_it->contact_point_grid_->contact_point_list_[i][j].setClearance(dist_to_boundary_point);
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
    std::vector< std::vector<TrimeshSurface>::iterator > feet_contact_structures;
    for(std::vector<TrimeshSurface>::iterator st_it = structures_.begin(); st_it != structures_.end(); st_it++)
    {
        std::array<int,2> surface_grid_dim = st_it->contact_point_grid_->getDimensions();
        // if(st_it->getType() == TrimeshType::GROUND && st_it->getId() != 99999 && st_it->getId() != 49999 &&
        //    surface_grid_dim[0] > 1 && surface_grid_dim[1] > 1)
        if(st_it->getType() == TrimeshType::GROUND &&
           surface_grid_dim[0] > 1 && surface_grid_dim[1] > 1)
        {
            feet_contact_structures.push_back(st_it);
        }
        
    }

    std::array<int,2> feet_contact_point_grid_dim = feet_contact_point_grid_->getDimensions();
    for(int i = 0; i < feet_contact_point_grid_dim[0]; i++)
    {
        std::vector<float> tmp_score_list(feet_contact_point_grid_dim[1],0.0);

        for(int j = 0; j < feet_contact_point_grid_dim[1]; j++)
        {
            GridPositions2D cell_position = feet_contact_point_grid_->indicesToPositions({i,j});
            float cell_x = cell_position[0];
            float cell_y = cell_position[1];

            for(int k = 0; k < feet_contact_structures.size(); k++)
            {
                std::vector<TrimeshSurface>::iterator st_it = feet_contact_structures[k];
                Translation3D surface_center = st_it->getCenter();

                if(hypot(surface_center[0]-cell_x,surface_center[1]-cell_y) < st_it->getCircumRadius())
                {
                    GridPositions2D proj_feet_contact_point_positions = translation2DToGridPositions2D(st_it->projectionPlaneFrame(Translation3D(cell_x,cell_y,9999.0),GLOBAL_NEGATIVE_Z));

                    std::array<int,2> surface_contact_grid_dim = st_it->contact_point_grid_->getDimensions();

                    // Check if the projection is inside the grid
                    if(st_it->contact_point_grid_->insideGrid(proj_feet_contact_point_positions))
                    {
                        GridIndices2D proj_feet_contact_point_indices = st_it->contact_point_grid_->positionsToIndices(proj_feet_contact_point_positions);
                        
                        if(proj_feet_contact_point_indices[0] < surface_contact_grid_dim[0]-1 &&
                        proj_feet_contact_point_indices[1] < surface_contact_grid_dim[1]-1)
                        {
                                                        
                            int pfcp_ix = proj_feet_contact_point_indices[0];
                            int pfcp_iy = proj_feet_contact_point_indices[1];

                            ContactPoint p1 = st_it->contact_point_grid_->contact_point_list_[pfcp_ix][pfcp_iy];
                            ContactPoint p2 = st_it->contact_point_grid_->contact_point_list_[pfcp_ix+1][pfcp_iy];
                            ContactPoint p3 = st_it->contact_point_grid_->contact_point_list_[pfcp_ix][pfcp_iy+1];
                            ContactPoint p4 = st_it->contact_point_grid_->contact_point_list_[pfcp_ix+1][pfcp_iy+1];

                            if(p1.isFeasible() && p2.isFeasible() && p3.isFeasible() && p4.isFeasible())
                            {
                                float p1_score = p1.getTotalScore(ContactType::FOOT, GLOBAL_NEGATIVE_Z);
                                float p2_score = p2.getTotalScore(ContactType::FOOT, GLOBAL_NEGATIVE_Z);
                                float p3_score = p3.getTotalScore(ContactType::FOOT, GLOBAL_NEGATIVE_Z);
                                float p4_score = p4.getTotalScore(ContactType::FOOT, GLOBAL_NEGATIVE_Z);

                                GridPositions2D cell_center_positions = st_it->contact_point_grid_->indicesToPositions(proj_feet_contact_point_indices);

                                float lx1 = proj_feet_contact_point_positions[0] - (cell_center_positions[0] - 0.5*st_it->contact_point_grid_->getResolution());
                                float lx2 = st_it->contact_point_grid_->getResolution() - lx1;
                                float ly1 = proj_feet_contact_point_positions[1] - (cell_center_positions[1] - 0.5*st_it->contact_point_grid_->getResolution());
                                float ly2 = st_it->contact_point_grid_->getResolution() - ly1;

                                tmp_score_list[j] = st_it->contact_point_grid_->getInterpolatedScore({p1_score,p2_score,p3_score,p4_score}, {lx1,lx2,ly1,ly2});

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

std::map<std::array<int,5>,float> EscherMotionPlanning::calculateFootstepTransitionTraversability(std::array<float,3> torso_grid_dimensions, std::vector<std::array<int,5>> transitions)
{
    std::map<std::array<int,5>,float> traversability_map;

	float torso_grid_min_x = torso_grid_dimensions[0];
	float torso_grid_min_y = torso_grid_dimensions[1];
	int torso_grid_min_theta = TORSO_GRID_MIN_THETA;
    float torso_grid_resolution = torso_grid_dimensions[2];
    int torso_grid_angular_resolution = TORSO_GRID_ANGULAR_RESOLUTION;

	int ix1, iy1, itheta1;
	int ix2, iy2;
	
    float x1, y1;
	int theta1;
	
    int window_theta, window_dix, window_diy;
	
    std::array<int,4> correspondence;
	
	// printf("torso grid dimension: (%5.3f,%5.3f).\n",torso_grid_min_x,torso_grid_min_y);
	// printf("footstep grid dimension: (%5.3f,%5.3f).\n",footstep_window_grid_min_x,footstep_window_grid_min_y);

	for(int i = 0; i < transitions.size(); i++)
	{
		ix1 = transitions[i][0];
		iy1 = transitions[i][1];
		itheta1 = transitions[i][2];

		ix2 = transitions[i][3];
		iy2 = transitions[i][4];

		// printf("From (%d,%d,%d) to (%d,%d): \n",ix1,iy1,itheta1,ix2,iy2);

		x1 = torso_grid_min_x + torso_grid_resolution * float(ix1 + 0.5);
		y1 = torso_grid_min_y + torso_grid_resolution * float(iy1 + 0.5);
		theta1 = torso_grid_min_theta + torso_grid_angular_resolution * itheta1;

		// printf("(x1,y1,theta1)=(%5.5f,%5.5f,%d)\n",x1,y1,theta1);

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

		// printf("Window dixy:(%d,%d), correspondence = [%d,%d,%d,%d].\n",window_dix,window_diy,correspondence[0],correspondence[1],correspondence[2],correspondence[3]);
		// printf("footstep_window_grid_min_xy:(%5.3f,%5.3f).\n",footstep_window_grid_min_x,footstep_window_grid_min_y);
		// printf("Torso:(%d,%d)\n",torso_ix,torso_iy);

        std::array<int,3> window_key = {window_dix,window_diy,window_theta};

        std::vector< std::array<std::array<int,2>,3> > footstep_window = footstep_transition_checking_cells_.find(window_key)->second;

		// printf("Now entering score calculation.\n");
        float traversability = sumFootstepTransitionTraversability(correspondence, footstep_window, torso_indices);
		
        traversability_map.insert(std::make_pair(transitions[i],traversability));   

		// getchar();
	}

	return traversability_map;
}

float EscherMotionPlanning::sumFootstepTransitionTraversability(std::array<int,4> correspondence, std::vector< std::array<std::array<int,2>,3> > footstep_window, GridIndices2D torso_indices)
{
	int x_sign = correspondence[0];
	int x_addition_index = correspondence[1];
	int y_sign = correspondence[2];
	int y_addition_index = correspondence[3];

	// printf("[%d,%d,%d,%d].%d.\n",x_sign,x_addition_index,y_sign,y_addition_index,footstep_num);

	float footstep_window_score = 0.0;

	// #pragma omp parallel for
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
            footstep_window_score = footstep_window_score + feet_contact_point_grid_->score_cell_list_[footstep_cell_x_global][footstep_cell_y_global]
                                                          * feet_contact_point_grid_->score_cell_list_[left_cell_x_global][left_cell_y_global]
                                                          * feet_contact_point_grid_->score_cell_list_[right_cell_x_global][right_cell_y_global];
        }

	}

	// printf("FootstepNum: %d, Score: %5.3f.\n",footstep_window.size(),footstep_window_score);
	return footstep_window_score;
}

std::map< std::array<int,3>, std::array<float,4> > EscherMotionPlanning::calculateHandTransitionTraversability(std::array<float,3> torso_grid_dimensions, std::vector< std::array<int,3> > torso_poses)
{
    // given torso pose, and the environment structures find projection score of each hand transition model
    std::map< std::array<int,3>, std::array<float,4> > hand_transition_traversability;
    float torso_grid_min_x = torso_grid_dimensions[0];
    float torso_grid_min_y = torso_grid_dimensions[1];
    float torso_grid_resolution = torso_grid_dimensions[2];

    for(int i = 0; i < torso_poses.size(); i++)
    {
        int ix = torso_poses[i][0];
        int iy = torso_poses[i][1];
        int itheta = torso_poses[i][2];

        float x = torso_grid_min_x + torso_grid_resolution * (float(ix) + 0.5);
		float y = torso_grid_min_y + torso_grid_resolution * (float(iy) + 0.5);
        float z = 0.0; // assume the robot height is 0, which is clearly not precise
		float theta = TORSO_GRID_MIN_THETA + TORSO_GRID_ANGULAR_RESOLUTION * itheta;
        float theta_rad = theta * DEG2RAD;

        float forward_left_hand_score = 0;
        float forward_right_hand_score = 0;
        float backward_left_hand_score = 0;
        float backward_right_hand_score = 0;

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
                std::vector<TrimeshSurface>::iterator contact_st;

                for(std::vector<TrimeshSurface>::iterator st_it = structures_.begin(); st_it != structures_.end(); st_it++)
                {
                    Translation3D tmp_proj_point = st_it->projectionGlobalFrame(current_shoulder_position, contact_direction);
                    float tmp_proj_dist = (tmp_proj_point-current_shoulder_position).norm();

                    if(isValidPosition(tmp_proj_point) && st_it->insidePolygon(tmp_proj_point) && tmp_proj_dist < proj_dist)
                    {
                        proj_dist = tmp_proj_dist;
                        proj_point = tmp_proj_point;
                        contact_st = st_it;
                    }
                }

                if(proj_dist > MIN_ARM_LENGTH && proj_dist < MAX_ARM_LENGTH)
                {
                    GridPositions2D proj_point_plane_frame = translation2DToGridPositions2D(contact_st->projectionPlaneFrame(proj_point));
                    float score = contact_st->contact_point_grid_->getScore(proj_point_plane_frame,ContactType::HAND,contact_direction);

                    if(manip == ContactManipulator::L_ARM)
                    {
                        if(ht_it->at(0) > -20)
                        {
                            forward_left_hand_score = forward_left_hand_score + score;
                        }
                        
                        if(ht_it->at(0) < 20)
                        {
                            backward_left_hand_score = backward_left_hand_score + score;
                        }

                    }
                    else if(manip == ContactManipulator::R_ARM)
                    {
                        if(ht_it->at(0) > -20)
                        {
                            forward_right_hand_score = forward_right_hand_score + score;
                        }
                        
                        if(ht_it->at(0) < 20)
                        {
                            backward_right_hand_score = backward_right_hand_score + score;
                        }

                    }
                }

            }
        }

        std::array<float,4> traversability_scores = {forward_left_hand_score,backward_left_hand_score,forward_right_hand_score,backward_right_hand_score};
        hand_transition_traversability.insert(std::make_pair(torso_poses[i],traversability_scores));
        
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

bool EscherMotionPlanning::Planning(std::ostream& sout, std::istream& sinput)
{
    std::string robot_name;

    std::string param;

    while(!sinput.eof())
    {
        sinput >> param;
        if(!sinput)
        {
            break;
        }

        if(strcmp(param.c_str(), "robotname") == 0)
        {
            sinput >> robot_name;
        }

        if(strcmp(param.c_str(), "goal") == 0)
        {
            goal_.resize(3);
            for(int i = 0; i < 3; i++)
            {
                sinput >> goal_[i];
            }
            std::cout<<"The goal is: (x,y,z) = ("<<goal_[0]<<","<<goal_[1]<<","<<goal_[2]<<")"<<std::endl;
        }

        if(strcmp(param.c_str(), "parallelization") == 0)
        {
            sinput >> param;
            if(strcmp(param.c_str(), "0") == 0)
            {
                is_parallel_ = false;
                std::cout<<"Don't do parallelization."<<std::endl;
            }
            else
            {
                is_parallel_ = true;
                std::cout<<"Do parallelization."<<std::endl;
            }
        }

    }

    // vector<RobotBasePtr> robots;

    // penv_ = GetEnv();

    // GetEnv()->GetRobots(robots);
    // SetActiveRobots(robot_name,robots);
    // try
    // {
    //     // Construct the environment objects. (See KinBody in OpenRAVE API, and env_handler.py) 
    //     Environment_handler env_handler{GetEnv()};
    //     // sout << "Nearest boundary: " << env_handler.dist_to_boundary(0, 0, 0) << "\n";
    //     // ****************************************************************************//
    //     // Something about constructing environment objects. (walls, ground, and etc.)//
    //     // ****************************************************************************//

    //     // After loading all the parameters, environment object and robot objects, you can execute the main planning function.

    //     // **************************//
    //     // **************************//
    //     // Something about planning //

    //     Motion_plan_library mpl;
    //     Drawing_handler dh{GetEnv()};

    //     std::chrono::time_point<std::chrono::system_clock> start, end;
    //     const vector<Contact_region> &cr = env_handler.get_contact_regions();
    //     start = std::chrono::system_clock::now();

    //     mpl.query(dh, cr,{0,0,0, .5},{goal_[0], goal_[1], goal_[2], 0.5});

    //     end = std::chrono::system_clock::now();
    //     std::chrono::duration<double> elapsed_seconds = end-start;
    //     cout << elapsed_seconds.count() << " seconds!" << endl;
        
    //     int a;
    //     std::cout << "enter any input to exit" << std::endl; 
    //     std::cin>>a; // block
    // }
    // catch(std::exception & e)
    // {
    //     sout << "Exception caught: " << e.what() << "\n";
    // }

    //return the result
    return true;
}


void EscherMotionPlanning::SetActiveRobots(std::string robot_name, const std::vector<OpenRAVE::RobotBasePtr>& robots)
{
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
        RAVELOG_ERRORA("Failed to find %S\n", robot_name.c_str());
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
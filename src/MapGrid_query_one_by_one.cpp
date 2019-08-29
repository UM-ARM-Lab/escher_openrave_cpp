#include "Utilities.hpp"
#include "MapGrid.hpp"
#include <math.h>
#include <cmath>
// #include "PointGrid.hpp"

// int MapCell3D::getTravelDirection(MapCell3D goal_cell)
// {
//     if(goal_cell.ix_ != ix_ || goal_cell.iy_ != iy_)
//     {
//         float direction_theta = atan2(goal_cell.y_ - y_, goal_cell.x_ - x_) * RAD2DEG;
//         float relative_direction_theta = direction_theta - goal_cell.theta_;

//         while(relative_direction_theta >= 360 - TORSO_GRID_ANGULAR_RESOLUTION/2 || relative_direction_theta < -TORSO_GRID_ANGULAR_RESOLUTION/2)
//         {
//             if(relative_direction_theta >= 360 - TORSO_GRID_ANGULAR_RESOLUTION/2)
//             {
//                 relative_direction_theta  = relative_direction_theta - 360;
//             }
//             else if(relative_direction_theta < -TORSO_GRID_ANGULAR_RESOLUTION/2)
//             {
//                 relative_direction_theta  = relative_direction_theta + 360;
//             }
//         }

//         int direction_index = int((relative_direction_theta - (-TORSO_GRID_ANGULAR_RESOLUTION/2))/float(TORSO_GRID_ANGULAR_RESOLUTION));
//     }
//     else
//     {
//         return -1;
//     }
// }




MapGrid::MapGrid(float _min_x, float _max_x, float _min_y, float _max_y, float _xy_resolution, float _theta_resolution, std::shared_ptr<DrawingHandler> _drawing_handler):
xy_resolution_(_xy_resolution),
theta_resolution_(_theta_resolution),
min_x_(_min_x),
max_x_(_max_x),
min_y_(_min_y),
max_y_(_max_y),
min_theta_(-180 - _theta_resolution/2.0),
max_theta_(180 - _theta_resolution/2.0),
dim_x_(int(round((_max_x-_min_x)/_xy_resolution))),
dim_y_(int(round((_max_y-_min_y)/_xy_resolution))),
dim_theta_(360/_theta_resolution),
drawing_handler_(_drawing_handler)
{
    // resize cell_lists to its dimension
    cell_2D_list_.resize(dim_x_, vector<MapCell2DPtr>(dim_y_));
    cell_3D_list_.resize(dim_x_, vector< vector<MapCell3DPtr> >(dim_y_,vector<MapCell3DPtr>(dim_theta_)));

    for(int ix = 0; ix < dim_x_; ix++)
    {
        for(int iy = 0; iy < dim_y_; iy++)
        {
            GridPositions2D xy_positions = indicesToPositionsXY({ix,iy});
            float x = xy_positions[0];
            float y = xy_positions[1];
            for(int itheta = 0; itheta < dim_theta_; itheta++)
            {
                float theta = indicesToPositionsTheta(itheta);
                cell_3D_list_[ix][iy][itheta] = std::make_shared<MapCell3D>(x, y, theta, ix, iy, itheta);
            }

            cell_2D_list_[ix][iy] = std::make_shared<MapCell2D>(x, y, ix, iy);
        }
    }
    // std::cout << "dim: " << dim_x_ << " " << dim_y_ << " " << dim_theta_ << std::endl;
}

GridIndices2D MapGrid::positionsToIndicesXY(GridPositions2D xy_position)
{
    float x = xy_position[0];
    float y = xy_position[1];

    int index_x = int(floor((x-min_x_)/xy_resolution_));
    int index_y = int(floor((y-min_y_)/xy_resolution_));

    if(index_x >= dim_x_ || index_x < 0 || index_y >= dim_y_ ||  index_y < 0)
    {
        RAVELOG_ERROR("Error: Input position (%5.3f,%5.3f) out of bound.\n",x,y);
    }

    return {index_x,index_y};
}

int MapGrid::positionsToIndicesTheta(float theta_position)
{
    int index_theta = int(floor((theta_position - min_theta_) / theta_resolution_));

    if(index_theta >= dim_theta_ || index_theta < 0)
    {
        RAVELOG_ERROR("Error: Input theta %5.3f out of bound.\n",theta_position);
    }

    return index_theta;
}

GridIndices3D MapGrid::positionsToIndices(GridPositions3D position)
{
    GridIndices2D xy_indices = positionsToIndicesXY({position[0],position[1]});
    int theta_index = positionsToIndicesTheta(position[2]);

    return {xy_indices[0], xy_indices[1], theta_index};
}

GridPositions2D MapGrid::indicesToPositionsXY(GridIndices2D xy_indices)
{
    int index_x = xy_indices[0];
    int index_y = xy_indices[1];

    float position_x = min_x_ + (index_x+0.5) * xy_resolution_;
    float position_y = min_y_ + (index_y+0.5) * xy_resolution_;

    if(index_x >= dim_x_ || index_x < 0 || index_y >= dim_y_ ||  index_y < 0)
    {
        RAVELOG_ERROR("Error: Input index (%d,%d) out of bound: Dim=(%d,%d).\n",index_x,index_y,dim_x_,dim_y_);
    }

    return {position_x,position_y};
}

float MapGrid::indicesToPositionsTheta(int theta_index)
{
    float position_theta = min_theta_ + (theta_index + 0.5) * theta_resolution_;

    if(theta_index >= dim_theta_ || theta_index < 0)
    {
        RAVELOG_ERROR("Error: Input theta index %d out of bound.\n",theta_index);
    }

    return position_theta;
}

GridPositions3D MapGrid::indicesToPositions(GridIndices3D indices)
{
    GridPositions2D xy_positions = indicesToPositionsXY({indices[0],indices[1]});

    float theta_position = indicesToPositionsTheta(indices[2]);

    return {xy_positions[0], xy_positions[1], theta_position};
}

void MapGrid::obstacleAndGapMapping(OpenRAVE::EnvironmentBasePtr env, std::vector< std::shared_ptr<TrimeshSurface> > structures)
{
    // !!!!!!!!!!
    heuristic_helper_.saveStructures(structures);

    // gap mapping and obstacle mapping
    OpenRAVE::KinBodyPtr body_collision_box = env->GetKinBody("body_collision_box");
    OpenRAVE::Transform out_of_env_transform = body_collision_box->GetTransform();

    {
        OpenRAVE::EnvironmentMutex::scoped_lock lockenv(env->GetMutex());

        std::vector< std::vector<float> > temp_height_map(dim_x_, std::vector<float>(dim_y_, -99.0));
        std::vector< std::vector<bool> > temp_has_projection_map(dim_x_, std::vector<bool>(dim_y_, false));
        std::vector< std::vector<bool> > has_projection_map(dim_x_, std::vector<bool>(dim_y_, false));

        // std::cout << cell_2D_list_.size() << " " << cell_2D_list_[0].size() << std::endl;
        // std::cout << cell_3D_list_.size() << " " << cell_3D_list_[0].size() << " " << cell_3D_list_[0][0].size() << std::endl;
        // std::cout << dim_x_ << " " << dim_y_ << " " << dim_theta_ << std::endl;

        Translation3D projection_ray(0,0,-1);
        std::cout << "Height map mapping." << std::endl;
        for(int ix = 0; ix < dim_x_; ix++)
        {
            for(int iy = 0; iy < dim_y_; iy++)
            {
                GridPositions2D cell_position = cell_2D_list_[ix][iy]->getPositions();
                Translation3D projection_start_point(cell_position[0], cell_position[1], 99.0);
                float height = -99.0;
                for(auto structure : structures)
                {
                    if(structure->getType() == TrimeshType::GROUND)
                    {
                        Translation3D projected_point = structure->projectionGlobalFrame(projection_start_point, projection_ray);
                        if(structure->insidePolygon(projected_point))
                        {
                            height = projected_point[2] > height ? projected_point[2] : height;
                            temp_has_projection_map[ix][iy] = true;
                        }
                    }
                }

                temp_height_map[ix][iy] = height;
            }
        }

        // Filter(Smooth) the height map (or you can just fill in holes)
        std::cout << "Height map smoothing." << std::endl;
        int window_size = 1; // must be a odd number
        for(int ix = 1; ix < dim_x_-1; ix++)
        {
            for(int iy = 1; iy < dim_y_-1; iy++)
            {
                float height = 0;
                int cell_with_ground_number = 0;

                for(int nix = ix-(window_size-1)/2; nix <= ix+(window_size-1)/2; nix++)
                {
                    for(int niy = iy-(window_size-1)/2; niy <= iy+(window_size-1)/2; niy++)
                    {
                        if(temp_has_projection_map[nix][niy])
                        {
                            height += temp_height_map[nix][niy];
                            cell_with_ground_number++;
                        }
                    }
                }

                if(cell_with_ground_number != 0)
                {
                    cell_2D_list_[ix][iy]->height_ = height / cell_with_ground_number;
                    has_projection_map[ix][iy] = true;
                }
                else
                {
                    cell_2D_list_[ix][iy]->height_ = -99.0;
                }
            }
        }

        // Find out the terrain types (Gap, Solid, and Obstacle)
        std::cout << "Determine terrain types." << std::endl;
        for(int ix = 0; ix < dim_x_; ix++)
        {
            for(int iy = 0; iy < dim_y_; iy++)
            {
                if(!has_projection_map[ix][iy]) // GAP
                {
                    for(int itheta = 0; itheta < dim_theta_; itheta++)
                    {
                        cell_3D_list_[ix][iy][itheta]->terrain_type_ = TerrainType::GAP;
                    }
                }
                else // see if the obstacle is close
                {
                    for(int itheta = 0; itheta < dim_theta_; itheta++)
                    {
                        GridPositions3D cell_3d_position = cell_3D_list_[ix][iy][itheta]->getPositions();
                        RPYTF body_collision_box_transform(cell_3d_position[0], cell_3d_position[1], cell_2D_list_[ix][iy]->height_, 0, 0, cell_3d_position[2]);
                        body_collision_box->SetTransform(body_collision_box_transform.GetRaveTransform());
                        bool in_collision = false;

                        for(auto structure : structures)
                        {
                            if(structure->getType() == TrimeshType::OTHERS)
                            {
                                if(env->CheckCollision(body_collision_box, structure->getKinbody()))
                                {
                                    in_collision = true;
                                    break;
                                }
                            }
                        }

                        if(in_collision)
                        {
                            cell_3D_list_[ix][iy][itheta]->terrain_type_ = TerrainType::OBSTACLE;
                        }
                        else
                        {
                            cell_3D_list_[ix][iy][itheta]->terrain_type_ = TerrainType::SOLID;
                        }

                        // if(itheta == 6)
                        // {
                        //     // std::cout << cell_3d_position[0] << " " << cell_3d_position[1] << " " << cell_2D_list_[ix][iy]->height_ << " " << in_collision << std::endl;
                        //     // getchar();
                        // }

                    }
                }
            }
        }

        std::cout << "Terrain Visualization: " << std::endl;
        for(int ix = dim_x_-1; ix >= 0; ix--)
        {
            for(int iy = dim_y_-1; iy >= 0; iy--)
            {
                if(cell_3D_list_[ix][iy][6]->terrain_type_ == TerrainType::SOLID)
                {
                    std::cout << "1 ";
                }
                else if(cell_3D_list_[ix][iy][6]->terrain_type_ == TerrainType::OBSTACLE)
                {
                    std::cout << "2 ";
                }
                else if(cell_3D_list_[ix][iy][6]->terrain_type_ == TerrainType::GAP)
                {
                    std::cout << "0 ";
                }
                else
                {
                    std::cout << "? ";
                }

            }
            std::cout << std::endl;
        }
    }
}

void MapGrid::generateDijkstraHeuristics(MapCell3DPtr& goal_cell, std::map< int,std::vector<GridIndices3D> > reverse_transition_model, std::unordered_set<GridIndices3D, hash<GridIndices3D> > region_mask)
{
    resetCellCostsAndParent();

    std::priority_queue< MapCell3DPtr, std::vector< MapCell3DPtr >, pointer_more > open_heap;
    goal_cell->g_ = 0;
    goal_cell->h_ = 0;
    goal_cell->is_root_ = true;

    open_heap.push(goal_cell);

    float global_smallest_cost = std::numeric_limits<float>::max();
    float global_highest_cost = 0;

    // assume 8-connected transition model
    while(!open_heap.empty())
    {
        MapCell3DPtr current_cell = open_heap.top();
        open_heap.pop();

        if(current_cell->explore_state_ != ExploreState::OPEN)
        {
            continue;
        }

        if(current_cell->g_ < global_smallest_cost)
        {
            global_smallest_cost = current_cell->g_;
        }

        if(current_cell->g_ > global_highest_cost)
        {
            global_highest_cost = current_cell->g_;
        }

        GridIndices3D current_cell_indices = current_cell->getIndices();
        current_cell->explore_state_ = ExploreState::CLOSED;

        for(auto & transition : reverse_transition_model[current_cell->itheta_])
        {
            int ix = transition[0];
            int iy = transition[1];
            int itheta = transition[2];

            GridIndices3D child_cell_indices = {current_cell_indices[0]+ix, current_cell_indices[1]+iy, (current_cell_indices[2]+itheta)%dim_theta_};

            if(insideGrid(child_cell_indices) && (region_mask.empty() || region_mask.find(child_cell_indices) != region_mask.end()))
            {
                MapCell3DPtr child_cell = cell_3D_list_[child_cell_indices[0]][child_cell_indices[1]][child_cell_indices[2]];
                if(child_cell->terrain_type_ == TerrainType::SOLID)
                // if(true)
                {
                    float dynamic_cost = heuristic_helper_.getDynamicCost(this, child_cell_indices, current_cell_indices);
                    // float dynamic_cost = 0;
                    float edge_cost = euclideanDistBetweenCells(current_cell, child_cell) + step_cost_weight_ + dynamics_cost_weight_ * dynamic_cost; // modify this to include the estimate dynamic cost
                    
                    // float edge_cost = euclideanDistBetweenCells(current_cell, child_cell);
                    if(current_cell->g_ + edge_cost < child_cell->g_)
                    {
                        child_cell->g_ = current_cell->g_ + edge_cost;
                        child_cell->step_num_ = current_cell->step_num_+1;
                        child_cell->parent_ = current_cell;
                        open_heap.push(child_cell);
                    }
                }
            }
        }
    }

    std::cout << global_smallest_cost << " " << global_highest_cost << std::endl;


    // visualize the cost returned by the dijkstra (select the lowest value among all theta in each x,y)
    std::cout << std::endl << "Dijkstra Cost Map Visualization: (0-9: number*0.1 is the cost ratio of the cost of each cell to the range between max/min costs), X: outside mask or not accessible" << std::endl;
    for(int ix = dim_x_-1; ix >= 0; ix--)
    {
        for(int iy = dim_y_-1; iy >= 0; iy--)
        {
            bool inside_mask = false;
            bool solid_ground = false;

            float cell_cost = std::numeric_limits<float>::max();
            for(int itheta = 0; itheta < dim_theta_; itheta++)
            {
                if(region_mask.find({ix,iy,itheta}) != region_mask.end())
                    inside_mask = true;

                if(cell_3D_list_[ix][iy][itheta]->terrain_type_ == TerrainType::SOLID)
                    solid_ground = true;

                if(cell_3D_list_[ix][iy][itheta]->g_ < cell_cost)
                {
                    cell_cost = cell_3D_list_[ix][iy][itheta]->g_;
                }
            }

            if(inside_mask && solid_ground)
            {
                std::cout << int((cell_cost-global_smallest_cost)/(global_highest_cost-global_smallest_cost+0.001)*10) << " ";
            }
            else
            {
                std::cout << "X ";
            }

        }
        std::cout << std::endl;
    }

    RAVELOG_INFO("Finish Generation of Dijkstra Heuristic.");
}

std::vector<MapCell3DPtr> MapGrid::generateTorsoGuidingPath(MapCell3DPtr& initial_cell, MapCell3DPtr& goal_cell, std::map< int,std::vector<GridIndices3D> > transition_model)
{
    resetCellCostsAndParent();

    std::vector<MapCell3DPtr> torso_path;

    std::priority_queue< MapCell3DPtr, std::vector< MapCell3DPtr >, pointer_more > open_heap;
    initial_cell->g_ = 0;
    initial_cell->h_ = euclideanHeuristic(initial_cell, goal_cell);
    initial_cell->is_root_ = true;

    GridIndices3D initial_cell_indices = initial_cell->getIndices();
    GridIndices3D goal_cell_indices = goal_cell->getIndices();

    open_heap.push(initial_cell);

    if(!insideGrid(initial_cell_indices) || !insideGrid(goal_cell_indices))
    {
        RAVELOG_ERROR("Initial (%d,%d,%d) or Goal (%d,%d,%d) Node is not inside the grid. Return empty path.\n",initial_cell_indices[0],initial_cell_indices[1],initial_cell_indices[2],goal_cell_indices[0],goal_cell_indices[1],goal_cell_indices[2]);
        open_heap.pop();
    }

    if(initial_cell->terrain_type_ != TerrainType::SOLID || goal_cell->terrain_type_ != TerrainType::SOLID)
    {
        RAVELOG_ERROR("Initial (%d,%d,%d) or Goal (%d,%d,%d) Node is not SOLID terrain type. Return empty path.\n",initial_cell_indices[0],initial_cell_indices[1],initial_cell_indices[2],goal_cell_indices[0],goal_cell_indices[1],goal_cell_indices[2]);
        open_heap.pop();
    }

    // assume 8-connected transition model
    while(!open_heap.empty())
    {
        MapCell3DPtr current_cell = open_heap.top();
        open_heap.pop();

        // std::cout << "(" << current_cell->getIndices()[0] << ","
        //                  << current_cell->getIndices()[1] << ","
        //                  << current_cell->getIndices()[2] << "), " << current_cell->g_ << " " << current_cell->h_ << " " << current_cell->getF() << std::endl;

        if(current_cell->explore_state_ != ExploreState::OPEN)
        {
            continue;
        }

        GridIndices3D current_cell_indices = current_cell->getIndices();
        current_cell->explore_state_ = ExploreState::CLOSED;

        // drawing_handler_->DrawGridPath(current_cell);

        // ofstream ofs("log.txt", std::ios_base::app);
        ofstream g_ofs("env_5_log.txt", std::ios_base::app);
        // see if the search reaches goal
        if(current_cell_indices[0] == goal_cell_indices[0] &&
           current_cell_indices[1] == goal_cell_indices[1] &&
           current_cell_indices[2] == goal_cell_indices[2])
        {
            // retrace the path
            std::cout << "Found Torso Path." << std::endl;
            drawing_handler_->ClearHandler();
            drawing_handler_->DrawGridPath(current_cell);
            MapCell3DPtr path_cell = current_cell;
            std::cout << "Path Length: " << current_cell->step_num_ << std::endl;
            g_ofs << "Path Length: " << current_cell->step_num_ << std::endl;
            int path_cell_index = 0;
            while(true)
            {
                torso_path.push_back(path_cell);

                std::cout << "Cell " << path_cell_index << ": (" << path_cell->getPositions()[0] << "," << path_cell->getPositions()[1] << "," << path_cell->getPositions()[2] << ")" << 
                                                           ", (" << path_cell->getIndices()[0] << "," << path_cell->getIndices()[1] << "," << path_cell->getIndices()[2] << ")" << " g: " << path_cell->g_ << std::endl;
                
                g_ofs << "Cell " << path_cell_index << ": (" << path_cell->getPositions()[0] << "," << path_cell->getPositions()[1] << "," << path_cell->getPositions()[2] << ")" << 
                                                       ", (" << path_cell->getIndices()[0] << "," << path_cell->getIndices()[1] << "," << path_cell->getIndices()[2] << ")" << " g: " << path_cell->g_ << std::endl;


                if(path_cell->is_root_)
                {
                    break;
                }

                path_cell = path_cell->parent_;
                path_cell_index++;
            }

            std::reverse(torso_path.begin(), torso_path.end());
            std::cout << "Path Length: " << torso_path.size()-1 << std::endl;

            for (auto p1_cell_it = heuristic_helper_.dynamic_cost_map_.begin(); p1_cell_it != heuristic_helper_.dynamic_cost_map_.end(); p1_cell_it++) {
                for (auto p2_cell_it = p1_cell_it->second.begin(); p2_cell_it != p1_cell_it->second.end(); p2_cell_it++) {
                    g_ofs << "(" << p1_cell_it->first[0] << "," << p1_cell_it->first[1] << "," << p1_cell_it->first[2] << ")->("
                          << p2_cell_it->first[0] << "," << p2_cell_it->first[1] << "," << p2_cell_it->first[2] << "): " << p2_cell_it->second << std::endl;
                }
            }

            // for (auto p1_cell_it = heuristic_helper_.ground_depth_and_boundary_map_map_.begin(); p1_cell_it != heuristic_helper_.ground_depth_and_boundary_map_map_.end(); p1_cell_it++) {
            //     std::cout << "one ground map: " << p1_cell_it->first[0] << "," << p1_cell_it->first[1] << "," << p1_cell_it->first[2] << std::endl;
            //     ofstream ofs("depth_and_boundary_maps/" + std::to_string(p1_cell_it->first[0]) + "_" + std::to_string(p1_cell_it->first[1]) + "_" + std::to_string(p1_cell_it->first[2]) + "_ground.txt");
            //     auto temp_a = heuristic_helper_.ground_depth_and_boundary_map_map_[p1_cell_it->first].accessor<float, 2>();
            //     for (int idy = 0; idy < heuristic_helper_.GROUND_MAP_SIDE; idy++) {
            //         for (int idx = 0; idx < heuristic_helper_.GROUND_MAP_SIDE; idx++) {
            //             ofs << temp_a[idy][idx] << " ";
            //         }
            //         ofs << std::endl;
            //     }
            //     ofs.close();
            // }

            // for (auto p1_xy_cell_it = heuristic_helper_.wall_depth_and_boundary_map_map_.begin(); p1_xy_cell_it != heuristic_helper_.wall_depth_and_boundary_map_map_.end(); p1_xy_cell_it++) {
            //     for (auto p1_z_it = p1_xy_cell_it->second.begin(); p1_z_it != p1_xy_cell_it->second.end(); p1_z_it++) {
            //         std::cout << "one wall map: " << p1_xy_cell_it->first[0] << "," << p1_xy_cell_it->first[1] << "," << p1_z_it->first << std::endl;
            //         ofstream ofs("depth_and_boundary_maps/" + std::to_string(p1_xy_cell_it->first[0]) + "_" + std::to_string(p1_xy_cell_it->first[1]) + "_" + std::to_string(p1_z_it->first) + "_wall.txt");
            //         auto temp_a = heuristic_helper_.wall_depth_and_boundary_map_map_[p1_xy_cell_it->first][p1_z_it->first].accessor<float, 2>();
            //         for (int idy = 0; idy < heuristic_helper_.WALL_MAP_WIDTH; idy++) {
            //             for (int idx = 0; idx < heuristic_helper_.WALL_MAP_LENGTH; idx++) {
            //                 ofs << temp_a[idy][idx] << " ";
            //             }
            //             ofs << std::endl;
            //         }
            //         ofs.close();
            //     }
            // }

            break;
        }

        // std::cout << "child nodes: " << std::endl;

        for(auto & transition : transition_model[current_cell->itheta_])
        {
            int ix = transition[0];
            int iy = transition[1];
            int itheta = transition[2];

            // std::cout << "transition: " << ix << " " << iy << " " << itheta << std::endl;

            GridIndices3D child_cell_indices = {current_cell_indices[0]+ix, current_cell_indices[1]+iy, (current_cell_indices[2]+itheta)%dim_theta_};

            if(insideGrid(child_cell_indices))
            {
                MapCell3DPtr child_cell = cell_3D_list_[child_cell_indices[0]][child_cell_indices[1]][child_cell_indices[2]];

                if(child_cell->terrain_type_ == TerrainType::SOLID)
                // if(true)
                {
                    float dynamic_cost = heuristic_helper_.getDynamicCost(this, current_cell_indices, child_cell_indices);
                    // float dynamic_cost = 0;
                    float edge_cost = euclideanDistBetweenCells(current_cell, child_cell) + step_cost_weight_ + dynamics_cost_weight_ * dynamic_cost; // modify this to include the estimate dynamic cost
                    // float edge_cost = euclideanDistBetweenCells(current_cell, child_cell); // modify this to include the estimate dynamic cost
                    child_cell->h_ = euclideanHeuristic(child_cell, goal_cell);

                    if(current_cell->g_ + edge_cost < child_cell->g_)
                    {
                        child_cell->g_ = current_cell->g_ + edge_cost;
                        child_cell->parent_ = current_cell;
                        child_cell->step_num_ = current_cell->step_num_+1;
                        open_heap.push(child_cell);

                        // std::cout << "(" << child_cell->getIndices()[0] << ","
                        //       << child_cell->getIndices()[1] << ","
                        //       << child_cell->getIndices()[2] << "), " << child_cell->g_ << " " << child_cell->h_ << " " << child_cell->getF() << std::endl;
                        // std::cout << (*current_cell < *child_cell) << std::endl;
                    }
                }
            }
        }

        // getchar();
    }

    return torso_path;
}

std::unordered_set<GridIndices3D, hash<GridIndices3D> > MapGrid::getRegionMask(std::vector<MapCell3DPtr> torso_path, float neighbor_distance_range, float neighbor_orientation_range)
{
    std::vector<GridIndices3D> grid_indices_vec(torso_path.size());
    int cell_counter = 0;

    for(auto & cell : torso_path)
    {
        grid_indices_vec[cell_counter] = cell->getIndices();
        cell_counter++;
    }

    return getRegionMask(grid_indices_vec, neighbor_distance_range, neighbor_orientation_range);
}

std::unordered_set<GridIndices3D, hash<GridIndices3D> > MapGrid::getRegionMask(std::vector<GridIndices3D> grid_indices_vec, float neighbor_distance_range, float neighbor_orientation_range)
{
    assert(neighbor_distance_range > 0);
    assert(neighbor_orientation_range > 0);

    std::vector<GridIndices3D> local_mask;

    // get the local mask
    int distance_range = int(neighbor_distance_range / xy_resolution_);
    int orientation_range = int(neighbor_orientation_range / theta_resolution_);

    for(int ix = -distance_range; ix <= distance_range; ix++)
    {
        for(int iy = -distance_range; iy <= distance_range; iy++)
        {
            if(std::hypot(ix*1.0, iy*1.0) * xy_resolution_ < neighbor_distance_range)
            {
                for(int itheta = -orientation_range; itheta <= orientation_range; itheta++)
                {
                    local_mask.push_back({ix,iy,itheta});
                }
            }
        }
    }

    // map the local mask to all the grid indices in grid_indices_vec
    std::unordered_set<GridIndices3D, hash<GridIndices3D> > region_mask;
    std::unordered_set<GridIndices3D, hash<GridIndices3D> > path_mask;

    for(auto & grid_indices : grid_indices_vec)
    {
        path_mask.insert(grid_indices);
        for(auto & local_grid_indices : local_mask)
        {
            GridIndices3D candidate_grid_indices = {grid_indices[0]+local_grid_indices[0], grid_indices[1]+local_grid_indices[1], grid_indices[2]+local_grid_indices[2]};

            if(insideGrid(candidate_grid_indices))
            {
                region_mask.insert(candidate_grid_indices);
            }
        }
    }

    std::cout << "Mask Visualization (in XY): (P: The path, O: Inside mask with solid terrain, I: Inside mask but not solid terrain, X: Outside mask)" << std::endl;
    for(int ix = dim_x_-1; ix >= 0; ix--)
    {
        for(int iy = dim_y_-1; iy >= 0; iy--)
        {
            bool inside_mask = false;
            bool solid_ground = false;
            bool inside_path = false;

            for(int itheta = 0; itheta < dim_theta_; itheta++)
            {
                if(path_mask.find({ix,iy,itheta}) != path_mask.end())
                    inside_path = true;

                if(region_mask.find({ix,iy,itheta}) != region_mask.end())
                    inside_mask = true;

                if(cell_3D_list_[ix][iy][itheta]->terrain_type_ == TerrainType::SOLID)
                    solid_ground = true;
            }

            if(inside_mask)
                if(inside_path)
                    std::cout << "P ";
                else
                    if(solid_ground)
                        std::cout << "O ";
                    else
                        std::cout << "I ";
            else
                std::cout << "X ";
        }
        std::cout << std::endl;
    }

    return region_mask;
}

void MapGrid::resetCellCostsAndParent()
{
    for(int ix = 0; ix < dim_x_; ix++)
    {
        for(int iy = 0; iy < dim_y_; iy++)
        {
            for(int itheta = 0; itheta < dim_theta_; itheta++)
            {
                cell_3D_list_[ix][iy][itheta]->g_ = std::numeric_limits<float>::max();
                cell_3D_list_[ix][iy][itheta]->h_ = 0;
                cell_3D_list_[ix][iy][itheta]->step_num_ = 0;
                cell_3D_list_[ix][iy][itheta]->is_root_ = false;
                cell_3D_list_[ix][iy][itheta]->explore_state_ = ExploreState::OPEN;
            }
        }
    }
}

float MapGrid::euclideanDistBetweenCells(MapCell3DPtr& cell1, MapCell3DPtr& cell2)
{
    GridIndices3D cell1_indices = cell1->getIndices();
    GridIndices3D cell2_indices = cell2->getIndices();
    int ix = cell2_indices[0] - cell1_indices[0];
    int iy = cell2_indices[1] - cell1_indices[1];
    return std::hypot(ix*1.0, iy*1.0) * xy_resolution_;
}

float MapGrid::euclideanHeuristic(MapCell3DPtr& current_cell, MapCell3DPtr& goal_cell)
{
    return euclideanDistBetweenCells(current_cell, goal_cell);
}


MapGrid::HeuristicHelper::HeuristicHelper() {
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("model.pt");
    } catch (const c10::Error& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "error loading the model\n";
        exit(1);
    }
}


void MapGrid::HeuristicHelper::saveStructures(std::vector< std::shared_ptr<TrimeshSurface> > _structures) {
    structures_ = _structures;
}


std::vector<float> MapGrid::HeuristicHelper::getBoundary(const std::vector<std::vector<Translation3D> >& structure_vertices) {
    float env_min_x = std::numeric_limits<float>::max();
    float env_max_x = std::numeric_limits<float>::min();
    float env_min_y = std::numeric_limits<float>::max();
    float env_max_y = std::numeric_limits<float>::min();

    for (auto structure_it = structure_vertices.begin(); structure_it != structure_vertices.end(); structure_it++) {
        for (auto vertex_it = (*structure_it).begin(); vertex_it != (*structure_it).end(); vertex_it++) {
            env_min_x = min((*vertex_it)[0], env_min_x);
            env_max_x = max((*vertex_it)[0], env_max_x);
            env_min_y = min((*vertex_it)[1], env_min_y);
            env_max_y = max((*vertex_it)[1], env_max_y);
        }
    }

    float min_x = floor(env_min_x / MAP_RESOLUTION) * MAP_RESOLUTION - MAP_RESOLUTION / 2.0;
    float max_x = ceil(env_max_x / MAP_RESOLUTION) * MAP_RESOLUTION + MAP_RESOLUTION / 2.0;
    float min_y = floor(env_min_y / MAP_RESOLUTION) * MAP_RESOLUTION - MAP_RESOLUTION / 2.0;
    float max_y = ceil(env_max_y / MAP_RESOLUTION) * MAP_RESOLUTION + MAP_RESOLUTION / 2.0;
    
    std::vector<float> xy_boundary;
    float temp[] = {min_x, max_x, min_y, max_y};
    xy_boundary.assign(temp, temp + 4);
    return xy_boundary;
}


torch::Tensor MapGrid::HeuristicHelper::getBoundaryMap(const std::vector<std::vector<int>>& structure_id_map, torch::Tensor initialized_boundary_map,
                             int dx, int dy, int edge) {
    auto boundary_map_a = initialized_boundary_map.accessor<float, 2>();

    std::unordered_set<GridIndices2D, hash<GridIndices2D> > level_zero_positions;
    for (int idy = edge; idy < edge + dy; idy++) {
        for (int idx = edge; idx < edge + dx; idx++) {
            if (structure_id_map[idy][idx] != structure_id_map[idy - 1][idx]
                || structure_id_map[idy][idx] != structure_id_map[idy + 1][idx]
                || structure_id_map[idy][idx] != structure_id_map[idy][idx - 1]
                || structure_id_map[idy][idx] != structure_id_map[idy][idx + 1]) {
                boundary_map_a[idy][idx] = 1;
                level_zero_positions.insert({idx, idy});
            }
        }
    }

    std::unordered_set<GridIndices2D, hash<GridIndices2D> > level_one_positions;
    for (auto position_it = level_zero_positions.begin(); position_it != level_zero_positions.end(); position_it++) {
        int temp_x = (*position_it)[0];
        int temp_y = (*position_it)[1];
        if (boundary_map_a[temp_y - 1][temp_x] == 0) {
            boundary_map_a[temp_y - 1][temp_x] = 1;
            level_one_positions.insert({temp_x, temp_y - 1});
        }
        if (boundary_map_a[temp_y + 1][temp_x] == 0) {
            boundary_map_a[temp_y + 1][temp_x] = 1;
            level_one_positions.insert({temp_x, temp_y + 1});
        }
        if (boundary_map_a[temp_y][temp_x - 1] == 0) {
            boundary_map_a[temp_y][temp_x - 1] = 1;
            level_one_positions.insert({temp_x - 1, temp_y});
        }
        if (boundary_map_a[temp_y][temp_x + 1] == 0) {
            boundary_map_a[temp_y][temp_x + 1] = 1;
            level_one_positions.insert({temp_x + 1, temp_y});
        }
    }

    for (auto position_it = level_one_positions.begin(); position_it != level_one_positions.end(); position_it++) {
        int temp_x = (*position_it)[0];
        int temp_y = (*position_it)[1];
        if (boundary_map_a[temp_y - 1][temp_x] == 0) {
            boundary_map_a[temp_y - 1][temp_x] = 1;
        }
        if (boundary_map_a[temp_y + 1][temp_x] == 0) {
            boundary_map_a[temp_y + 1][temp_x] = 1;
        }
        if (boundary_map_a[temp_y][temp_x - 1] == 0) {
            boundary_map_a[temp_y][temp_x - 1] = 1;
        }
        if (boundary_map_a[temp_y][temp_x + 1] == 0) {
            boundary_map_a[temp_y][temp_x + 1] = 1;
        }
    }
    return initialized_boundary_map;
}


torch::Tensor MapGrid::HeuristicHelper::getGroundDepthBoundaryMap(MapGrid* map_grid_ptr, GridIndices3D indices) {
    
    // case 1: the specific ground depth map has been generated before
    auto it = ground_depth_and_boundary_map_map_.find(indices);
    if (it != ground_depth_and_boundary_map_map_.end()) {
        return it->second;
    }

    // std::cout << indices[0] << " " << indices[1] << " " << indices[2] << std::endl;
    // case 2: the specific ground depth map has not been generated but the entire one has been generated before
    auto orientation_it = entire_ground_depth_and_boundary_map_map_.find(indices[2]);
    if (orientation_it != entire_ground_depth_and_boundary_map_map_.end()) {
        GridPositions3D position = map_grid_ptr->indicesToPositions(indices);
        // need to check here
        int x_index = pointToLineDistance({position[0], position[1]}, boundary_line_map_[indices[2]][0]) / MAP_RESOLUTION;
        int y_index = pointToLineDistance({position[0], position[1]}, boundary_line_map_[indices[2]][3]) / MAP_RESOLUTION;
        ground_depth_and_boundary_map_map_[indices] = entire_ground_depth_and_boundary_map_map_[indices[2]].slice(0, y_index, y_index + 2 * GROUND_MAP_EDGE + 1).slice(1, x_index, x_index + 2 * GROUND_MAP_EDGE + 1);

        return ground_depth_and_boundary_map_map_[indices];
    }

    // case 3: even the entire ground depth map has not been generated before
    // first quadrant seems more intuitive ...
    int first_quadrant_theta_index = indices[2] % (90 / ANGLE_RESOLUTION) + 180 / ANGLE_RESOLUTION;
    // std::cout << first_quadrant_theta_index << std::endl;
    const RotationMatrix rotation_matrix = RPYToSO3(RPYTF(0, 0, 0, 0, 0, -first_quadrant_theta_index * ANGLE_RESOLUTION + 180));
    std::vector<std::vector<Translation3D> > structure_vertices;
    for (auto structure : structures_) {
        if (structure->getType() == TrimeshType::GROUND) {
            std::vector<Translation3D> temp_vector;
            std::vector<Translation3D> vertices = structure->getVertices();
            for (auto vertex_it = vertices.begin(); vertex_it != vertices.end(); vertex_it++) {
                temp_vector.push_back(rotation_matrix * (*vertex_it));
            }
            structure_vertices.push_back(temp_vector);
        }
    }
    std::vector<float> boundaries = getBoundary(structure_vertices);
    // std::cout << boundaries[0] << " " << boundaries[1] << " " << boundaries[2] << " " << boundaries[3] << std::endl;
    float first_quadrant_theta_radian = (first_quadrant_theta_index * ANGLE_RESOLUTION - 180) / 180.0 * M_PI;
    Line2D line_arr[] = {{cos(first_quadrant_theta_radian), sin(first_quadrant_theta_radian), -boundaries[0]},
                         {cos(first_quadrant_theta_radian), sin(first_quadrant_theta_radian), -boundaries[1]},
                         {sin(first_quadrant_theta_radian), -cos(first_quadrant_theta_radian), boundaries[2]},
                         {sin(first_quadrant_theta_radian), -cos(first_quadrant_theta_radian), boundaries[3]}};
    boundary_line_map_[first_quadrant_theta_index].assign(line_arr, line_arr + 4);
    int dx = round(fabs(boundaries[1] - boundaries[0]) / MAP_RESOLUTION) + 1;
    int dy = round(fabs(boundaries[3] - boundaries[2]) / MAP_RESOLUTION) + 1;
    // std::cout << dx << " " << dy << std::endl; 
    float denominator = line_arr[3][1] * line_arr[0][0] - line_arr[0][1] * line_arr[3][0];
    assert(abs(denominator) > 0.001);
    float numerator_y = line_arr[0][2] * line_arr[3][0] - line_arr[3][2] * line_arr[0][0];
    float start_point_y = numerator_y / denominator;
    float numerator_x = line_arr[3][2] * line_arr[0][1] - line_arr[0][2] * line_arr[3][1];
    float start_point_x = numerator_x / denominator;
    float x_interval = cos((first_quadrant_theta_index * ANGLE_RESOLUTION - 180) * M_PI / 180.0) * MAP_RESOLUTION; // M_PI is defined in cmath
    float y_interval = sin((first_quadrant_theta_index * ANGLE_RESOLUTION - 180) * M_PI / 180.0) * MAP_RESOLUTION;
    // std::cout << x_interval << " " << y_interval << std::endl;
    torch::Tensor ground_depth_map = torch::ones({dy + 2 * GROUND_MAP_EDGE, dx + 2 * GROUND_MAP_EDGE}) * GROUND_DEFAULT_DEPTH;
    // accessors are temporary views of a Tensor
    // assert ground_depth_map is 2-dimensional and holds floats.
    auto ground_depth_map_a = ground_depth_map.accessor<float, 2>();
    std::vector<std::vector<int>> structure_id_map(dy + 2 * GROUND_MAP_EDGE, std::vector<int>(dx + 2 * GROUND_MAP_EDGE, std::numeric_limits<int>::min()));
    Translation3D projection_ray(0, 0, -1);
    // note: pay attention to the x, y order here.
    for (int iy = 0; iy < dy; iy++) {
        float start_x = start_point_x + iy * y_interval;
        float start_y = start_point_y - iy * x_interval;
        for (int ix = 0; ix < dx; ix++) {
            Translation3D projection_start_point(start_x + ix * x_interval, start_y + ix * y_interval, 99.0);
            float height = GROUND_DEFAULT_DEPTH;
            for (auto structure: structures_) {
                if (structure->getType() == TrimeshType::GROUND) {
                    if (euclideanDistance2D({start_x + ix * x_interval, start_y + ix * y_interval}, {structure->getCenter()[0], structure->getCenter()[1]}) <= structure->getCircumRadius()) {
                        Translation3D projected_point = structure->projectionGlobalFrame(projection_start_point, projection_ray);
                        if (structure->insidePolygon(projected_point)) {
                            height = projected_point[2] > height ? projected_point[2] : height;
                            structure_id_map[iy + GROUND_MAP_EDGE][ix + GROUND_MAP_EDGE] = structure->getId();
                        }
                    }
                }
            }
            ground_depth_map_a[iy + GROUND_MAP_EDGE][ix + GROUND_MAP_EDGE] = height;
        }
    }

    torch::Tensor ground_initialized_boundary_map = torch::ones({dy + 2 * GROUND_MAP_EDGE, dx + 2 * GROUND_MAP_EDGE});
    auto ground_initialized_boundary_map_a = ground_initialized_boundary_map.accessor<float, 2>();
    for (int idy = GROUND_MAP_EDGE; idy < GROUND_MAP_EDGE + dy; idy++) {
        for (int idx = GROUND_MAP_EDGE; idx < GROUND_MAP_EDGE + dx; idx++) {
            ground_initialized_boundary_map_a[idy][idx] = 0.0;
        }
    }

    torch::Tensor ground_boundary_map = getBoundaryMap(structure_id_map, ground_initialized_boundary_map, dx, dy, GROUND_MAP_EDGE);
    entire_ground_depth_and_boundary_map_map_[first_quadrant_theta_index] = torch::clamp_max(torch::clamp_min(ground_depth_map + ground_boundary_map * -2, -1), 1);
    // ofstream ofs_1("depth_and_boundary_maps/" + std::to_string(first_quadrant_theta_index) + ".txt");
    // for (int idy = 0; idy < 2 * GROUND_MAP_EDGE + dy; idy++) {
    //     for (int idx = 0; idx < 2 * GROUND_MAP_EDGE + dx; idx++) {
    //         ofs_1 << ground_depth_map_a[idy][idx] << " ";
    //     }
    //     ofs_1 << std::endl;
    // }
    // ofs_1.close();

    // ofstream ofs_1("depth_and_boundary_maps/" + std::to_string(first_quadrant_theta_index) + ".txt");
    // auto entire_ground_depth_and_boundary_map_1_a = entire_ground_depth_and_boundary_map_map_[first_quadrant_theta_index].accessor<float, 2>();
    // for (int idy = 0; idy < 2 * GROUND_MAP_EDGE + dy; idy++) {
    //     for (int idx = 0; idx < 2 * GROUND_MAP_EDGE + dx; idx++) {
    //         ofs_1 << entire_ground_depth_and_boundary_map_1_a[idy][idx] << " ";
    //     }
    //     ofs_1 << std::endl;
    // }
    // ofs_1.close();
    std::vector<Line2D> first_quadrant_boundaries = boundary_line_map_[first_quadrant_theta_index];
    
    // need to think more
    // second quadrant
    entire_ground_depth_and_boundary_map_map_[first_quadrant_theta_index + 90 / ANGLE_RESOLUTION] = torch::rot90(entire_ground_depth_and_boundary_map_map_[first_quadrant_theta_index], {1});
    // std::cout << entire_ground_depth_and_boundary_map_map_[first_quadrant_theta_index + 90 / ANGLE_RESOLUTION];
    // ofstream ofs_2("depth_and_boundary_maps/" + std::to_string(first_quadrant_theta_index + 90 / ANGLE_RESOLUTION) + ".txt");
    // auto entire_ground_depth_and_boundary_map_2_a = entire_ground_depth_and_boundary_map_map_[first_quadrant_theta_index + 90 / ANGLE_RESOLUTION].accessor<float, 2>();
    // for (int idy = 0; idy < 2 * GROUND_MAP_EDGE + dx; idy++) {
    //     for (int idx = 0; idx < 2 * GROUND_MAP_EDGE + dy; idx++) {
    //         ofs_2 << entire_ground_depth_and_boundary_map_2_a[idy][idx] << " ";
    //     }
    //     ofs_2 << std::endl;
    // }
    // ofs_2.close();
    Line2D temp_second[] = {first_quadrant_boundaries[2], first_quadrant_boundaries[3], first_quadrant_boundaries[1], first_quadrant_boundaries[0]};
    boundary_line_map_[first_quadrant_theta_index + 90 / ANGLE_RESOLUTION].assign(temp_second, temp_second + 4);
    // std::cout << boundary_line_map_[first_quadrant_theta_index + 90 / ANGLE_RESOLUTION][0] << " " << boundary_line_map_[first_quadrant_theta_index + 90 / ANGLE_RESOLUTION][1] << " "
    // << boundary_line_map_[first_quadrant_theta_index + 90 / ANGLE_RESOLUTION][2] << " " << boundary_line_map_[first_quadrant_theta_index + 90 / ANGLE_RESOLUTION][3] << std::endl;

    // third quadrant
    entire_ground_depth_and_boundary_map_map_[first_quadrant_theta_index - 180 / ANGLE_RESOLUTION] = torch::rot90(entire_ground_depth_and_boundary_map_map_[first_quadrant_theta_index + 90 / ANGLE_RESOLUTION], {1});
    // std::cout << entire_ground_depth_and_boundary_map_map_[first_quadrant_theta_index - 180 / ANGLE_RESOLUTION];
    // ofstream ofs_3("depth_and_boundary_maps/" + std::to_string(first_quadrant_theta_index - 180 / ANGLE_RESOLUTION) + ".txt");
    // auto entire_ground_depth_and_boundary_map_3_a = entire_ground_depth_and_boundary_map_map_[first_quadrant_theta_index - 180 / ANGLE_RESOLUTION].accessor<float, 2>();
    // for (int idy = 0; idy < 2 * GROUND_MAP_EDGE + dy; idy++) {
    //     for (int idx = 0; idx < 2 * GROUND_MAP_EDGE + dx; idx++) {
    //         ofs_3 << entire_ground_depth_and_boundary_map_3_a[idy][idx] << " ";
    //     }
    //     ofs_3 << std::endl;
    // }
    // ofs_3.close();
    Line2D temp_third[] = {first_quadrant_boundaries[1], first_quadrant_boundaries[0], first_quadrant_boundaries[3], first_quadrant_boundaries[2]};
    boundary_line_map_[first_quadrant_theta_index - 180 / ANGLE_RESOLUTION].assign(temp_third, temp_third + 4);
    // std::cout << boundary_line_map_[first_quadrant_theta_index - 180 / ANGLE_RESOLUTION][0] << " " << boundary_line_map_[first_quadrant_theta_index - 180 / ANGLE_RESOLUTION][1] << " "
    // << boundary_line_map_[first_quadrant_theta_index - 180 / ANGLE_RESOLUTION][2] << " " << boundary_line_map_[first_quadrant_theta_index - 180 / ANGLE_RESOLUTION][3] << std::endl;

    // fourth quadrant
    entire_ground_depth_and_boundary_map_map_[first_quadrant_theta_index - 90 / ANGLE_RESOLUTION] = torch::rot90(entire_ground_depth_and_boundary_map_map_[first_quadrant_theta_index - 180 / ANGLE_RESOLUTION], {1});
    // ofstream ofs_4("depth_and_boundary_maps/" + std::to_string(first_quadrant_theta_index - 90 / ANGLE_RESOLUTION) + ".txt");
    // auto entire_ground_depth_and_boundary_map_4_a = entire_ground_depth_and_boundary_map_map_[first_quadrant_theta_index - 90 / ANGLE_RESOLUTION].accessor<float, 2>();
    // for (int idy = 0; idy < 2 * GROUND_MAP_EDGE + dx; idy++) {
    //     for (int idx = 0; idx < 2 * GROUND_MAP_EDGE + dy; idx++) {
    //         ofs_4 << entire_ground_depth_and_boundary_map_4_a[idy][idx] << " ";
    //     }
    //     ofs_4 << std::endl;
    // }
    // ofs_4.close();
    Line2D temp_fourth[] = {first_quadrant_boundaries[3], first_quadrant_boundaries[2], first_quadrant_boundaries[0], first_quadrant_boundaries[1]};
    boundary_line_map_[first_quadrant_theta_index - 90 / ANGLE_RESOLUTION].assign(temp_fourth, temp_fourth + 4);
    // std::cout << boundary_line_map_[first_quadrant_theta_index - 90 / ANGLE_RESOLUTION][0] << " " << boundary_line_map_[first_quadrant_theta_index - 90 / ANGLE_RESOLUTION][1] << " "
    // << boundary_line_map_[first_quadrant_theta_index - 90 / ANGLE_RESOLUTION][2] << " " << boundary_line_map_[first_quadrant_theta_index - 90 / ANGLE_RESOLUTION][3] << std::endl;

    GridPositions3D position = map_grid_ptr->indicesToPositions(indices);
    // need to check here
    // std::cout << position[0] << " " << position[1] << std::endl;
    // std::cout << boundary_line_map_[indices[2]][0][0] << " " << boundary_line_map_[indices[2]][0][1] << " " << boundary_line_map_[indices[2]][0][2] << std::endl;
    // std::cout << boundary_line_map_[indices[2]][3][0] << " " << boundary_line_map_[indices[2]][3][1] << " " << boundary_line_map_[indices[2]][3][2] << std::endl;
    int x_index = pointToLineDistance({position[0], position[1]}, boundary_line_map_[indices[2]][0]) / MAP_RESOLUTION;
    int y_index = pointToLineDistance({position[0], position[1]}, boundary_line_map_[indices[2]][3]) / MAP_RESOLUTION;
    // std::cout << x_index << " " << y_index << std::endl;
    ground_depth_and_boundary_map_map_[indices] = entire_ground_depth_and_boundary_map_map_[indices[2]].slice(0, y_index, y_index + 2 * GROUND_MAP_EDGE + 1).slice(1, x_index, x_index + 2 * GROUND_MAP_EDGE + 1);
    return ground_depth_and_boundary_map_map_[indices];
    // return torch::ones({65, 65});
}


torch::Tensor MapGrid::HeuristicHelper::getWallDepthBoundaryMap(MapGrid* map_grid_ptr, GridIndices3D indices) {
    GridIndices2D indices_2d = {indices[0], indices[1]};
    auto xy_it = wall_depth_and_boundary_map_map_.find(indices_2d);
    if (xy_it != wall_depth_and_boundary_map_map_.end()) {
        auto theta_it = xy_it->second.find(indices[2]);
        // case 1: the specific wall depth map has been generated before
        if (theta_it != xy_it->second.end()) {
            return theta_it->second;
        }
        // case 2: the specific wall depth map has not been generated but a related one has been generated before
        auto existing_it = xy_it->second.begin();
        int boundary_index = int(WALL_DEPTH_AND_BOUNDARY_MAP_RADIUS * (existing_it->first - indices[2]) * ANGLE_RESOLUTION * M_PI / 180 / MAP_RESOLUTION);
        wall_depth_and_boundary_map_map_[indices_2d][indices[2]] = torch::cat({existing_it->second.slice(1, boundary_index, WALL_MAP_LENGTH), existing_it->second.slice(1, 0, boundary_index)}, 1);        
        return wall_depth_and_boundary_map_map_[indices_2d][indices[2]];
    }

    // case 3: even a related wall depth map has not been generated before
    std::cout << indices[0] << "," << indices[1] << "," << indices[2] << std::endl;
    torch::Tensor wall_depth_map = torch::ones({WALL_MAP_WIDTH, WALL_MAP_LENGTH}) * WALL_DEFAULT_DEPTH;
    auto wall_depth_map_a = wall_depth_map.accessor<float, 2>();
    // 1 is the edge of wall_depth_map
    std::vector<std::vector<int>> structure_id_map(WALL_MAP_WIDTH + 2, std::vector<int>(WALL_MAP_LENGTH + 2, std::numeric_limits<int>::min()));
   
    float theta_interval = 2 * M_PI / WALL_MAP_LENGTH;
    GridPositions3D position = map_grid_ptr->indicesToPositions(indices);
    for (int ix = 0; ix < WALL_MAP_LENGTH; ix++) {
        float projection_angle = position[2] * M_PI / 180 - M_PI - ix * theta_interval;
        // std::cout << projection_angle / M_PI * 180 << std::endl;
        Translation3D projection_ray(cos(projection_angle), sin(projection_angle), 0);
        for (int iy = 0; iy < WALL_MAP_WIDTH; iy++) {
            Translation3D projection_start_point(position[0], position[1], WALL_MAX_HEIGHT - iy * MAP_RESOLUTION);
            float dist = WALL_DEPTH_AND_BOUNDARY_MAP_RADIUS + 0.001;
            int structure_id = 0;
            for (auto structure: structures_) {
                if (structure->getType() == TrimeshType::OTHERS) {
                    Translation3D projected_point = structure->projectionGlobalFrame(projection_start_point, projection_ray);
                    if (structure->insidePolygon(projected_point)) {
                        float new_dist = euclideanDistance3D(projection_start_point, projected_point);
                        if (new_dist < dist) {
                            dist = new_dist;
                            structure_id = structure->getId();
                        } 
                    }
                }
            }
            if (dist < WALL_DEPTH_AND_BOUNDARY_MAP_RADIUS) {
                wall_depth_map_a[iy][ix] = dist;
                structure_id_map[iy + 1][ix + 1] = structure_id;
            }
        }
    }

    torch::Tensor wall_initialized_boundary_map = torch::zeros({WALL_MAP_WIDTH + 2, WALL_MAP_LENGTH + 2});
    auto wall_initialized_boundary_map_a = wall_initialized_boundary_map.accessor<float, 2>();
    for (int idx = 0; idx < WALL_MAP_LENGTH + 2; idx++) {
        wall_initialized_boundary_map_a[0][idx] = 1;
        wall_initialized_boundary_map_a[WALL_MAP_WIDTH + 1][idx] = 1;
    }
    for (int idy = 1; idy < WALL_MAP_WIDTH + 1; idy++) {
        wall_initialized_boundary_map_a[idy][0] = 1;
        wall_initialized_boundary_map_a[idy][WALL_MAP_LENGTH + 1] = 1;
    }

    torch::Tensor wall_boundary_map = getBoundaryMap(structure_id_map, wall_initialized_boundary_map, WALL_MAP_LENGTH, WALL_MAP_WIDTH, 1);
    
    wall_depth_and_boundary_map_map_[indices_2d][indices[2]] = torch::clamp_max(torch::clamp_min(wall_depth_map + (wall_boundary_map * 2).slice(0,1,-1).slice(1,1,-1), 0), 2);
    // ofstream ofs("wall_depth_and_boundary_maps/" + std::to_string(indices[0]) + '_' + std::to_string(indices[1]) + '_' + std::to_string(indices[2]) + "_wall.txt");
    // // auto wall_depth_and_boundary_map_a = wall_depth_and_boundary_map_map_[indices_2d][indices[2]].accessor<float, 2>();
    // for (int idy = 0; idy < WALL_MAP_WIDTH; idy++) {
    //     for (int idx = 0; idx < WALL_MAP_LENGTH; idx++) {
    //         ofs << wall_depth_map_a[idy][idx] << " ";
    //     }
    //     ofs << std::endl;
    // }
    // ofs.close();

    return  wall_depth_and_boundary_map_map_[indices_2d][indices[2]];
    // return torch::ones({25, 252});
}


float MapGrid::HeuristicHelper::getDynamicCost(MapGrid* map_grid_ptr, GridIndices3D current_cell_indices, GridIndices3D child_cell_indices) {
    
    auto current_it = dynamic_cost_map_.find(current_cell_indices);
    if (current_it != dynamic_cost_map_.end()) {
        auto child_it = current_it->second.find(child_cell_indices);
        if (child_it != current_it->second.end()) {
            return child_it->second;
        }
    }

    // current_cell_indices[0] = 1;
    // current_cell_indices[1] = 11;
    // current_cell_indices[2] = 5;

    std::vector<torch::jit::IValue> inputs;
    // std::cout << current_cell_indices[0] << " " << current_cell_indices[1] << " " << current_cell_indices[2] << std::endl;
    // std::cout << child_cell_indices[0] << " " << child_cell_indices[1] << " " << child_cell_indices[2] << std::endl;
    inputs.push_back(getGroundDepthBoundaryMap(map_grid_ptr, current_cell_indices).reshape({1,1,GROUND_MAP_SIDE,GROUND_MAP_SIDE}));
    inputs.push_back(getWallDepthBoundaryMap(map_grid_ptr, current_cell_indices).reshape({1,1,WALL_MAP_WIDTH,WALL_MAP_LENGTH}));
    const RotationMatrix rotation_matrix = RPYToSO3(RPYTF(0, 0, 0, 0, 0, -map_grid_ptr->indicesToPositionsTheta(current_cell_indices[2])));
    Translation3D rotated_position = rotation_matrix * Translation3D((child_cell_indices[0] - current_cell_indices[0]) * GRID_RESOLUTION, (child_cell_indices[1] - current_cell_indices[1]) * GRID_RESOLUTION, 0);
    int discretized_x = discretize(rotated_position[0], GRID_RESOLUTION);
    int discretized_y = discretize(rotated_position[1], GRID_RESOLUTION);
    float p2[3] = {discretized_x, discretized_y, child_cell_indices[2] - current_cell_indices[2]};
    // std::cout << rotated_position[0] << " " << rotated_position[1] << std::endl;
    // std::cout << discretized_x << " " << discretized_y << std::endl;
    inputs.push_back(torch::from_blob(p2, {1,3}));
    torch::Tensor output = module.forward(inputs).toTensor();
    float dynamic_cost = output.accessor<float, 2>()[0][0];
    if (dynamic_cost < 0) {
        std::cout << "negative dynamic cost: " << dynamic_cost << " transition: " << current_cell_indices[0] << "," << current_cell_indices[1] << "," << current_cell_indices[2] << "->"
                  << child_cell_indices[0] << "," << child_cell_indices[1] << "," << child_cell_indices[2] << std::endl;
        dynamic_cost = 0;
    }
    dynamic_cost_map_[current_cell_indices][child_cell_indices] = dynamic_cost;
    return dynamic_cost;
}

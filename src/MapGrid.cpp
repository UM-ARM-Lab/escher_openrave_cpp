#include "Utilities.hpp"
#include "MapGrid.hpp"
#include <cuda.h>
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
dim_theta_(int(round(360/_theta_resolution))),
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
    while (theta_position - min_theta_ > 360 - 0.1 /* error tolerance*/) {
        theta_position -= 360;
    }
    int index_theta = int(floor((theta_position - min_theta_) / theta_resolution_));

    if(index_theta >= dim_theta_ || index_theta < 0)
    {
        // !!!!!!!!!!!!!!!!!!!!
        // getchar();
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
    heuristic_helper_.saveStructures(this, structures);

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
        // !!!!!!!!!!!!!!!!!!!!
        int window_size = 3;
        // int window_size = 1; // must be a odd number
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

        // !!!!!!!!!!!!!!!!!!!!
        // int edge_width = 6;
        // for (int iy = 0; iy < dim_y_; iy++) {
        //     for (int ix = 0; ix < edge_width; ix++) {
        //         for (int itheta = 0; itheta < dim_theta_; itheta++) {
        //             cell_3D_list_[ix][iy][itheta]->terrain_type_ = TerrainType::GAP;
        //             GridPositions3D temp = indicesToPositions({ix, iy, 0});
        //             drawing_handler_->DrawLocation(Translation3D(temp[0], temp[1], 0.2), Vector3D(0,1,0));
        //         }
        //     }
        //     for (int ix = dim_x_ - edge_width; ix < dim_x_; ix++) {
        //         for (int itheta = 0; itheta < dim_theta_; itheta++) {
        //             cell_3D_list_[ix][iy][itheta]->terrain_type_ = TerrainType::GAP;
        //             GridPositions3D temp = indicesToPositions({ix, iy, 0});
        //             drawing_handler_->DrawLocation(Translation3D(temp[0], temp[1], 0.2), Vector3D(0,1,0));
        //         }
        //     }
        // }

        // for (int ix = edge_width; ix < dim_x_ - edge_width; ix++) {
        //     for (int iy = 0; iy < edge_width; iy++) {
        //         for (int itheta = 0; itheta < dim_theta_; itheta++) {
        //             cell_3D_list_[ix][iy][itheta]->terrain_type_ = TerrainType::GAP;
        //             GridPositions3D temp = indicesToPositions({ix, iy, 0});
        //             drawing_handler_->DrawLocation(Translation3D(temp[0], temp[1], 0.2), Vector3D(0,1,0));
        //         }
        //     }
        //     for (int iy = dim_y_ - edge_width; iy < dim_y_; iy++) {
        //         for (int itheta = 0; itheta < dim_theta_; itheta++) {
        //             cell_3D_list_[ix][iy][itheta]->terrain_type_ = TerrainType::GAP;
        //             GridPositions3D temp = indicesToPositions({ix, iy, 0});
        //             drawing_handler_->DrawLocation(Translation3D(temp[0], temp[1], 0.2), Vector3D(0,1,0));
        //         }
        //     }
        // }

        // for (int iy = edge_width; iy < 35; iy++) {
        //     for (int ix = 22; ix < 60; ix++) {
        //         for (int itheta = 0; itheta < dim_theta_; itheta++) {
        //             cell_3D_list_[ix][iy][itheta]->terrain_type_ = TerrainType::GAP;
        //             GridPositions3D temp = indicesToPositions({ix, iy, 0});
        //             drawing_handler_->DrawLocation(Translation3D(temp[0], temp[1], 0.2), Vector3D(0,1,0));
        //         }
        //     }
        // }

        // std::cout << "Terrain Visualization: " << std::endl;
        // for(int ix = dim_x_-1; ix >= 0; ix--)
        // {
        //     for(int iy = dim_y_-1; iy >= 0; iy--)
        //     {
        //         if(cell_3D_list_[ix][iy][6]->terrain_type_ == TerrainType::SOLID)
        //         {
        //             std::cout << "1 ";
        //         }
        //         else if(cell_3D_list_[ix][iy][6]->terrain_type_ == TerrainType::OBSTACLE)
        //         {
        //             std::cout << "2 ";
        //         }
        //         else if(cell_3D_list_[ix][iy][6]->terrain_type_ == TerrainType::GAP)
        //         {
        //             std::cout << "0 ";
        //         }
        //         else
        //         {
        //             std::cout << "? ";
        //         }

        //     }
        //     std::cout << std::endl;
        // }
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

            GridIndices3D child_cell_indices = {current_cell_indices[0]+ix, current_cell_indices[1]+iy, (current_cell_indices[2]+itheta+dim_theta_)%dim_theta_};

            if(insideGrid(child_cell_indices) && (region_mask.empty() || region_mask.find(child_cell_indices) != region_mask.end()))
            {
                MapCell3DPtr child_cell = cell_3D_list_[child_cell_indices[0]][child_cell_indices[1]][child_cell_indices[2]];
                if(child_cell->terrain_type_ == TerrainType::SOLID)
                // if(true)
                {
                    auto child_it = heuristic_helper_.dynamic_cost_map_.find(child_cell_indices);
                    auto current_it = child_it->second.find(current_cell_indices);
                    float dynamic_cost = current_it->second;
                    float edge_cost = euclideanDistBetweenCells(current_cell, child_cell) + step_cost_weight_ + dynamics_cost_weight_ * dynamic_cost; // modify this to include the estimate dynamic cost
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
    std::cout << "global smallest cost: " << global_smallest_cost << " global highest cost: " <<  global_highest_cost << std::endl;

    // ofstream ofs_route("route.txt");
    // for (int ix = 0; ix < dim_x_; ix++) {
    //     for (int iy = 0; iy < dim_y_; iy++) {
    //         for (int itheta = 0; itheta < dim_theta_; itheta++) {
    //             MapCell3DPtr temp_cell = cell_3D_list_[ix][iy][itheta];
    //             while (temp_cell->step_num_ != 0) {
    //                 ofs_route << "(" << temp_cell->ix_ << "," << temp_cell->iy_ << "," << temp_cell->itheta_ << "): g: " << temp_cell->g_ << " -> ";
    //                 temp_cell = temp_cell->parent_;
    //             }
    //             ofs_route << std::endl;
    //         }
    //     }
    // }


    // // visualize the cost returned by the dijkstra (select the lowest value among all theta in each x,y)
    // std::cout << std::endl << "Dijkstra Cost Map Visualization: (0-9: number*0.1 is the cost ratio of the cost of each cell to the range between max/min costs), X: outside mask or not accessible" << std::endl;
    // for(int ix = dim_x_-1; ix >= 0; ix--)
    // {
    //     for(int iy = dim_y_-1; iy >= 0; iy--)
    //     {
    //         bool inside_mask = false;
    //         bool solid_ground = false;

    //         float cell_cost = std::numeric_limits<float>::max();
    //         for(int itheta = 0; itheta < dim_theta_; itheta++)
    //         {
    //             if(region_mask.find({ix,iy,itheta}) != region_mask.end())
    //                 inside_mask = true;

    //             if(cell_3D_list_[ix][iy][itheta]->terrain_type_ == TerrainType::SOLID)
    //                 solid_ground = true;

    //             if(cell_3D_list_[ix][iy][itheta]->g_ < cell_cost)
    //             {
    //                 cell_cost = cell_3D_list_[ix][iy][itheta]->g_;
    //             }
    //         }

    //         if(inside_mask && solid_ground)
    //         {
    //             std::cout << int((cell_cost-global_smallest_cost)/(global_highest_cost-global_smallest_cost+0.001)*10) << " ";
    //         }
    //         else
    //         {
    //             std::cout << "X ";
    //         }

    //     }
    //     std::cout << std::endl;
    // }


    // std::cout << std::fixed;
    // std::cout << std::setprecision(2);
    // for(int itheta = 0; itheta < dim_theta_; itheta++) {
    //     std::cout << "theta: " << itheta << std::endl;
    //     for(int ix = dim_x_-1; ix >= 0; ix--)
    //     {
    //         for(int iy = dim_y_-1; iy >= 0; iy--)
    //         {
    //             bool inside_mask = false;
    //             bool solid_ground = false;

    //             if(region_mask.find({ix,iy,itheta}) != region_mask.end())
    //                 inside_mask = true;

    //             if(cell_3D_list_[ix][iy][itheta]->terrain_type_ == TerrainType::SOLID)
    //                 solid_ground = true;

    //             if(inside_mask && solid_ground)
    //             {
    //                 std::cout << cell_3D_list_[ix][iy][itheta]->g_ << "\t";
    //             }
    //             else
    //             {
    //                 std::cout << "X\t";
    //             }

    //         }
    //         std::cout << std::endl;
    //     }
    // }

    // ofstream ofs("env_11_dijkstra_once.txt");
    // ofs << std::fixed;
    // ofs << std::setprecision(2);
    // for (int itheta = 0; itheta < dim_theta_; itheta++) {
    //     for (int ix = dim_x_-1; ix >= 0; ix--) {
    //         for (int iy = dim_y_-1; iy >= 0; iy--) {
    //             bool inside_mask = false;
    //             bool solid_ground = false;

    //             if (region_mask.find({ix,iy,itheta}) != region_mask.end())
    //                 inside_mask = true;

    //             if (cell_3D_list_[ix][iy][itheta]->terrain_type_ == TerrainType::SOLID)
    //                 solid_ground = true;

    //             if (inside_mask && solid_ground) {
    //                 ofs << cell_3D_list_[ix][iy][itheta]->g_ << " ";
    //             } else {
    //                 ofs << "9999.0 ";
    //             }
    //         }
    //         ofs << std::endl;
    //     }
    // }
    
    // ofstream g_ofs("env_5_log_dijkstra.txt", std::ios_base::app);

    // for (auto p1_cell_it = heuristic_helper_.dynamic_cost_map_.begin(); p1_cell_it != heuristic_helper_.dynamic_cost_map_.end(); p1_cell_it++) {
    //     for (auto p2_cell_it = p1_cell_it->second.begin(); p2_cell_it != p1_cell_it->second.end(); p2_cell_it++) {
    //         g_ofs << "(" << p1_cell_it->first[0] << "," << p1_cell_it->first[1] << "," << p1_cell_it->first[2] << ")->("
    //                 << p2_cell_it->first[0] << "," << p2_cell_it->first[1] << "," << p2_cell_it->first[2] << "): " << p2_cell_it->second << std::endl;
    //     }
    // }

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
        // ofstream g_ofs("env_5_log.txt", std::ios_base::app);
        // see if the search reaches goal
        if(current_cell_indices[0] == goal_cell_indices[0] &&
           current_cell_indices[1] == goal_cell_indices[1] &&
           current_cell_indices[2] == goal_cell_indices[2])
        {
            // retrace the path
            std::cout << "Found Torso Path." << std::endl;
            // drawing_handler_->ClearHandler();
            drawing_handler_->DrawGridPath(current_cell);
            MapCell3DPtr path_cell = current_cell;
            std::cout << "Path Length: " << current_cell->step_num_ << std::endl;
            // g_ofs << "Path Length: " << current_cell->step_num_ << std::endl;
            int path_cell_index = 0;
            while(true)
            {
                torso_path.push_back(path_cell);

                std::cout << "Cell " << path_cell_index << ": (" << path_cell->getPositions()[0] << "," << path_cell->getPositions()[1] << "," << path_cell->getPositions()[2] << ")" << 
                                                           ", (" << path_cell->getIndices()[0] << "," << path_cell->getIndices()[1] << "," << path_cell->getIndices()[2] << ")" << " g: " << path_cell->g_ << std::endl;
                
                // g_ofs << "Cell " << path_cell_index << ": (" << path_cell->getPositions()[0] << "," << path_cell->getPositions()[1] << "," << path_cell->getPositions()[2] << ")" << 
                //                                        ", (" << path_cell->getIndices()[0] << "," << path_cell->getIndices()[1] << "," << path_cell->getIndices()[2] << ")" << " g: " << path_cell->g_ << std::endl;


                if(path_cell->is_root_)
                {
                    break;
                }

                path_cell = path_cell->parent_;
                path_cell_index++;
            }

            std::reverse(torso_path.begin(), torso_path.end());
            std::cout << "Path Length: " << torso_path.size()-1 << std::endl;

            // for (auto p1_cell_it = heuristic_helper_.dynamic_cost_map_.begin(); p1_cell_it != heuristic_helper_.dynamic_cost_map_.end(); p1_cell_it++) {
            //     for (auto p2_cell_it = p1_cell_it->second.begin(); p2_cell_it != p1_cell_it->second.end(); p2_cell_it++) {
            //         g_ofs << "(" << p1_cell_it->first[0] << "," << p1_cell_it->first[1] << "," << p1_cell_it->first[2] << ")->("
            //               << p2_cell_it->first[0] << "," << p2_cell_it->first[1] << "," << p2_cell_it->first[2] << "): " << p2_cell_it->second << std::endl;
            //     }
            // }

            // for (auto p1_cell_it = heuristic_helper_.ground_depth_and_boundary_map_map_.begin(); p1_cell_it != heuristic_helper_.ground_depth_and_boundary_map_map_.end(); p1_cell_it++) {
            //     // std::cout << "one ground map: " << p1_cell_it->first[0] << "," << p1_cell_it->first[1] << "," << p1_cell_it->first[2] << std::endl;
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
            //         // std::cout << "one wall map: " << p1_xy_cell_it->first[0] << "," << p1_xy_cell_it->first[1] << "," << p1_z_it->first << std::endl;
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

            GridIndices3D child_cell_indices = {current_cell_indices[0]+ix, current_cell_indices[1]+iy, (current_cell_indices[2]+itheta+dim_theta_)%dim_theta_};

            if(insideGrid(child_cell_indices))
            {
                MapCell3DPtr child_cell = cell_3D_list_[child_cell_indices[0]][child_cell_indices[1]][child_cell_indices[2]];

                if(child_cell->terrain_type_ == TerrainType::SOLID)
                // if(true)
                {
                    auto current_it = heuristic_helper_.dynamic_cost_map_.find(current_cell_indices);
                    auto child_it = current_it->second.find(child_cell_indices);
                    float dynamic_cost = child_it->second;
                    // dynamic_cost = 0;
                    float edge_cost = euclideanDistBetweenCells(current_cell, child_cell) + step_cost_weight_ + dynamics_cost_weight_ * dynamic_cost; // modify this to include the estimate dynamic cost
                    // float edge_cost = euclideanDistBetweenCells(current_cell, child_cell); // modify this to include the estimate dynamic cost
                    child_cell->h_ = euclideanHeuristic(child_cell, goal_cell);

                    if(current_cell->g_ + edge_cost < child_cell->g_)
                    {
                        child_cell->g_ = current_cell->g_ + edge_cost;
                        child_cell->parent_ = current_cell;
                        child_cell->step_num_ = current_cell->step_num_+1;
                        open_heap.push(child_cell);
                    }
                }
            }
        }
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


void MapGrid::read_transition_model(std::map< int,std::vector<GridIndices3D> > transition_model) {
    this->transition_model = transition_model;
    // !!!!!!!!!!!!!!!!!!!!
    heuristic_helper_.predict_dynamic_costs_of_all_transitions(this);
}


MapGrid::HeuristicHelper::HeuristicHelper() {
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        // module = torch::jit::load("model.pt");
        torch::jit::script::Module module_0 = torch::jit::load("depth_and_boundary_combined_B_17_weighted_0.001_checkpoint.pt");
        module_0.to(torch::Device(torch::kCUDA, 0));
        modules.push_back(module_0);
        torch::jit::script::Module module_1 = torch::jit::load("depth_and_boundary_combined_B_17_weighted_0.001_checkpoint.pt");
        module_1.to(torch::Device(torch::kCUDA, 0));
        modules.push_back(module_1);
        torch::jit::script::Module module_2 = torch::jit::load("depth_and_boundary_combined_B_17_weighted_0.001_checkpoint.pt");
        module_2.to(torch::Device(torch::kCUDA, 0));
        modules.push_back(module_2);
        torch::jit::script::Module module_3 = torch::jit::load("depth_and_boundary_combined_B_17_weighted_0.001_checkpoint.pt");
        module_3.to(torch::Device(torch::kCUDA, 0));
        modules.push_back(module_3);
    } catch (const c10::Error& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "error loading the model\n";
        exit(1);
    }
}


torch::Tensor MapGrid::HeuristicHelper::getBoundaryMap(const std::vector<std::vector<int>>& structure_id_map, int dx, int dy) {
    torch::Tensor boundary_map = torch::zeros({dy, dx});
    
    auto boundary_map_a = boundary_map.accessor<float, 2>();
    std::unordered_set<GridIndices2D, hash<GridIndices2D> > level_zero_positions;

    for (int idy = 0; idy < dy; idy++) {
        for (int idx = 0; idx < dx; idx++) {
            if (structure_id_map[idy][idx] == -1) {
                boundary_map_a[idy][idx] = 1;
                level_zero_positions.insert({idx, idy});
            }
        }
    }

    for (int idy = 0; idy < dy; idy++) {
        for (int idx = 0; idx < dx; idx++) {
            if (idy > 1 && structure_id_map[idy][idx] != structure_id_map[idy - 1][idx]) {
                boundary_map_a[idy][idx] = 1;
                level_zero_positions.insert({idx, idy});
            } else if (idy < dy - 1 && structure_id_map[idy][idx] != structure_id_map[idy + 1][idx]) {
                boundary_map_a[idy][idx] = 1;
                level_zero_positions.insert({idx, idy});
            } else if (idx > 1 && structure_id_map[idy][idx] != structure_id_map[idy][idx - 1]) {
                boundary_map_a[idy][idx] = 1;
                level_zero_positions.insert({idx, idy});
            } else if (idx < dx - 1 && structure_id_map[idy][idx] != structure_id_map[idy][idx + 1]) {
                boundary_map_a[idy][idx] = 1;
                level_zero_positions.insert({idx, idy});
            }
        }
    }

    std::unordered_set<GridIndices2D, hash<GridIndices2D> > level_one_positions;
    for (auto position_it = level_zero_positions.begin(); position_it != level_zero_positions.end(); position_it++) {
        int temp_x = (*position_it)[0];
        int temp_y = (*position_it)[1];
        if (temp_y > 1 && boundary_map_a[temp_y - 1][temp_x] < 0.1) {
            boundary_map_a[temp_y - 1][temp_x] = 1;
            level_one_positions.insert({temp_x, temp_y - 1});
        }
        if (temp_y < dy - 1 && boundary_map_a[temp_y + 1][temp_x] < 0.1) {
            boundary_map_a[temp_y + 1][temp_x] = 1;
            level_one_positions.insert({temp_x, temp_y + 1});
        }
        if (temp_x > 1 && boundary_map_a[temp_y][temp_x - 1] < 0.1) {
            boundary_map_a[temp_y][temp_x - 1] = 1;
            level_one_positions.insert({temp_x - 1, temp_y});
        }
        if (temp_x < dx - 1 && boundary_map_a[temp_y][temp_x + 1] < 0.1) {
            boundary_map_a[temp_y][temp_x + 1] = 1;
            level_one_positions.insert({temp_x + 1, temp_y});
        }
    }

    for (auto position_it = level_one_positions.begin(); position_it != level_one_positions.end(); position_it++) {
        int temp_x = (*position_it)[0];
        int temp_y = (*position_it)[1];
        if (temp_y > 1 && boundary_map_a[temp_y - 1][temp_x] < 0.1) {
            boundary_map_a[temp_y - 1][temp_x] = 1;
        }
        if (temp_y < dy - 1 && boundary_map_a[temp_y + 1][temp_x] < 0.1) {
            boundary_map_a[temp_y + 1][temp_x] = 1;
        }
        if (temp_x > 1 && boundary_map_a[temp_y][temp_x - 1] < 0.1) {
            boundary_map_a[temp_y][temp_x - 1] = 1;
        }
        if (temp_x < dx - 1 && boundary_map_a[temp_y][temp_x + 1] < 0.1) {
            boundary_map_a[temp_y][temp_x + 1] = 1;
        }
    }
    return boundary_map;
}


void MapGrid::HeuristicHelper::saveStructures(MapGrid* map_grid_ptr, std::vector< std::shared_ptr<TrimeshSurface> > _structures) {
    structures_ = _structures;

    int dx = map_grid_ptr->dim_x_ * int(round(GRID_RESOLUTION / MAP_RESOLUTION)) + 1;
    int dy = map_grid_ptr->dim_y_ * int(round(GRID_RESOLUTION / MAP_RESOLUTION)) + 1;
    torch::Tensor entire_ground_depth_map = torch::ones({dy + 2 * GROUND_MAP_EDGE, dx + 2 * GROUND_MAP_EDGE}) * GROUND_DEFAULT_DEPTH;
    auto entire_ground_depth_map_a = entire_ground_depth_map.accessor<float, 2>();
    std::vector<std::vector<int>> structure_id_map(dy + 2 * GROUND_MAP_EDGE, std::vector<int>(dx + 2 * GROUND_MAP_EDGE, -1));
    Translation3D projection_ray(0, 0, -1);
    float start_min_x = map_grid_ptr->min_x_;
    float start_max_y = map_grid_ptr->max_y_;
    for (int iy = 0; iy < dy; iy++) {
        float start_y = start_max_y - iy * MAP_RESOLUTION;
        for (int ix = 0; ix < dx; ix++) {
            float start_x = start_min_x + ix * MAP_RESOLUTION;
            Translation3D projection_start_point(start_x, start_y, 99.0);
            for (auto structure: structures_) {
                if (structure->getType() == TrimeshType::GROUND) {
                    if (euclideanDistance2D({start_x, start_y}, {structure->getCenter()[0], structure->getCenter()[1]}) <= structure->getCircumRadius()) {
                        Translation3D projected_point = structure->projectionGlobalFrame(projection_start_point, projection_ray);
                        if (structure->insidePolygon(projected_point)) {
                            if (projected_point[2] > entire_ground_depth_map_a[iy + GROUND_MAP_EDGE][ix + GROUND_MAP_EDGE]) {
                                entire_ground_depth_map_a[iy + GROUND_MAP_EDGE][ix + GROUND_MAP_EDGE] = projected_point[2];
                                structure_id_map[iy + GROUND_MAP_EDGE][ix + GROUND_MAP_EDGE] = structure->getId();
                            } 
                        }
                    }
                }
            }
            
        }
    }

    torch::Tensor entire_ground_boundary_map = getBoundaryMap(structure_id_map, dx + 2 * GROUND_MAP_EDGE, dy + 2 * GROUND_MAP_EDGE);
    entire_ground_map_ = torch::stack({entire_ground_depth_map, entire_ground_boundary_map}, 0 /* dimension */);
    
    auto entire_ground_map_a = entire_ground_map_.accessor<float, 3>();
    // ofstream ofs_ground_depth("depth_and_boundary_maps/entire_ground_depth_map.txt");
    // for (int idy = 0; idy < 2 * GROUND_MAP_EDGE + dy; idy++) {
    //     for (int idx = 0; idx < 2 * GROUND_MAP_EDGE + dx; idx++) {
    //         ofs_ground_depth << entire_ground_map_a[0][idy][idx] << " ";
    //     }
    //     ofs_ground_depth << std::endl;
    // }
    // ofs_ground_depth.close();
    // ofstream ofs_ground_boundary("depth_and_boundary_maps/entire_ground_boundary_map.txt");
    // for (int idy = 0; idy < 2 * GROUND_MAP_EDGE + dy; idy++) {
    //     for (int idx = 0; idx < 2 * GROUND_MAP_EDGE + dx; idx++) {
    //         ofs_ground_boundary << entire_ground_map_a[1][idy][idx] << " ";
    //     }
    //     ofs_ground_boundary << std::endl;
    // }
    // ofs_ground_boundary.close();

    // std::cout << "entire ground map visualization done" << std::endl;
}


torch::Tensor MapGrid::HeuristicHelper::getGroundDepthBoundaryMap(MapGrid* map_grid_ptr, GridIndices3D indices) {
    int model_index = indices[2] % NUM_MODELS;
    int orientation_index = indices[2] - (model_index + 8);

    auto it = ground_maps_map_.find({indices[0], indices[1], orientation_index});
    if (it == ground_maps_map_.end()) {
        GridPositions3D position = map_grid_ptr->indicesToPositions(indices);
        int x_start_index = round((position[0] - map_grid_ptr->min_x_) / MAP_RESOLUTION);
        int y_start_index = round((map_grid_ptr->max_y_ - position[1]) / MAP_RESOLUTION);
        ground_maps_map_[{indices[0], indices[1], 0}] = entire_ground_map_.slice(1, y_start_index, y_start_index + GROUND_MAP_SIDE).slice(2, x_start_index, x_start_index + GROUND_MAP_SIDE);
        ground_maps_map_[{indices[0], indices[1], -4}] = torch::rot90(ground_maps_map_[{indices[0], indices[1], 0}], 1 /* rotation times */, {1,2});
        ground_maps_map_[{indices[0], indices[1], -8}] = torch::rot90(ground_maps_map_[{indices[0], indices[1], -4}], 1 /* rotation times */, {1,2});
        ground_maps_map_[{indices[0], indices[1], 4}] = torch::rot90(ground_maps_map_[{indices[0], indices[1], -8}], 1 /* rotation times */, {1,2});
    }
    return ground_maps_map_[{indices[0], indices[1], orientation_index}];
}


float calculate_altitude(std::vector<float> numbers) {
    int num = 0;
    float sum = 0;
    for (unsigned int i = 0; i < numbers.size(); i++) {
        if (numbers[i] > -0.9) {
            num += 1;
            sum += numbers[i];
        }
    }
    return sum / num;
}


std::vector<torch::Tensor> MapGrid::HeuristicHelper::getWallDepthBoundaryMap(MapGrid* map_grid_ptr, GridIndices3D indices) {
    auto it = triple_wall_maps_map_.find({indices[0], indices[1]});
    if (it == triple_wall_maps_map_.end()) {
        // generate entire wall depth map
        torch::Tensor entire_wall_depth_map = torch::ones({WALL_MAP_WIDTH, WALL_MAP_LENGTH}) * WALL_DEFAULT_DEPTH;
        auto entire_wall_depth_map_a = entire_wall_depth_map.accessor<float, 2>();
        std::vector<std::vector<int>> structure_id_map(WALL_MAP_WIDTH, std::vector<int>(WALL_MAP_LENGTH, -1));
        // calculate altitude
        torch::Tensor temp_ground_depth_map = getGroundDepthBoundaryMap(map_grid_ptr, indices);
        auto temp_ground_depth_map_a = temp_ground_depth_map.accessor<float, 3>();
        float numbers[9] = {temp_ground_depth_map_a[0][27][27], temp_ground_depth_map_a[0][27][33], temp_ground_depth_map_a[0][27][39],
                            temp_ground_depth_map_a[0][33][27], temp_ground_depth_map_a[0][33][33], temp_ground_depth_map_a[0][33][39],
                            temp_ground_depth_map_a[0][39][27], temp_ground_depth_map_a[0][39][33], temp_ground_depth_map_a[0][39][39]};
        std::vector<float> temp_vector(numbers, numbers + 9);
        float altitude = calculate_altitude(temp_vector);
        float theta_interval = MAP_RESOLUTION / WALL_DEPTH_AND_BOUNDARY_MAP_RADIUS;
        GridPositions3D position = map_grid_ptr->indicesToPositions(indices);
        for (int ix = 0; ix < WALL_MAP_LENGTH; ix++) {
            float projection_angle = M_PI - ix * theta_interval;
            Translation3D projection_ray(cos(projection_angle), sin(projection_angle), 0);
            for (int iy = 0; iy < WALL_MAP_WIDTH; iy++) {
                Translation3D projection_start_point(position[0], position[1], altitude + WALL_MAX_HEIGHT - iy * MAP_RESOLUTION);
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
                    entire_wall_depth_map_a[iy][ix] = dist;
                    structure_id_map[iy][ix] = structure_id;
                }
            }
        }

        torch::Tensor entire_wall_boundary_map = getBoundaryMap(structure_id_map, WALL_MAP_LENGTH, WALL_MAP_WIDTH);
        triple_wall_maps_map_[{indices[0], indices[1]}] = torch::stack({entire_wall_depth_map, entire_wall_boundary_map}, 0 /* dim */).repeat({1,1,3});
        
        for (int itheta = 0; itheta < map_grid_ptr->dim_theta_; itheta++) {
            int edge1 = WALL_MAP_LENGTH * 1 / 12.0 - WALL_MAP_LENGTH * (itheta - 8) / 16.0 + WALL_MAP_LENGTH;
            int edge3 = WALL_MAP_LENGTH * 7 / 12.0 - WALL_MAP_LENGTH * (itheta - 8) / 16.0 + WALL_MAP_LENGTH;
            left_wall_maps_map_[{indices[0], indices[1], itheta}] = triple_wall_maps_map_[{indices[0], indices[1]}].slice(2, edge1, edge1 + WALL_MAP_LENGTH_ONE_THIRD);
            right_wall_maps_map_[{indices[0], indices[1], itheta}] = triple_wall_maps_map_[{indices[0], indices[1]}].slice(2, edge3, edge3 + WALL_MAP_LENGTH_ONE_THIRD);
        }    
    }
    std::vector<torch::Tensor> temp;
    temp.push_back(left_wall_maps_map_[indices]);
    temp.push_back(right_wall_maps_map_[indices]);
    return temp;
}


void MapGrid::HeuristicHelper::predict_dynamic_costs_of_all_transitions(MapGrid* map_grid_ptr) {
    for (int ix = 0; ix < map_grid_ptr->dim_x_; ix++) {
        for (int iy = 0; iy < map_grid_ptr->dim_y_; iy++) {
            for (int itheta = 0; itheta < map_grid_ptr->dim_theta_; itheta++) {
                if (!map_grid_ptr->insideGrid(GridIndices3D({ix, iy, itheta})) || map_grid_ptr->cell_3D_list_[ix][iy][itheta]->terrain_type_ != TerrainType::SOLID) continue;

                MapCell3DPtr current_cell = map_grid_ptr->cell_3D_list_[ix][iy][itheta];

                std::vector<GridIndices3D> pending_transitions;
                for (auto & transition : map_grid_ptr->transition_model[itheta]) {
                    GridIndices3D child_cell_indices = {ix + transition[0], iy + transition[1], (itheta + transition[2] + map_grid_ptr->dim_theta_) % map_grid_ptr->dim_theta_};
                    if (map_grid_ptr->insideGrid(child_cell_indices) && 
                        map_grid_ptr->cell_3D_list_[child_cell_indices[0]][child_cell_indices[1]][child_cell_indices[2]]->terrain_type_ == TerrainType::SOLID) {
                        pending_transitions.push_back(transition);
                    }
                }
                
                if (!pending_transitions.empty()) {
                    std::vector<torch::jit::IValue> inputs;
                    torch::Tensor ground_map = getGroundDepthBoundaryMap(map_grid_ptr, {ix,iy,itheta});
                    // auto ground_map_a = ground_map.accessor<float, 3>();
                    // ofstream ofs_ground_depth("depth_and_boundary_maps/ground_depth_map" + std::to_string(ix) + "_" + std::to_string(iy) + "_" + std::to_string(itheta) + ".txt");
                    // for (int idy = 0; idy < GROUND_MAP_SIDE; idy++) {
                    //     for (int idx = 0; idx < GROUND_MAP_SIDE; idx++) {
                    //         ofs_ground_depth << ground_map_a[0][idy][idx] << " ";
                    //     }
                    //     ofs_ground_depth << std::endl;
                    // }
                    // ofs_ground_depth.close();
                    // ofstream ofs_ground_boundary("depth_and_boundary_maps/ground_boundary_map" + std::to_string(ix) + "_" + std::to_string(iy) + "_" + std::to_string(itheta) + ".txt");
                    // for (int idy = 0; idy < GROUND_MAP_SIDE; idy++) {
                    //     for (int idx = 0; idx < GROUND_MAP_SIDE; idx++) {
                    //         ofs_ground_boundary << ground_map_a[1][idy][idx] << " ";
                    //     }
                    //     ofs_ground_boundary << std::endl;
                    // }
                    // ofs_ground_boundary.close();
                    inputs.push_back(ground_map.unsqueeze(0).repeat({pending_transitions.size(),1,1,1}).cuda());
                    std::vector<torch::Tensor> wall_maps = getWallDepthBoundaryMap(map_grid_ptr, {ix,iy,itheta});
                    // auto left_wall_map_a = wall_maps[0].accessor<float, 3>();
                    // ofstream ofs_wall_depth("depth_and_boundary_maps/left_wall_depth_map_" + std::to_string(ix) + "_" + std::to_string(iy) + "_" + std::to_string(itheta) + ".txt");
                    // for (int idy = 0; idy < WALL_MAP_WIDTH; idy++) {
                    //     for (int idx = 0; idx < WALL_MAP_LENGTH_ONE_THIRD; idx++) {
                    //         ofs_wall_depth << left_wall_map_a[0][idy][idx] << " ";
                    //     }
                    //     ofs_wall_depth << std::endl;
                    // }
                    // ofs_wall_depth.close();
                    // ofstream ofs_wall_boundary("depth_and_boundary_maps/left_wall_boundary_map_" + std::to_string(ix) + "_" + std::to_string(iy) + "_" + std::to_string(itheta) + ".txt");
                    // for (int idy = 0; idy < WALL_MAP_WIDTH; idy++) {
                    //     for (int idx = 0; idx < WALL_MAP_LENGTH_ONE_THIRD; idx++) {
                    //         ofs_wall_boundary << left_wall_map_a[1][idy][idx] << " ";
                    //     }
                    //     ofs_wall_boundary << std::endl;
                    // }
                    // ofs_wall_boundary.close();
                    auto right_wall_map_a = wall_maps[1].accessor<float, 3>();
                    ofstream ofs_wall_depth("depth_and_boundary_maps/right_wall_depth_map_" + std::to_string(ix) + "_" + std::to_string(iy) + "_" + std::to_string(itheta) + ".txt");
                    for (int idy = 0; idy < WALL_MAP_WIDTH; idy++) {
                        for (int idx = 0; idx < WALL_MAP_LENGTH_ONE_THIRD; idx++) {
                            ofs_wall_depth << right_wall_map_a[0][idy][idx] << " ";
                        }
                        ofs_wall_depth << std::endl;
                    }
                    ofs_wall_depth.close();
                    ofstream ofs_wall_boundary("depth_and_boundary_maps/right_wall_boundary_map_" + std::to_string(ix) + "_" + std::to_string(iy) + "_" + std::to_string(itheta) + ".txt");
                    for (int idy = 0; idy < WALL_MAP_WIDTH; idy++) {
                        for (int idx = 0; idx < WALL_MAP_LENGTH_ONE_THIRD; idx++) {
                            ofs_wall_boundary << right_wall_map_a[1][idy][idx] << " ";
                        }
                        ofs_wall_boundary << std::endl;
                    }
                    ofs_wall_boundary.close();
                    inputs.push_back(wall_maps[0].unsqueeze(0).repeat({pending_transitions.size(),1,1,1}).cuda());
                    inputs.push_back(wall_maps[1].unsqueeze(0).repeat({pending_transitions.size(),1,1,1}).cuda());
                    float* p2_arr = new float[pending_transitions.size() * 3];
                    // 4 neural networks
                    int model_index = itheta % NUM_MODELS;
                    int orientation_index = itheta - (model_index + 8);
                    for (unsigned int i = 0; i < pending_transitions.size(); i++) {
                        switch (orientation_index) {
                            case 4:  
                                p2_arr[3 * i] = pending_transitions[i][1];
                                p2_arr[3 * i + 1] = -pending_transitions[i][0];
                                p2_arr[3 * i + 2] = pending_transitions[i][2];
                                break;
                            case 0:
                                p2_arr[3 * i] = pending_transitions[i][0];
                                p2_arr[3 * i + 1] = pending_transitions[i][1];
                                p2_arr[3 * i + 2] = pending_transitions[i][2];
                                break;
                            case -4:
                                p2_arr[3 * i] = -pending_transitions[i][1];
                                p2_arr[3 * i + 1] = pending_transitions[i][0];
                                p2_arr[3 * i + 2] = pending_transitions[i][2];
                                break;
                            case -8:
                                p2_arr[3 * i] = -pending_transitions[i][0];
                                p2_arr[3 * i + 1] = -pending_transitions[i][1];
                                p2_arr[3 * i + 2] = pending_transitions[i][2];
                                break;
                            default:
                                std::cout << "MapGrid.cpp line 1154" << std::endl;
                                exit(1);
                        }
                    }
                    inputs.push_back(torch::from_blob(p2_arr, {pending_transitions.size(),3}).cuda());
                    // !!!!!!!!!!!!!!!!!!!!
                    // at::Tensor output = modules[model_index].forward(inputs).toTensor().cpu();
                    // auto output_a = output.accessor<float, 2>();
                    
                    for (unsigned int i = 0; i < pending_transitions.size(); i++) {
                        GridIndices3D child_cell_indices = {ix + pending_transitions[i][0],
                                                            iy + pending_transitions[i][1],
                                                            (itheta + pending_transitions[i][2] + map_grid_ptr->dim_theta_) % map_grid_ptr->dim_theta_};
                        // float dynamic_cost = output_a[i][0];
                        // !!!!!!!!!!!!!!!!!!!!
                        float dynamic_cost = 0;
                        dynamic_cost_map_[{ix, iy, itheta}][child_cell_indices] = dynamic_cost;
                    }
                }
            }
        }
    }

    // ofstream ofs("env_11_pred_cost.txt");
    // ofs << std::fixed;
    // ofs << std::setprecision(2);
    // for (auto current_it = dynamic_cost_map_.begin(); current_it != dynamic_cost_map_.end(); current_it++) {
    //     auto temp = dynamic_cost_map_[current_it->first];
    //     for (auto child_it = temp.begin(); child_it != temp.end(); child_it++) {
    //         ofs << current_it->first[0] << " " << current_it->first[1] << " " << current_it->first[2] << " "
    //         << child_it->first[0] << " " << child_it->first[1] << " " << child_it->first[2] << " "
    //         << child_it->second << std::endl;
    //     }
    // }

    // for (auto p1_cell_it = ground_depth_and_boundary_map_map_.begin(); p1_cell_it != ground_depth_and_boundary_map_map_.end(); p1_cell_it++) {
    //     // std::cout << "one ground map: " << p1_cell_it->first[0] << "," << p1_cell_it->first[1] << "," << p1_cell_it->first[2] << std::endl;
    //     ofstream ofs("depth_and_boundary_maps/" + std::to_string(p1_cell_it->first[0]) + "_" + std::to_string(p1_cell_it->first[1]) + "_" + std::to_string(p1_cell_it->first[2]) + "_ground.txt");
    //     auto temp_a = ground_depth_and_boundary_map_map_[p1_cell_it->first].accessor<float, 2>();
    //     for (int idy = 0; idy < GROUND_MAP_SIDE; idy++) {
    //         for (int idx = 0; idx < GROUND_MAP_SIDE; idx++) {
    //             ofs << temp_a[idy][idx] << " ";
    //         }
    //         ofs << std::endl;
    //     }
    //     ofs.close();
    // }

    // for (auto p1_xy_cell_it = wall_depth_and_boundary_map_map_.begin(); p1_xy_cell_it != wall_depth_and_boundary_map_map_.end(); p1_xy_cell_it++) {
    //     for (auto p1_z_it = p1_xy_cell_it->second.begin(); p1_z_it != p1_xy_cell_it->second.end(); p1_z_it++) {
    //         // std::cout << "one wall map: " << p1_xy_cell_it->first[0] << "," << p1_xy_cell_it->first[1] << "," << p1_z_it->first << std::endl;
    //         ofstream ofs("depth_and_boundary_maps/" + std::to_string(p1_xy_cell_it->first[0]) + "_" + std::to_string(p1_xy_cell_it->first[1]) + "_" + std::to_string(p1_z_it->first) + "_wall.txt");
    //         auto temp_a = wall_depth_and_boundary_map_map_[p1_xy_cell_it->first][p1_z_it->first].accessor<float, 2>();
    //         for (int idy = 0; idy < WALL_MAP_WIDTH; idy++) {
    //             for (int idx = 0; idx < WALL_MAP_LENGTH; idx++) {
    //                 ofs << temp_a[idy][idx] << " ";
    //             }
    //             ofs << std::endl;
    //         }
    //         ofs.close();
    //     }
    // }
}

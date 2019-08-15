#include "Utilities.hpp"
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
min_theta_(TORSO_GRID_MIN_THETA),
max_theta_(360 + TORSO_GRID_MIN_THETA - _theta_resolution),
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
    float position_theta = min_theta_ + theta_index * theta_resolution_;

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
                    float edge_cost = euclideanDistBetweenCells(current_cell, child_cell); // modify this to include the estimate dynamic cost
                    if(current_cell->g_ + edge_cost < child_cell->g_)
                    {
                        child_cell->g_ = current_cell->g_ + edge_cost;
                        child_cell->parent_ = current_cell;
                        open_heap.push(child_cell);
                    }
                }
            }
        }
    }

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
            while(true)
            {
                torso_path.push_back(path_cell);

                if(path_cell->is_root_)
                {
                    break;
                }

                path_cell = path_cell->parent_;
            }

            std::reverse(torso_path.begin(), torso_path.end());
            std::cout << "Path Length: " << torso_path.size()-1 << std::endl;
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
                    float edge_cost = euclideanDistBetweenCells(current_cell, child_cell); // modify this to include the estimate dynamic cost
                    child_cell->h_ = euclideanHeuristic(child_cell, goal_cell);

                    if(current_cell->g_ + edge_cost < child_cell->g_)
                    {
                        child_cell->g_ = current_cell->g_ + edge_cost;
                        child_cell->parent_ = current_cell;
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
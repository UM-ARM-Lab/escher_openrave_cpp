#include "Utilities.hpp"
// #include "PointGrid.hpp"

int MapCell3D::getTravelDirection(MapCell3D goal_cell)
{
    if(goal_cell.ix_ != ix_ || goal_cell.iy_ != iy_)
    {
        float direction_theta = atan2(goal_cell.y_ - y_, goal_cell.x_ - x_) * RAD2DEG;
        float relative_direction_theta = direction_theta - goal_cell.theta_;

        while(relative_direction_theta >= 360 - TORSO_GRID_ANGULAR_RESOLUTION/2 || relative_direction_theta < -TORSO_GRID_ANGULAR_RESOLUTION/2)
        {
            if(relative_direction_theta >= 360 - TORSO_GRID_ANGULAR_RESOLUTION/2)
            {
                relative_direction_theta  = relative_direction_theta - 360;
            }
            else if(relative_direction_theta < -TORSO_GRID_ANGULAR_RESOLUTION/2)
            {
                relative_direction_theta  = relative_direction_theta + 360;
            }
        }

        int direction_index = int((relative_direction_theta - (-TORSO_GRID_ANGULAR_RESOLUTION/2))/float(TORSO_GRID_ANGULAR_RESOLUTION));
    }
    else
    {
        return -1;
    }

}

MapGrid::MapGrid(float _min_x, float _max_x, float _min_y, float _max_y, float _xy_resolution, float _theta_resolution):
xy_resolution_(_xy_resolution),
theta_resolution_(_theta_resolution),
min_x_(_min_x),
max_x_(_max_x),
min_y_(_min_y),
max_y_(_max_y),
min_theta_(-180),
max_theta_(180 - TORSO_GRID_ANGULAR_RESOLUTION),
dim_x_(int(round((_max_x-_min_x)/_xy_resolution))),
dim_y_(int(round((_max_y-_min_y)/_xy_resolution))),
dim_theta_(360/TORSO_GRID_ANGULAR_RESOLUTION)
{
    for(int i = 0; i < dim_x_; i++)
    {
        std::vector<MapCell2D> tmp_cell_2D_list;
        std::vector< std::vector<MapCell3D> > tmp_cell_3D_list_2;
        for(int j = 0; j < dim_y_; j++)
        {
            GridPositions2D xy_positions = indicesToPositionsXY({i,j});
            float x = xy_positions[0];
            float y = xy_positions[1];
            std::vector<MapCell3D> tmp_cell_3D_list;
            for(int k = 0; k < dim_theta_; k++)
            {
                float theta = indicesToPositionsTheta(k);

                tmp_cell_3D_list.push_back(MapCell3D(x, y, theta, i, j, k));
            }
            tmp_cell_2D_list.push_back(MapCell2D(x, y, i, j));
            tmp_cell_3D_list_2.push_back(tmp_cell_3D_list);
        }

        cell_2D_list_.push_back(tmp_cell_2D_list);
        cell_3D_list_.push_back(tmp_cell_3D_list_2);
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

        Translation3D projection_ray(0,0,-1);
        for(int ix = 0; ix < dim_x_; ix++)
        {
            for(int iy = 0; iy < dim_y_; iy++)
            {
                GridPositions2D cell_position = cell_2D_list_[ix][iy].getPositions();
                Translation3D projection_start_point(cell_position[0], cell_position[1], 99.0);
                bool has_projection = false;
                float height = -99.0;
                for(auto structure : structures)
                {
                    if(structure->getType() == TrimeshType::GROUND)
                    {
                        Translation3D projected_point = structure->projectionGlobalFrame(projection_start_point, projection_ray);
                        if(structure->insidePolygon(projected_point))
                        {
                            has_projection = true;
                            height = projected_point[2] > height ? projected_point[2] : height;
                        }
                    }
                }

                temp_height_map[ix][iy] = height;
            }
        }

        // Filter(Smooth) the height map (or you can just fill in holes)
        int window_size = 3; // must be a odd number
        for(int ix = 1; ix < dim_x_-1; ix++)
        {
            for(int iy = 1; iy < dim_y_-1; iy++)
            {
                float height = 0;
                int cell_with_ground_number = 0;

                for(int iix = ix-(window_size-1)/2; iix <= ix+(window_size+1)/2; ix++)
                {
                    for(int iiy = iy-(window_size-1)/2; iiy <= iy+(window_size+1)/2; iy++)
                    {
                        if(temp_height_map[ix][iy] != -99.0)
                        {
                            height += temp_height_map[ix][iy];
                            cell_with_ground_number++;
                        }
                    }
                }

                if(cell_with_ground_number != 0)
                {
                    cell_2D_list_[ix][iy].height_ = height / cell_with_ground_number;
                }
                else
                {
                    cell_2D_list_[ix][iy].height_ = -99.0;
                }
            }
        }


        for(int ix = 0; ix < dim_x_; ix++)
        {
            for(int iy = 0; iy < dim_y_; iy++)
            {
                if(cell_2D_list_[ix][iy].height_ == -99.0) // GAP
                {
                    // std::cout << "0 ";
                    for(int itheta = 0; itheta < dim_theta_; itheta++)
                    {
                        cell_3D_list_[ix][iy][itheta].terrain_type_ = TerrainType::GAP;
                    }
                }
                else // see if the obstacle is close
                {
                    // std::cout << "1 ";
                    for(int itheta = 0; itheta < dim_theta_; itheta++)
                    {
                        GridPositions3D cell_3d_position = cell_3D_list_[ix][iy][itheta].getPositions();
                        RPYTF body_collision_box_transform(cell_3d_position[0], cell_3d_position[1], cell_2D_list_[ix][iy].height_, 0, 0, cell_3d_position[2]);
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
                            cell_3D_list_[ix][iy][itheta].terrain_type_ = TerrainType::OBSTACLE;
                        }
                    }
                }
            }
            // std::cout << std::endl;
        }
    }
}

void MapGrid::generateDijkstrHeuristics(MapCell3D goal_cell, std::map< int,std::vector<GridIndices3D> > reverse_transition_model)
{
    resetCellCostsAndParent();

    std::priority_queue< MapCell3D*, std::vector< MapCell3D* >, pointer_less > open_heap;
    goal_cell.g_ = 0;
    goal_cell.h_ = 0;
    goal_cell.is_root_ = true;

    open_heap.push(&goal_cell);

    // assume 8-connected transition model
    while(!open_heap.empty())
    {
        MapCell3D* current_cell = open_heap.top();
        GridIndices3D current_cell_indices = current_cell->getIndices();

        for(auto & transition : reverse_transition_model[current_cell->itheta_])
        {
            int ix = transition[0];
            int iy = transition[1];
            int itheta = transition[2];

            GridIndices3D child_cell_indices = {current_cell_indices[0]+ix, current_cell_indices[1]+iy, (current_cell_indices[2]+itheta)%dim_theta_};

            if(insideGrid(child_cell_indices))
            {
                MapCell3D* child_cell_ptr = &cell_3D_list_[child_cell_indices[0]][child_cell_indices[1]][child_cell_indices[2]];
                if(child_cell_ptr->terrain_type_ == TerrainType::SOLID)
                // if(true)
                {
                    float edge_cost = std::sqrt(ix*ix*1.0 + iy*iy*1.0) * xy_resolution_;
                    if(current_cell->getF() + edge_cost < child_cell_ptr->getF())
                    {
                        child_cell_ptr->g_ = current_cell->g_ + edge_cost;
                        child_cell_ptr->parent_indices_ = current_cell_indices;
                        open_heap.push(child_cell_ptr);
                    }
                }
            }
        }

        open_heap.pop();
    }
}

void MapGrid::generateTorsoGuidingPath(MapCell3D initial_cell, MapCell3D goal_cell, std::map< int,std::vector<GridIndices3D> > transition_model)
{
    resetCellCostsAndParent();

    std::priority_queue< MapCell3D*, std::vector< MapCell3D* >, pointer_less > open_heap;
    initial_cell.g_ = 0;
    initial_cell.h_ = euclideanHeuristic(initial_cell, goal_cell);
    initial_cell.is_root_ = true;

    open_heap.push(&initial_cell);

    // assume 8-connected transition model
    while(!open_heap.empty())
    {
        MapCell3D* current_cell_ptr = open_heap.top();
        GridIndices3D current_cell_indices = current_cell_ptr->getIndices();

        for(auto & transition : transition_model[current_cell_ptr->itheta_])
        {
            int ix = transition[0];
            int iy = transition[1];
            int itheta = transition[2];

            GridIndices3D child_cell_indices = {current_cell_indices[0]+ix, current_cell_indices[1]+iy, (current_cell_indices[2]+itheta)%dim_theta_};

            if(insideGrid(child_cell_indices))
            {
                MapCell3D* child_cell_ptr = &cell_3D_list_[child_cell_indices[0]][child_cell_indices[1]][child_cell_indices[2]];

                if(child_cell_ptr->terrain_type_ == TerrainType::SOLID)
                // if(true)
                {
                    float edge_cost = euclideanDistBetweenCells(*current_cell_ptr, *child_cell_ptr); // modify this to include the estimate dynamic cost
                    child_cell_ptr->h_ = euclideanHeuristic(*child_cell_ptr, goal_cell);

                    if(current_cell_ptr->getF() + edge_cost < child_cell_ptr->getF())
                    {
                        child_cell_ptr->g_ = current_cell_ptr->g_ + edge_cost;
                        child_cell_ptr->parent_indices_ = current_cell_indices;
                        open_heap.push(child_cell_ptr);
                    }
                }
            }
        }

        open_heap.pop();
    }
}

void MapGrid::resetCellCostsAndParent()
{
    for(int ix = 0; ix < dim_x_; ix++)
    {
        for(int iy = 0; iy < dim_y_; iy++)
        {
            for(int itheta = 0; itheta < dim_theta_; itheta++)
            {
                cell_3D_list_[ix][iy][itheta].g_ = std::numeric_limits<float>::max();
                cell_3D_list_[ix][iy][itheta].h_ = 0;
                cell_3D_list_[ix][iy][itheta].step_num_ = 0;
                cell_3D_list_[ix][iy][itheta].parent_indices_ = {-99,-99,-99};
                cell_3D_list_[ix][iy][itheta].is_root_ = false;
            }
        }
    }
}

float MapGrid::euclideanDistBetweenCells(MapCell3D& cell1, MapCell3D& cell2)
{
    GridIndices3D cell1_indices = cell1.getIndices();
    GridIndices3D cell2_indices = cell2.getIndices();
    int ix = cell2_indices[0] - cell1_indices[0];
    int iy = cell2_indices[1] - cell1_indices[1];
    return std::hypot(ix*ix*1.0, iy*iy*1.0) * xy_resolution_;
}

float MapGrid::euclideanHeuristic(MapCell3D& current_cell, MapCell3D& goal_cell)
{
    return euclideanDistBetweenCells(current_cell, goal_cell);
}
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
                    Translation3D projected_point = structure->projectionGlobalFrame(projection_start_point, projection_ray);
                    if(structure->insidePolygon(projected_point))
                    {
                        has_projection = true;
                        height = projected_point[2] > height ? projected_point[2] : height;
                    }
                }

                cell_2D_list_[ix][iy].height_ = height;

                if(!has_projection)
                {
                    // std::cout << "0 ";
                    for(int itheta = 0; itheta < dim_theta_; itheta++)
                    {
                        cell_3D_list_[ix][iy][itheta].terrain_type_ = TerrainType::GAP;
                    }
                }
                else
                {
                    // std::cout << "1 ";
                    for(int itheta = 0; itheta < dim_theta_; itheta++)
                    {
                        GridPositions3D cell_3d_position = cell_3D_list_[ix][iy][itheta].getPositions();
                        RPYTF body_collision_box_transform(cell_3d_position[0], cell_3d_position[1], height, 0, 0, cell_3d_position[2]);
                        body_collision_box->SetTransform(body_collision_box_transform.GetRaveTransform());
                        bool in_collision = false;

                        for(auto structure : structures)
                        {
                            if(env->CheckCollision(body_collision_box, structure->getKinbody()))
                            {
                                in_collision = true;
                                break;
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

void MapGrid::generateDijkstrHeuristics(MapCell3D goal_cell)
{
    std::priority_queue< MapCell3D*, std::vector< MapCell3D* >, MapCell3D::pointer_less > open_heap;
    goal_cell.g_ = 0;

    open_heap.push(&goal_cell);

    // assume 8-connected transition model
    while(!open_heap.empty())
    {
        MapCell3D* current_cell = open_heap.top();

        GridIndices3D current_cell_indices = current_cell->getIndices();

        for(int ix = -1; ix <= 1; ix++)
        {
            for(int iy = -1; iy <= 1; iy++)
            {
                for(int itheta = -1; itheta <= 1; itheta++)
                {
                    if(ix != 0 || iy !=0 || itheta != 0)
                    {
                        GridIndices3D new_cell_indices = {current_cell_indices[0]+ix, current_cell_indices[1]+iy, (current_cell_indices[2]+itheta)%dim_theta_};

                        if(insideGrid(new_cell_indices))
                        {
                            MapCell3D* cell_ptr = &cell_3D_list_[new_cell_indices[0]][new_cell_indices[1]][new_cell_indices[2]];
                            // if(cell_ptr->terrain_type_ == TerrainType::SOLID)
                            if(true)
                            {
                                float edge_cost = std::sqrt(ix*ix*1.0 + iy*iy*1.0) * xy_resolution_;
                                if(current_cell->getF() + edge_cost < cell_ptr->getF())
                                {
                                    cell_ptr->g_ = current_cell->g_ + edge_cost;
                                    cell_ptr->parent_indices_ = current_cell_indices;
                                    open_heap.push(cell_ptr);
                                }
                            }
                        }
                    }
                }
            }
        }

        open_heap.pop();
    }
}
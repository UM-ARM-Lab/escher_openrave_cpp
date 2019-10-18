#ifndef MAPGRID_HPP
#define MAPGRID_HPP

#include "Utilities.hpp"

// // OpenRAVE
// #include <openrave/plugin.h>

#include <memory> // not sure what this is for
#include <math.h>

#include <torch/torch.h>
#include <torch/script.h>


class MapCell2D
{
    public:
        MapCell2D(float _x, float _y, float _ix, float _iy):
        x_(_x),
        y_(_y),
        ix_(_ix),
        iy_(_iy)
        {};

        float x_;
        float y_;
        int ix_;
        int iy_;

        float height_;

        std::pair<bool, std::shared_ptr<TrimeshSurface> > foot_ground_projection_;
        std::vector< std::shared_ptr<TrimeshSurface> > all_ground_structures_;

        inline GridIndices2D getIndices() {return {ix_, iy_};}
        inline GridPositions2D getPositions() {return {x_, y_};}

};

class MapCell3D : public MapCell2D
{
    public:
        MapCell3D(float _x, float _y, float _theta, float _ix, float _iy, float _itheta):
        MapCell2D(_x, _y, _ix, _iy),
        theta_(_theta),
        itheta_(_itheta),
        g_(std::numeric_limits<float>::max()),
        h_(0.0),
        is_root_(false),
        terrain_type_(TerrainType::SOLID),
        explore_state_(ExploreState::OPEN)
        {};

        std::shared_ptr<MapCell3D> parent_;

        float g_;
        float h_;
        int step_num_;

        float theta_;
        int itheta_;

        std::vector< std::shared_ptr<TrimeshSurface> > left_hand_checking_surfaces_;
        std::vector< std::shared_ptr<TrimeshSurface> > right_hand_checking_surfaces_;

        bool left_hand_contact_region_exist_;
        bool right_hand_contact_region_exist_;

        bool near_obstacle_;

        bool is_root_;
        ExploreState explore_state_;

        TerrainType terrain_type_;

        std::array<float,4> hand_contact_env_feature_;

        inline const float getF() const {return (g_ + h_);}
        inline bool operator<(const MapCell3D& other) const {return (this->getF() < other.getF());}
        inline bool operator>(const MapCell3D& other) const {return (this->getF() > other.getF());}

        // struct pointer_less
        // {
        //     template <typename T>
        //     bool operator()(const T& lhs, const T& rhs) const
        //     {
        //         return *lhs < *rhs;
        //     }
        // };

        // int getTravelDirection(MapCell3D goal_cell);

        inline GridIndices3D getIndices() {return {ix_, iy_, itheta_};}
        inline GridPositions3D getPositions() {return {x_, y_, theta_};}

};

typedef std::shared_ptr<MapCell2D> MapCell2DPtr;
typedef std::shared_ptr<MapCell3D> MapCell3DPtr;

class MapGrid
{
    public:
        MapGrid(float _min_x, float _max_x, float _min_y, float _max_y, float _xy_resolution, float _theta_resolution, std::shared_ptr<DrawingHandler> _drawing_handler);

        GridIndices2D positionsToIndicesXY(GridPositions2D xy_position);
        int positionsToIndicesTheta(float theta_position);
        GridIndices3D positionsToIndices(GridPositions3D position);

        GridPositions2D indicesToPositionsXY(GridIndices2D xy_indices);
        float indicesToPositionsTheta(int theta_index);
        GridPositions3D indicesToPositions(GridIndices3D indices);

        inline std::array<int,2> getXYDimensions(){return {dim_x_, dim_y_};}
        inline int getThetaDimensions(){return dim_theta_;}
        inline std::array<int,3> getDimensions(){return {dim_x_, dim_y_, dim_theta_};}

        inline std::array<float,6> getBoundaries(){return {min_x_, max_x_, min_y_, max_y_, min_theta_, max_theta_};}
        inline float getXYResolution(){return xy_resolution_;}
        inline bool insideGrid(GridPositions3D positions){return (positions[0] >= min_x_ && positions[0] < max_x_ && positions[1] >= min_y_ && positions[1] < max_y_ && positions[2] >= min_theta_ && positions[2] < max_theta_);}
        inline bool insideGrid(GridIndices3D indices){return (indices[0] >= 0 && indices[0] < dim_x_ && indices[1] >= 0 && indices[1] < dim_y_ && indices[2] >= 0 && indices[2] < dim_theta_);}
        inline MapCell2DPtr get2DCell(GridIndices2D indices) {return cell_2D_list_[indices[0]][indices[1]];}
        inline MapCell2DPtr get2DCell(GridPositions2D positions) {return get2DCell(positionsToIndicesXY(positions));}
        inline MapCell3DPtr get3DCell(GridIndices3D indices) {return cell_3D_list_[indices[0]][indices[1]][indices[2]];}
        inline MapCell3DPtr get3DCell(GridPositions3D positions) {return get3DCell(positionsToIndices(positions));}

        void obstacleAndGapMapping(OpenRAVE::EnvironmentBasePtr env, std::vector< std::shared_ptr<TrimeshSurface> > structures);
        void read_transition_model(std::map< int,std::vector<GridIndices3D> > transition_model);
        void generateDijkstraHeuristics(MapCell3DPtr& goal_cell, std::map< int,std::vector<GridIndices3D> > reverse_transition_model, std::unordered_set<GridIndices3D, hash<GridIndices3D> > region_mask=std::unordered_set<GridIndices3D, hash<GridIndices3D> >());
        std::vector<MapCell3DPtr> generateTorsoGuidingPath(MapCell3DPtr& initial_cell, MapCell3DPtr& goal_cell, std::map< int,std::vector<GridIndices3D> > transition_model);
        std::unordered_set<GridIndices3D, hash<GridIndices3D> > getRegionMask(std::vector<MapCell3DPtr> torso_path, float neighbor_distance_range, float neighbor_orientation_range);
        std::unordered_set<GridIndices3D, hash<GridIndices3D> > getRegionMask(std::vector<GridIndices3D> grid_indices_vec, float neighbor_distance_range, float neighbor_orientation_range);

        float euclideanDistBetweenCells(MapCell3DPtr& cell1, MapCell3DPtr& cell2);
        float euclideanHeuristic(MapCell3DPtr& current_cell, MapCell3DPtr& goal_cell);

        void resetCellCostsAndParent();

        std::map< int,std::vector<GridIndices3D> > transition_model;

        const float xy_resolution_;
        const float theta_resolution_;

        const int dim_x_;
        const int dim_y_;
        const int dim_theta_;

        const float min_x_;
        const float max_x_;
        const float min_y_;
        const float max_y_;
        const float min_theta_;
        const float max_theta_;

        const float step_cost_weight_ = 3.0;
        const float dynamics_cost_weight_ = 0.1;

        std::map<int, std::vector< GridIndices2D > > left_foot_neighbor_window_;
        std::map<int, std::vector< GridIndices2D > > right_foot_neighbor_window_;
        std::map<int, std::vector< GridIndices2D > > torso_neighbor_window_;

        int max_step_size;

        std::vector< std::vector<MapCell2DPtr> > cell_2D_list_;
        std::vector< std::vector< std::vector<MapCell3DPtr> > > cell_3D_list_;

        // self.feet_cp_grid = None

        std::map< std::array<int,5>, float > footstep_env_transition_feature_dict_;
        std::map< std::array<int,3>, std::array<float,4> > hand_env_transition_feature_dict_;
        // self.env_transition_feature_dict = {}
        // self.env_transition_prediction_dict = {}

        // // OpenRAVE object
        // OpenRAVE::EnvironmentBasePtr env_;

        // the drawing handler
        std::shared_ptr<DrawingHandler> drawing_handler_;

    public:
        class HeuristicHelper {
            public:
                HeuristicHelper();
                void saveStructures(MapGrid* map_grid_ptr, std::vector< std::shared_ptr<TrimeshSurface> > _structures);
                // return value is a 3D tensor
                // output[0] is a 2D tensor, which is the depth map
                // output[1] is a 2D tensor, which is the boundary map
                torch::Tensor getGroundDepthBoundaryMap(MapGrid* map_grid_ptr, GridIndices3D indices);
                // return value is two 3D tensors
                // the first tensor is for left
                // the second tensor is for right
                std::vector<torch::Tensor> getWallDepthBoundaryMap(MapGrid* map_grid_ptr, GridIndices3D indices);
       
                const float ANGLE_RESOLUTION = 22.5;
                const float GRID_RESOLUTION = 0.15;
                const float MAP_RESOLUTION = 0.025;

                const float GROUND_DEPTH_AND_BOUNDARY_MAP_SIDE = 1.6;
                const float GROUND_DEFAULT_DEPTH = -1.0;
                const int GROUND_MAP_EDGE = ceil(GROUND_DEPTH_AND_BOUNDARY_MAP_SIDE / 2 / MAP_RESOLUTION);
                const int GROUND_MAP_SIDE = GROUND_MAP_EDGE * 2 + 1;
                
                const float WALL_DEPTH_AND_BOUNDARY_MAP_RADIUS = 1.0;
                const float WALL_DEFAULT_DEPTH = 2.0;
                const float WALL_MIN_HEIGHT = 1.1;
                const float WALL_MAX_HEIGHT = 1.7;
                const int WALL_MAP_LENGTH = ceil(2 * M_PI * WALL_DEPTH_AND_BOUNDARY_MAP_RADIUS / MAP_RESOLUTION);
                const int WALL_MAP_LENGTH_ONE_THIRD = WALL_MAP_LENGTH / 3;
                const int WALL_MAP_WIDTH = (WALL_MAX_HEIGHT - WALL_MIN_HEIGHT) / MAP_RESOLUTION + 1;

                const int NUM_MODELS = 4;
                std::vector<torch::jit::script::Module> modules;

                std::vector< std::shared_ptr<TrimeshSurface> > structures_;
              
                // entire_ground_map_ is a 3D tensor
                // entire_ground_map_[0] is a 2D tensor, which is the entire depth map
                // entire_ground_map_[1] is a 2D tensor, which is the entire boundary map
                torch::Tensor entire_ground_map_;
                // map to 3D tensors
                std::unordered_map<GridIndices3D, torch::Tensor, hash<GridIndices3D> > ground_maps_map_;
                // map to 3D tensors 
                std::unordered_map<GridIndices2D, torch::Tensor, hash<GridIndices2D> > triple_wall_maps_map_;
                // map to 3D tensors
                std::unordered_map<GridIndices3D, torch::Tensor, hash<GridIndices3D> > left_wall_maps_map_;
                // map to 3D tensors
                std::unordered_map<GridIndices3D, torch::Tensor, hash<GridIndices3D> > right_wall_maps_map_;
                std::unordered_map<GridIndices3D, std::unordered_map<GridIndices3D, float, hash<GridIndices3D> >, hash<GridIndices3D> > dynamic_cost_map_;
                
                // return value is a 2D array
                torch::Tensor getBoundaryMap(const std::vector<std::vector<int>>& structure_id_map, int dx, int dy);
                void predict_dynamic_costs_of_all_transitions(MapGrid* map_grid_ptr);
        };

        HeuristicHelper heuristic_helper_;
};

#endif
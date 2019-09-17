import pickle, IPython, os, math, shutil
import numpy as np
import matplotlib.pyplot as plt

from structures_2 import trimesh_surface


MAP_RESOLUTION = 0.025
GROUND_DEPTH_AND_BOUNDARY_MAP_SIDE = 1.6
GROUND_MAP_EDGE = int(math.ceil(GROUND_DEPTH_AND_BOUNDARY_MAP_SIDE / 2 / MAP_RESOLUTION))
GROUND_MAP_SIDE = GROUND_MAP_EDGE * 2 + 1
GROUND_DEFAULT_DEPTH = -1.0
WALL_DEPTH_AND_BOUNDARY_MAP_RADIUS = 1.0
WALL_MIN_HEIGHT_RELATIVE = 1.1
WALL_MAX_HEIGHT_RELATIVE = 1.7
WALL_MAP_LENGTH = int(math.ceil(2 * math.pi * WALL_DEPTH_AND_BOUNDARY_MAP_RADIUS / MAP_RESOLUTION))
WALL_MAP_WIDTH = int((WALL_MAX_HEIGHT_RELATIVE - WALL_MIN_HEIGHT_RELATIVE) / MAP_RESOLUTION) + 1
WALL_DEFAULT_DEPTH = 2.0


def generateBoundaryMap(structure_id_map):
    dx = structure_id_map.shape[1] - 2
    dy = structure_id_map.shape[0] - 2

    boundary_map = np.ones_like(structure_id_map, dtype=int)
    boundary_map[1:dy+1, 1:dx+1] = np.zeros((dy, dx), dtype=int)

    level_zero_positions = set()
    
    for idy in range(1, dy + 1):
        for idx in range(1, dx + 1):
            if structure_id_map[idy][idx] != structure_id_map[idy - 1][idx] or structure_id_map[idy][idx] != structure_id_map[idy + 1][idx] or structure_id_map[idy][idx] != structure_id_map[idy][idx - 1] or structure_id_map[idy][idx] != structure_id_map[idy][idx + 1]:
                boundary_map[idy][idx] = 1
                level_zero_positions.add((idx, idy))

    level_one_positions = set()
    for position in level_zero_positions:
        temp_x = position[0]
        temp_y = position[1]
        if boundary_map[temp_y - 1][temp_x] == 0:
            boundary_map[temp_y - 1][temp_x] = 1
            level_one_positions.add((temp_x, temp_y - 1))
        if boundary_map[temp_y + 1][temp_x] == 0:
            boundary_map[temp_y + 1][temp_x] = 1
            level_one_positions.add((temp_x, temp_y + 1))
        if boundary_map[temp_y][temp_x - 1] == 0:
            boundary_map[temp_y][temp_x - 1] = 1
            level_one_positions.add((temp_x - 1, temp_y))
        if boundary_map[temp_y][temp_x + 1] == 0:
            boundary_map[temp_y][temp_x + 1] = 1
            level_one_positions.add((temp_x + 1, temp_y))

    for position in level_one_positions:
        temp_x = position[0]
        temp_y = position[1]
        if boundary_map[temp_y - 1][temp_x] == 0:
            boundary_map[temp_y - 1][temp_x] = 1
        if boundary_map[temp_y + 1][temp_x] == 0:
            boundary_map[temp_y + 1][temp_x] = 1
        if boundary_map[temp_y][temp_x - 1] == 0:
            boundary_map[temp_y][temp_x - 1] = 1
        if boundary_map[temp_y][temp_x + 1] == 0:
            boundary_map[temp_y][temp_x + 1] = 1

    return boundary_map[1:dy+1, 1:dx+1]


def generateGroundDepthBoundaryMap(ground_structures_parameters, x_position, y_position):
    ground_structures = []
    for index, parameters in enumerate(ground_structures_parameters):
        ground_structures.append(trimesh_surface(index, parameters[0], parameters[1], parameters[2]))
    
    ground_depth_map = np.ones((GROUND_MAP_SIDE, GROUND_MAP_SIDE), dtype=float) * GROUND_DEFAULT_DEPTH
    # note the edge
    structure_id_map = np.ones((GROUND_MAP_SIDE + 2, GROUND_MAP_SIDE + 2), dtype=int) * -1
    projection_ray = np.array([[0], [0], [-1]])
    for iy in range(GROUND_MAP_SIDE):
        for ix in range(GROUND_MAP_SIDE):
            projection_start_point = np.array([[x_position - (GROUND_MAP_EDGE - ix) * MAP_RESOLUTION],
                                               [y_position - (GROUND_MAP_EDGE - iy) * MAP_RESOLUTION],
                                               [99.0]])
            for structure in ground_structures:
                if np.sum((structure.get_center()[:2] - projection_start_point[:2]) ** 2) <= structure.circumscribed_radius ** 2:
                    projected_point = structure.projection_global_frame(projection_start_point, projection_ray)
                    if structure.inside_polygon(projected_point):
                        ground_depth_map[iy][ix] = max(projected_point[2], ground_depth_map[iy][ix])
                        structure_id_map[iy + 1][ix + 1] = structure.get_id()

    ground_boundary_map = generateBoundaryMap(structure_id_map)
    return np.clip(ground_depth_map + ground_boundary_map * -2, -1, 1)


def generateWallDepthBoundaryMap(others_structures_parameters, x_position, y_position, altitude):
    wall_structures = []
    for index, parameters in enumerate(others_structures_parameters):
        wall_structures.append(trimesh_surface(index, parameters[0], parameters[1], parameters[2]))

    wall_depth_map = np.ones((WALL_MAP_WIDTH, WALL_MAP_LENGTH), dtype=float) * WALL_DEFAULT_DEPTH
    # note the edge
    structure_id_map = np.ones((WALL_MAP_WIDTH + 2, WALL_MAP_LENGTH + 2), dtype=int) * -1
    theta_interval = 2 * math.pi / WALL_MAP_LENGTH
    for ix in range(WALL_MAP_LENGTH):
        projection_angle = math.pi - ix * theta_interval
        projection_ray = np.array([[math.cos(projection_angle)], [math.sin(projection_angle)], [0]])
        for iy in range(WALL_MAP_WIDTH):
            projection_start_point = np.array([[x_position], [y_position], [altitude + WALL_MAX_HEIGHT_RELATIVE - iy * MAP_RESOLUTION]])
            dist = WALL_DEPTH_AND_BOUNDARY_MAP_RADIUS + 0.001
            structure_id = 0
            for structure in wall_structures:
                projected_point = structure.projection_global_frame(projection_start_point, projection_ray)
                if structure.inside_polygon(projected_point):
                    new_dist = math.sqrt(np.sum((projection_start_point - projected_point) ** 2))
                    if new_dist < dist:
                        dist = new_dist
                        structure_id = structure.get_id()

            if dist < WALL_DEPTH_AND_BOUNDARY_MAP_RADIUS:
                wall_depth_map[iy][ix] = dist
                structure_id_map[iy + 1][ix + 1] = structure_id
        
    wall_boundary_map = generateBoundaryMap(structure_id_map)
    return np.clip(wall_depth_map + wall_boundary_map * 2, 0, 2)


def main():
    with open('../data/dataset_225/complete_environments_0_0', 'r') as file:
        envs = pickle.load(file)
    ground_map = generateGroundDepthBoundaryMap(envs['ground_structures'], 0.0, 0.0)
    with open('../data/test/ground_0_0', 'w') as depth_map_file:
        pickle.dump(np.expand_dims(ground_map, axis=0).astype(np.float32), depth_map_file)
    wall_map = generateWallDepthBoundaryMap(envs['others_structures'], 0.0, 0.0, 0.0)
    with open('../data/test/wall_0_0', 'w') as depth_map_file:
        pickle.dump(np.expand_dims(wall_map, axis=0).astype(np.float32), depth_map_file)

    


    

if __name__ == '__main__':
    main()
import pickle, os, math, shutil, IPython
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
WALL_MAP_WIDTH = int(round((WALL_MAX_HEIGHT_RELATIVE - WALL_MIN_HEIGHT_RELATIVE) / MAP_RESOLUTION)) + 1
WALL_DEFAULT_DEPTH = 2.0
BOUNDARY_WIDTH = 6


def generateBoundaryMap(structure_id_map):
    dx = structure_id_map.shape[1]
    dy = structure_id_map.shape[0]

    boundary_map = np.zeros((dy, dx), dtype=int)

    level_zero_positions = set()

    for idy in range(dy):
        for idx in range(dx):
            if structure_id_map[idy][idx] == -1:
                boundary_map[idy][idx] = 1
                level_zero_positions.add((idx, idy))
    
    for idy in range(dy):
        for idx in range(dx):
            if idy > 1 and structure_id_map[idy][idx] != structure_id_map[idy - 1][idx]:
                boundary_map[idy][idx] = 1
                level_zero_positions.add((idx, idy))
            elif idy < dy - 1 and structure_id_map[idy][idx] != structure_id_map[idy + 1][idx]:
                boundary_map[idy][idx] = 1
                level_zero_positions.add((idx, idy))
            elif idx > 1 and structure_id_map[idy][idx] != structure_id_map[idy][idx - 1]:
                boundary_map[idy][idx] = 1
                level_zero_positions.add((idx, idy))
            elif idx < dx - 1 and structure_id_map[idy][idx] != structure_id_map[idy][idx + 1]:
                boundary_map[idy][idx] = 1
                level_zero_positions.add((idx, idy))

    level_one_positions = set()
    for position in level_zero_positions:
        temp_x = position[0]
        temp_y = position[1]
        if temp_y > 1 and boundary_map[temp_y - 1][temp_x] == 0:
            boundary_map[temp_y - 1][temp_x] = 1
            level_one_positions.add((temp_x, temp_y - 1))
        if temp_y < dy - 1 and boundary_map[temp_y + 1][temp_x] == 0:
            boundary_map[temp_y + 1][temp_x] = 1
            level_one_positions.add((temp_x, temp_y + 1))
        if temp_x > 1 and boundary_map[temp_y][temp_x - 1] == 0:
            boundary_map[temp_y][temp_x - 1] = 1
            level_one_positions.add((temp_x - 1, temp_y))
        if temp_x < dx - 1 and boundary_map[temp_y][temp_x + 1] == 0:
            boundary_map[temp_y][temp_x + 1] = 1
            level_one_positions.add((temp_x + 1, temp_y))

    for position in level_one_positions:
        temp_x = position[0]
        temp_y = position[1]
        if temp_y > 1 and boundary_map[temp_y - 1][temp_x] == 0:
            boundary_map[temp_y - 1][temp_x] = 1
        if temp_y < dy - 1 and boundary_map[temp_y + 1][temp_x] == 0:
            boundary_map[temp_y + 1][temp_x] = 1
        if temp_x > 1 and boundary_map[temp_y][temp_x - 1] == 0:
            boundary_map[temp_y][temp_x - 1] = 1
        if temp_x < dx - 1 and boundary_map[temp_y][temp_x + 1] == 0:
            boundary_map[temp_y][temp_x + 1] = 1

    return boundary_map


def generateGroundDepthBoundaryMap(ground_structures_parameters, x_position, y_position):
    ground_structures = []
    for index, parameters in enumerate(ground_structures_parameters):
        ground_structures.append(trimesh_surface(index, parameters[0], parameters[1], parameters[2]))
   
    ground_depth_map = np.ones((GROUND_MAP_SIDE + 2 * BOUNDARY_WIDTH, GROUND_MAP_SIDE + 2 * BOUNDARY_WIDTH), dtype=float) * GROUND_DEFAULT_DEPTH
    structure_id_map = np.ones((GROUND_MAP_SIDE + 2 * BOUNDARY_WIDTH, GROUND_MAP_SIDE + 2 * BOUNDARY_WIDTH), dtype=int) * -1

    projection_ray = np.array([[0], [0], [-1.0]])
    for iy in range(ground_depth_map.shape[0]):
        for ix in range(ground_depth_map.shape[1]):
            projection_start_point = np.array([[x_position + (ix - GROUND_MAP_EDGE - BOUNDARY_WIDTH) * MAP_RESOLUTION],
                                               [y_position + (GROUND_MAP_EDGE + BOUNDARY_WIDTH - iy) * MAP_RESOLUTION],
                                               [99.0]])

            for structure in ground_structures:
                if np.sum((structure.get_center()[:2,0] - projection_start_point[:2,0]) ** 2) <= structure.circumscribed_radius ** 2:
                    projected_point = structure.projection_global_frame(projection_start_point, projection_ray)
                    if structure.inside_polygon(projected_point):
                        if projected_point[2,0] > ground_depth_map[iy][ix]:
                            ground_depth_map[iy][ix] = projected_point[2,0]
                            structure_id_map[iy][ix] = structure.get_id()
    # for structure in ground_structures:
    #     for vertex in structure.vertices:
    #         print(structure.nx * vertex[0] + structure.ny * vertex[1] + structure.nz * vertex[2] + structure.c)
    # plt.imshow(ground_depth_map, cmap='gray')
    # plt.show()

    ground_boundary_map = generateBoundaryMap(structure_id_map)
    # return np.clip(ground_depth_map + ground_boundary_map * -2, -1, 1)
    return np.stack((ground_depth_map[BOUNDARY_WIDTH:GROUND_MAP_SIDE+BOUNDARY_WIDTH, BOUNDARY_WIDTH:GROUND_MAP_SIDE+BOUNDARY_WIDTH], 
                     ground_boundary_map[BOUNDARY_WIDTH:GROUND_MAP_SIDE+BOUNDARY_WIDTH, BOUNDARY_WIDTH:GROUND_MAP_SIDE+BOUNDARY_WIDTH]),
                     axis=0)


def generateWallDepthBoundaryMap(others_structures_parameters, x_position, y_position, altitude):
    wall_structures = []
    for index, parameters in enumerate(others_structures_parameters):
        wall_structures.append(trimesh_surface(index, parameters[0], parameters[1], parameters[2]))

    wall_depth_map = np.ones((WALL_MAP_WIDTH, WALL_MAP_LENGTH), dtype=float) * WALL_DEFAULT_DEPTH
    structure_id_map = np.ones((WALL_MAP_WIDTH, WALL_MAP_LENGTH), dtype=int) * -1
    theta_interval = MAP_RESOLUTION / WALL_DEPTH_AND_BOUNDARY_MAP_RADIUS
    for ix in range(WALL_MAP_LENGTH):
        projection_angle = math.pi - ix * theta_interval
        projection_ray = np.array([[math.cos(projection_angle)], [math.sin(projection_angle)], [0]])
        for iy in range(WALL_MAP_WIDTH):
            projection_start_point = np.array([[x_position], [y_position], [altitude + WALL_MAX_HEIGHT_RELATIVE - iy * MAP_RESOLUTION]])
            dist = WALL_DEPTH_AND_BOUNDARY_MAP_RADIUS + 0.001
            structure_id = 0
            for structure in wall_structures:
                projected_point = structure.projection_global_frame(projection_start_point, projection_ray)
                if projected_point is None:
                    continue
                if structure.inside_polygon(projected_point):
                    new_dist = math.sqrt(np.sum((projection_start_point - projected_point) ** 2))
                    if new_dist < dist:
                        dist = new_dist
                        structure_id = structure.get_id()

            if dist < WALL_DEPTH_AND_BOUNDARY_MAP_RADIUS:
                wall_depth_map[iy][ix] = dist
                structure_id_map[iy][ix] = structure_id

    # plt.imshow(wall_depth_map, cmap='gray')
    # plt.show()
    wall_boundary_map = generateBoundaryMap(structure_id_map)
    # return np.clip(wall_depth_map + wall_boundary_map * 2, 0, 2)
    return np.stack((wall_depth_map, wall_boundary_map), axis=0)


def main():
    for environment_index in range(20):
        print(environment_index)
        with open('/mnt/big_narstie_data/chenxi/data/dataset_225/complete_environments_3_' + str(environment_index), 'r') as file:
            envs = pickle.load(file)
        # ground_depth_map, ground_boundary_map = generateGroundDepthBoundaryMap(envs['ground_structures'], 0.0, 0.0)
        # with open('../data/test/ground_depth_3_' + str(environment_index) + '_new', 'w') as file:
        #     pickle.dump(np.expand_dims(ground_depth_map, axis=0).astype(np.float32), file)
        # with open('../data/test/ground_boundary_3_' + str(environment_index) + '_new', 'w') as file:
        #     pickle.dump(np.expand_dims(ground_boundary_map, axis=0).astype(np.float32), file)
        
        wall_depth_map, wall_boundary_map = generateWallDepthBoundaryMap(envs['others_structures'], 0.0, 0.0, 0.0)
        # with open('../data/test/wall_depth_3_' + str(environment_index), 'w') as file:
        #     pickle.dump(np.expand_dims(wall_depth_map, axis=0).astype(np.float32), file)
        with open('../data/test/wall_boundary_3_' + str(environment_index) + '_new', 'w') as file:
            pickle.dump(np.expand_dims(wall_boundary_map, axis=0).astype(np.float32), file)
        

    
if __name__ == '__main__':
    main()
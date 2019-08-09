import pickle, IPython, os, math, shutil
import numpy as np
import matplotlib.pyplot as plt

GRID_RESOLUTION = 0.15
ANGLE_RESOLUTION = 15.0

RESOLUTION = 0.025
SIDE = 1.6
GROUND_DEFAULT_DEPTH = -1.0
RADIUS = 1.0
# RADIUS = 0.7
WALL_DEFAULT_DEPTH = 2
# WALL_DEFAULT_DEPTH = 0.7
# WALL_MIN_HEIGHT = 1.0
# WALL_MAX_HEIGHT = 2.0
BOUNDARY_LEVEL_THRESHOLD = 2

def rotate_quadrilaterals(coordinates, theta):
    """
    Inputs:
    "coordinates" should be a list of a list of tuples.
        each tuple represents the xyz coordinate of a vertex;
        each inner list represents the vertices of a quadrilateral;
        the outer list represents a list of quadrilaterals.

    "theta" should be the angle (in degree), by which the points will be rotated clockwise.

    Output:
    a 2d numpy array which represents the coordinates of the vertices after rotation.
    """
    theta_in_radian = theta * np.pi / 180
    # rotating the coordinate system counter clockwise is the same as rotating the points clockwise
    rotation_matrix = np.array([[np.cos(-theta_in_radian), -np.sin(-theta_in_radian)],
                                [np.sin(-theta_in_radian), np.cos(-theta_in_radian)]])
    xyz = np.array(coordinates).reshape(-1, 3)
    rotated_xyz = np.zeros_like(xyz)
    rotated_xyz[:, 0:2] = np.matmul(rotation_matrix, xyz[:, 0:2].transpose()).transpose()
    rotated_xyz[:, 2] = xyz[:, 2]
    return rotated_xyz

def point_inside_polygon(point, vertices, normal):
    """
    Inputs:
    "point" should be a numpy array

    "vertices" should be a list of four numpy arrays which is the coordinates of vertices of a quadrilateral patch.

    Output:
    If the point is inside of the polygon, return True; otherwise, return False.
    """
    if abs(normal[2]) > 1e-10:
        coords = np.array(vertices)[:, [0,1]]
        point_coords = point[[0, 1]]
    elif abs(normal[1]) > 1e-10:
        coords = np.array(vertices)[:, [0,2]]
        point_coords = point[[0, 2]]
    elif abs(normal[0]) > 1e-10:
        coords = np.array(vertices)[:, [1,2]]
        point_coords = point[[1, 2]]
    else:
        print('invalid normal vector')
        exit(1)

    prev_value = None
    num_vertices = coords.shape[0]
    for i in range(num_vertices):
        begin = coords[i]
        end = coords[(i + 1) % num_vertices]
        vector1 = end - begin
        vector2 = point_coords - begin
        value = vector1[0] * vector2[1] - vector1[1] * vector2[0]
        if value == 0:
            return False

        if not prev_value:
            prev_value = value
        else:
            if prev_value * value < 0:
                return False

    return True


def angle(x, y):
    """
    Inputs:
    x: 1d numpy array

    y: 1d numpy array

    Output:
    angle (in radian) between the point (xi, yi) and pi
    """
    return np.pi - np.arctan(y * 1.0 / x) + (x < 0).astype(int) * ((y < 0).astype(int) * 2 - 1) * np.pi


def generate_patch_depth_map(entire_depth_map, patch_index_map, patch_index, map_type, resolution, vertices, wall_min_height=None, wall_max_height=None):
    """
    Inputs:
    "entire_depth_map" should be a 2d numpy array. It will be modified in place.

    "map_type" should be either "ground" or "wall".

    "resolution" should be a float which represents the distance (in meter) between adjacent pixels on the depth map.

    "vertices" should be the coordinates of vertices of a quadrilateral patch.
    """
    vertices_array = np.array(vertices)
    # can also use the normal attribute of 'structure'
    normal = np.cross(vertices[0] - vertices[1], vertices[0] - vertices[2])

    if map_type == "ground":
        x_min, x_max = np.min(vertices_array[:, 0]), np.max(vertices_array[:, 0])
        y_min, y_max = np.min(vertices_array[:, 1]), np.max(vertices_array[:, 1])
        if x_min > SIDE / 2.0 or x_max < -SIDE / 2.0 or y_min > SIDE / 2.0 or y_max < -SIDE / 2.0:
            return

        for x_temp in range(int(math.ceil(x_min / resolution)), int(math.ceil(x_max / resolution))):
            x = x_temp * resolution
            if x < -SIDE / 2.0:
                continue
            if x > SIDE / 2.0:
                break
            for y_temp in range(int(math.ceil(y_min / resolution)), int(math.ceil(y_max / resolution))):
                y = y_temp * resolution
                if y < -SIDE / 2.0:
                    continue
                if y > SIDE / 2.0:
                    break
                z = vertices[0][2] - ((x - vertices[0][0]) * normal[0] + (y - vertices[0][1]) * normal[1]) / normal[2]
                z = round(z, 2)
                if point_inside_polygon(np.array([x, y, z]), vertices_array, normal):
                    idx_1 = int(SIDE / 2.0 / resolution) - y_temp
                    idx_2 = int(SIDE / 2.0 / resolution) - x_temp
                    if entire_depth_map[idx_1][idx_2] < z:
                        entire_depth_map[idx_1][idx_2] = z
                        patch_index_map[idx_1][idx_2] = patch_index

        # IPython.embed()

    elif map_type == "wall":
        z_min, z_max = np.min(vertices_array[:, 2]), np.max(vertices_array[:, 2])
        if z_min > wall_max_height or z_max < wall_min_height:
            return

        if np.min(vertices_array[:, 0] < 0) and np.min(vertices_array[:, 1]) < 0 and np.max(vertices_array[:, 1]) > 0:
            theta_min1 = 0
            p = vertices_array[np.argwhere(vertices_array[:, 1] > 0).reshape(-1,)]
            theta_max1 = np.max(angle(p[:, 0], p[:, 1]))
            n = vertices_array[np.argwhere(vertices_array[:, 1] < 0).reshape(-1,)]
            theta_min2 = np.min(angle(n[:, 0], n[:, 1]))
            theta_max2 = 2 * np.pi
            for z_temp in range(int(math.ceil(z_min / resolution)), int(math.ceil(z_max / resolution))):
                z = z_temp * resolution
                if z < wall_min_height:
                    continue
                if z > wall_max_height:
                    break
                for theta_temp in range(int(math.ceil(theta_min1 * RADIUS / resolution)), int(math.ceil(theta_max1 * RADIUS / resolution))) + range(int(math.ceil(theta_min2 * RADIUS / resolution)), int(math.ceil(theta_max2 * RADIUS / resolution))):
                    theta = np.pi - (theta_temp * resolution / RADIUS)
                    rho = (vertices[0][0] * normal[0] + vertices[0][1] * normal[1] + vertices[0][2] * normal[2] - z * normal[2]) / (np.cos(theta) * normal[0] + np.sin(theta) * normal[1])
                    if point_inside_polygon(np.array([np.cos(theta) * rho, np.sin(theta) * rho, z]), vertices_array, normal):
                        idx_1 = int(wall_max_height / resolution) - z_temp
                        idx_2 = theta_temp
                        if entire_depth_map[idx_1][idx_2] > rho:
                            entire_depth_map[idx_1][idx_2] = rho
                            patch_index_map[idx_1][idx_2] = patch_index

        else:
            theta_min, theta_max = np.min(angle(vertices_array[:, 0], vertices_array[:, 1])), np.max(angle(vertices_array[:, 0], vertices_array[:, 1]))
            for z_temp in range(int(math.ceil(z_min / resolution)), int(math.ceil(z_max / resolution))):
                z = z_temp * resolution
                if z < wall_min_height:
                    continue
                if z > wall_max_height:
                    break
                for theta_temp in range(int(math.ceil(theta_min * RADIUS / resolution)), int(math.ceil(theta_max * RADIUS / resolution))):
                    theta = np.pi - (theta_temp * resolution / RADIUS)
                    rho = (vertices[0][0] * normal[0] + vertices[0][1] * normal[1] + vertices[0][2] * normal[2] - z * normal[2]) / (np.cos(theta) * normal[0] + np.sin(theta) * normal[1])
                    if point_inside_polygon(np.array([np.cos(theta) * rho, np.sin(theta) * rho, z]), vertices_array, normal):
                        idx_1 = int(wall_max_height / resolution) - z_temp
                        idx_2 = theta_temp
                        if entire_depth_map[idx_1][idx_2] > rho:
                            entire_depth_map[idx_1][idx_2] = rho
                            patch_index_map[idx_1][idx_2] = patch_index

    else:
        print('wrong type')
        exit(1)



def generate_depth_map_and_boundary_map(coordinates, map_type, resolution, wall_min_height=None, wall_max_height=None):
    """
    Inputs:
    "coordinates" should be a 2d numpy array.
        each row represents the xyz coordinates of a vertex.
        every four rows represents the vertices of a quadrilateral.

    "map_type" should be either "ground" or "wall".

    "resolution" should be a float which represents the distance (in meter) between adjacent pixels on the depth map.

    Outputs:
    a 2d depth map and a 2d boundary map (1: on the boundary, 0: not on the boundary)
    """
    if map_type == "ground":
        patch_index_map = np.ones((int(SIDE / resolution) + 1, int(SIDE / resolution) + 1), dtype=int) * -1
        entire_depth_map = np.ones((int(SIDE / resolution) + 1, int(SIDE / resolution) + 1), dtype=float) * GROUND_DEFAULT_DEPTH
        for i in range(coordinates.shape[0] // 4):
            generate_patch_depth_map(entire_depth_map, patch_index_map, i, "ground", resolution, [coordinates[4*i], coordinates[4*i+1], coordinates[4*i+2], coordinates[4*i+3]])
        entire_boundary_map = np.zeros((int(SIDE / resolution) + 1, int(SIDE / resolution) + 1), dtype=int)
        # IPython.embed()

    elif map_type == "wall":
        patch_index_map = np.ones((int(round((wall_max_height - wall_min_height) / resolution, 0)) + 1, int(2 * np.pi * RADIUS / resolution) + 1), dtype=int) * -1
        entire_depth_map = np.ones((int(round((wall_max_height - wall_min_height) / resolution, 0)) + 1, int(2 * np.pi * RADIUS / resolution) + 1), dtype=float) * WALL_DEFAULT_DEPTH
        for i in range(coordinates.shape[0] // 4):
            generate_patch_depth_map(entire_depth_map, patch_index_map, i, "wall", resolution, [coordinates[4*i], coordinates[4*i+1], coordinates[4*i+2], coordinates[4*i+3]], wall_min_height, wall_max_height)
        entire_boundary_map = np.zeros((int(round((wall_max_height - wall_min_height) / resolution, 0)) + 1, int(2 * np.pi * RADIUS / resolution) + 1), dtype=int)
        # IPython.embed()

    else:
        print('wrong type')
        exit(1)

    maximal_y = patch_index_map.shape[0] - 1
    for idx in range(patch_index_map.shape[1]):
        entire_boundary_map[0][idx] = 1
        entire_boundary_map[maximal_y][idx] = 1
    maximal_x = patch_index_map.shape[1] - 1
    for idy in range(1, patch_index_map.shape[0] - 1):
        entire_boundary_map[idy][0] = 1
        entire_boundary_map[idy][maximal_x] = 1


    level_sets = {0: set()}

    for idx in range(1, patch_index_map.shape[1] - 1):
        for idy in range(1, patch_index_map.shape[0] - 1):
            if patch_index_map[idy][idx] != patch_index_map[idy-1][idx] or patch_index_map[idy][idx] != patch_index_map[idy+1][idx] or patch_index_map[idy][idx] != patch_index_map[idy][idx-1] or patch_index_map[idy][idx] != patch_index_map[idy][idx+1]:
                entire_boundary_map[idy][idx] = 1
                level_sets[0].add((idy, idx))

    for level in range(BOUNDARY_LEVEL_THRESHOLD):
        level_sets[level+1] = set()
        for (idy, idx) in level_sets[level]:
            if entire_boundary_map[idy-1][idx] == 0:
                entire_boundary_map[idy-1][idx] = 1
                level_sets[level+1].add((idy-1, idx))
            if entire_boundary_map[idy+1][idx] == 0:
                entire_boundary_map[idy+1][idx] = 1
                level_sets[level+1].add((idy+1, idx))
            if entire_boundary_map[idy][idx-1] == 0:
                entire_boundary_map[idy][idx-1] = 1
                level_sets[level+1].add((idy, idx-1))
            if entire_boundary_map[idy][idx+1] == 0:
                entire_boundary_map[idy][idx+1] = 1
                level_sets[level+1].add((idy, idx+1))
            


    return entire_depth_map, entire_boundary_map


def generate_combined_map(coordinates, map_type, resolution, wall_min_height=None, wall_max_height=None):
    depth_map, boundary_map = generate_depth_map_and_boundary_map(coordinates, map_type, resolution, wall_min_height, wall_max_height)
    if map_type == "ground":
        combined_map = np.clip(depth_map + boundary_map * -2.0, -1, 1)

    elif map_type == "wall":
        combined_map = np.clip(depth_map + boundary_map * 2.0, 0, 2)

    else:
        print('wrong type')
        exit(1)

    return combined_map


def main():
    environment_type = 9
    environment_index = 181
    with open('/mnt/big_narstie_data/chenxi/data/medium_dataset_normal_wall/environments_' + str(environment_type) + '_' + str(environment_index), 'r') as env_file:
        environment = pickle.load(env_file)
    ground_vertices = environment['ground_vertices']
    others_vertices = environment['others_vertices']
    p1 = (0, 0, -4)
    depth_map_id = str(environment_type) + '_' + str(environment_index) + '_' + str(p1[0]) + str(p1[1]) + str(p1[2])
    ground_patch_coordinates = rotate_quadrilaterals(ground_vertices, p1[2] * ANGLE_RESOLUTION)
    ground_depth_map = generate_combined_map(ground_patch_coordinates, 'ground', RESOLUTION)
    with open('../data/test/ground_depth_maps/' + depth_map_id, 'w') as depth_map_file:  
        pickle.dump(np.expand_dims(ground_depth_map, axis=0).astype(np.float32), depth_map_file)
    wall_patch_coordinates = rotate_quadrilaterals(others_vertices, p1[2] * ANGLE_RESOLUTION)
    wall_depth_map = generate_combined_map(wall_patch_coordinates, 'wall', RESOLUTION, wall_min_height=1.1, wall_max_height=1.7)
    with open('../data/test/wall_depth_maps/' + depth_map_id, 'w') as depth_map_file:
        pickle.dump(np.expand_dims(wall_depth_map, axis=0).astype(np.float32), depth_map_file)



if __name__ == '__main__':
    main()


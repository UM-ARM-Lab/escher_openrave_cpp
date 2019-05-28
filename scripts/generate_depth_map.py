import pickle, IPython, os, math, shutil
import numpy as np
import matplotlib.pyplot as plt

RESOLUTION = 0.025
SIDE = 1.5
GROUND_DEFAULT_DEPTH = -1.0
RADIUS = 1.0
WALL_DEFAULT_DEPTH = 2.0
WALL_MIN_HEIGHT = 1.0
WALL_MAX_HEIGHT = 2.0


def rotate_coordinate_system(coordinates, theta):
    """
    Inputs:
    "coordinates" should be a list of a list of tuples.
        each tuple represents the xyz coordinate of a vertex;
        each inner list represents the vertices of a quadrilateral;
        the outer list represents a list of quadrilaterals.

    "theta" should be the angle (in degree), by which the coordinate system will be rotated counter clockwise.

    Output:
    a 2d numpy array which represents the coordinates of the vertices in the rotated coordinate system.
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


def patch_depth_map(entire_map, map_type, resolution, vertices):
    """
    Inputs:
    "entire_map" should be a 2d numpy array. It will be modified in place.

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
                    if entire_map[idx_1][idx_2] < z:
                        entire_map[idx_1][idx_2] = z

        # IPython.embed()

    elif map_type == "wall":
        z_min, z_max = np.min(vertices_array[:, 2]), np.max(vertices_array[:, 2])
        if z_min > WALL_MAX_HEIGHT or z_max < WALL_MIN_HEIGHT:
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
                if z < WALL_MIN_HEIGHT:
                    continue
                if z > WALL_MAX_HEIGHT:
                    break
                for theta_temp in range(int(math.ceil(theta_min1 * RADIUS / resolution)), int(math.ceil(theta_max1 * RADIUS / resolution))) + range(int(math.ceil(theta_min2 * RADIUS / resolution)), int(math.ceil(theta_max2 * RADIUS / resolution))):
                    theta = np.pi - (theta_temp * resolution / RADIUS)
                    rho = (vertices[0][0] * normal[0] + vertices[0][1] * normal[1] + vertices[0][2] * normal[2] - z * normal[2]) / (np.cos(theta) * normal[0] + np.sin(theta) * normal[1])
                    assert(rho > 0)
                    if point_inside_polygon(np.array([np.cos(theta) * rho, np.sin(theta) * rho, z]), vertices_array, normal):
                        idx_1 = int(WALL_MAX_HEIGHT / resolution) - z_temp
                        idx_2 = theta_temp
                        if entire_map[idx_1][idx_2] > rho:
                            entire_map[idx_1][idx_2] = rho

        else:
            theta_min, theta_max = np.min(angle(vertices_array[:, 0], vertices_array[:, 1])), np.max(angle(vertices_array[:, 0], vertices_array[:, 1]))
            for z_temp in range(int(math.ceil(z_min / resolution)), int(math.ceil(z_max / resolution))):
                z = z_temp * resolution
                if z < WALL_MIN_HEIGHT:
                    continue
                if z > WALL_MAX_HEIGHT:
                    break
                for theta_temp in range(int(math.ceil(theta_min * RADIUS / resolution)), int(math.ceil(theta_max * RADIUS / resolution))):
                    theta = np.pi - (theta_temp * resolution / RADIUS)
                    rho = (vertices[0][0] * normal[0] + vertices[0][1] * normal[1] + vertices[0][2] * normal[2] - z * normal[2]) / (np.cos(theta) * normal[0] + np.sin(theta) * normal[1])
                    assert(rho > 0)
                    if point_inside_polygon(np.array([np.cos(theta) * rho, np.sin(theta) * rho, z]), vertices_array, normal):
                        idx_1 = int(WALL_MAX_HEIGHT / resolution) - z_temp
                        idx_2 = theta_temp
                        if entire_map[idx_1][idx_2] > rho:
                            entire_map[idx_1][idx_2] = rho

    else:
        print('wrong type')
        exit(1)


def entire_depth_map(coordinates, map_type, resolution=RESOLUTION):
    """
    Inputs:
    "coordinates" should be a 2d numpy array.
        each row represents the xyz coordinates of a vertex.
        every four rows represents the vertices of a quadrilateral.

    "map_type" should be either "ground" or "wall".

    "resolution" should be a float which represents the distance (in meter) between adjacent pixels on the depth map.

    Output:
    a 2d numpy array which represents a depth map.
    """
    if map_type == "ground":
        entire_map = np.ones((int(SIDE / resolution) + 1, int(SIDE / resolution) + 1), dtype=float) * GROUND_DEFAULT_DEPTH
        for i in range(coordinates.shape[0] // 4):
            patch_depth_map(entire_map, "ground", resolution, [coordinates[4*i], coordinates[4*i+1], coordinates[4*i+2], coordinates[4*i+3]])
        # IPython.embed()

    elif map_type == "wall":
        entire_map = np.ones((int((WALL_MAX_HEIGHT - WALL_MIN_HEIGHT)/ resolution) + 1, int(2 * np.pi * RADIUS / resolution) + 1), dtype=float) * WALL_DEFAULT_DEPTH
        for i in range(coordinates.shape[0] // 4):
            patch_depth_map(entire_map, "wall", resolution, [coordinates[4*i], coordinates[4*i+1], coordinates[4*i+2], coordinates[4*i+3]])
        # IPython.embed()

    else:
        print('wrong type')
        exit(1)

    return entire_map


def main():
    file = open('../data/environments', 'r')
    environments = pickle.load(file)

    file = open('../data/environ_pose_to_ddyn', 'r')
    environ_pose_to_ddyn = pickle.load(file)

    if os.path.exists('../data/minimal'):
        shutil.rmtree('../data/minimal')
    os.makedirs('../data/minimal')

    os.makedirs('../data/minimal/ground_depth_maps')
    os.makedirs('../data/minimal/wall_depth_maps')

    example_id = 0
    final_status_list = []
    minimal_ddyn_list = []

    # # IPython.embed()

    for environment_index in environ_pose_to_ddyn:
        pose_to_ddyn = environ_pose_to_ddyn[environment_index]
        for pose in pose_to_ddyn:
            # pose has six entries:
            # (init_x, init_y, init_theta, final_x, final_y, final_theta)
            assert(pose[0] == 0.0 and pose[1] == 0.0)
            ground_patch_coordinates = rotate_coordinate_system(environments[environment_index]['ground'], pose[2])
            ground_depth_map = entire_depth_map(ground_patch_coordinates, 'ground', RESOLUTION)
            file = open('../data/minimal/ground_depth_maps/' + str(example_id), 'w')
            pickle.dump(ground_depth_map, file)
            wall_patch_coordinates = rotate_coordinate_system(environments[environment_index]['others'], pose[2])
            wall_depth_map = entire_depth_map(wall_patch_coordinates, 'wall', RESOLUTION)
            file = open('../data/minimal/wall_depth_maps/' + str(example_id), 'w')
            pickle.dump(wall_depth_map, file)
            final_status_list.append(pose[3:6])
            minimal_ddyn_list.append(min(pose_to_ddyn[pose]))
            example_id += 1
    
    file = open('../data/minimal/final_status', 'w')
    pickle.dump(np.array(final_status_list), file)
    file = open('../data/minimal/minimal_ddyn', 'w')
    pickle.dump(np.array(minimal_ddyn_list), file)

    # for idx, environment in enumerate(environments):
    #     ground_depth_map = entire_depth_map(np.array(environment['ground']).reshape(-1, 3), 'ground', RESOLUTION)
    #     file = open('../data/minimal/ground_depth_maps/' + str(idx), 'w')
    #     pickle.dump(ground_depth_map, file)
    #     wall_depth_map = entire_depth_map(np.array(environment['others']).reshape(-1, 3), 'wall', RESOLUTION)
    #     file = open('../data/minimal/wall_depth_maps/' + str(idx), 'w')
    #     pickle.dump(wall_depth_map, file)
        
    
if __name__ == '__main__':
    main()


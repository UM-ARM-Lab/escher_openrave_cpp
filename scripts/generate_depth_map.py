import pickle, IPython, os, math
import numpy as np

RESOLUTION = 0.025
SIDE = 3.0
GROUND_MAX_DEPTH = 3.0
RADIUS = 1.5
WALL_MAX_DEPTH = 1.5
WALL_MAX_HEIGHT = 3.0


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


def point_inside_polygon(point, vertices):
    """
    vertices: [vertex1, vertex2, vertex3, vertex4]
    """
    prev_value = None
    num_vertices = len(vertices)
    for i in range(num_vertices):
        begin = vertices[i]
        end = vertices[(i + 1) % i]
        vector1 = end - begin
        vector2 = point - begin
        value = vector1[0] * vector2[1] - vector1[1] * vector2[0]
        if value == 0:
            return False

        if not prev_value:
            prev_value = value
        else:
            if prev_value * value < 0:
                return False

    return True


def patch_depth_map(entire_map, map_type, resolution, vertices):
    """
    Inputs:
    "entire_map" should be a 2d numpy array. It will be modified in place.

    "map_type" should be either "ground" or "wall".
    
    "resolution" should be a float which represents the distance (in meter) between adjacent pixels on the depth map.

    "vertices" should be the coordinates of vertices of a quadrilateral patch. 
    """
    vertices_array = np.array(vertices)
    normal = np.cross(vertices[0] - vertices[1], vertices[0] - vertices[2])
    if map_type == "ground":
        x_min, x_max = np.min(vertices_array[:, 0]), np.max(vertices_array[:, 0])
        y_min, y_max = np.min(vertices_array[:, 1]), np.max(vertices_array[:, 1])
        if x_min > SIDE / 2 || x_max < -SIDE / 2 || y_min > side / 2 || y_max < -SIDE / 2:
            return

        for x in range(math.floor(x_min / resolution) * resolution, math.ceil(x_max / resolution) * resolution, resolution):
            if x > SIDE / 2 || x < -SIDE / 2:
                return

            for y in range(math.floor(y_min / resolution) * resolution, math.ceil(y_max / resolution) * resolution, resolution):
                if y > SIDE / 2 || y < -SIDE / 2:
                    return

                z = vertices[0][2] - ((x - vertices[0][0]) * normal[0] + (y - vertices[0][1]) * normal[1]) / normal[2]
                z = round(z, 2)
                if point_inside_polygon(np.array(x, y, z), vertices_array):
                    idx_1 = int((SIDE / 2 - y) / resolution) 
                    idx_2 = int((SIDE / 2 - x) / resolution)
                    if entire_depth_map[idx_1][idx_2] > z:
                        entire_depth_map[idx_1][idx_2] = z

    elif map_type == "wall":


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
        entire_map = np.ones((int(SIDE / resolution), int(SIDE / resolution)), dtype=float) * GROUND_MAX_DEPTH
        for i in range(coordinates.shape[0] // 4):
            patch_depth_map(entire_map, "ground", resolution, coordinates[4*i], coordinates[4*i+1], coordinates[4*i+2], coordinates[4*i+3])

    elif map_type == "wall":
        entire_map = np.ones((int(WALL_MAX_HEIGHT / resolution), int(2 * np.pi * RADIUS / resolution)), dtype=float) * WALL_MAX_DEPTH
        for i in range(coordinates.shape[0] // 4):
            patch_depth_map(entire_map, "wall", resolution, coordinates[4*i], coordinates[4*i+1], coordinates[4*i+2], coordinates[4*i+3])

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

    for environment_index in environ_pose_to_ddyn:
        pose_to_ddyn = environ_pose_to_ddyn[environment_index]
        for pose in pose_to_ddyn:
            # pose has six entries:
            # (init_x, init_y, init_theta, final_x, final_y, final_theta)
            assert(pose[0] == 0.0 and pose[1] == 0.0)
            ground_patch_coordinates = rotate_coordinate_system(environments[environment_index]['ground'], pose[2])
            ground_depth_map = entire_depth_map(ground_patch_coordinates, 'ground', RESOLUTION)
            file = ('../data/minimal/ground_depth_maps/' + str(example_id), 'w')
            pickle.dump(ground_depth_map, file)
            wall_patch_coordinates = rotate_coordinate_system(environments[environment_index]['others'], pose[2])
            wall_depth_map = entire_depth_map(wall_patch_coordinates, 'wall', RESOLUTION)
            file = ('../data/minimal/wall_depth_maps/' + str(example_id), 'w')
            pickle.dump(wall_depth_map, file)
            final_status_list.append(pose[3:6])
            minimal_ddyn_list.append(min(pose_to_ddyn[pose]))
            example_id += 1

    file = open('../data/minimal/final_status', 'w')
    pickle.dump(np.array(final_status_list), file)
    file = open('../data/minimal/minimal_ddyn', 'w')
    pickle.dump(np.array(minimal_ddyn_list), file)



if __name__ == '__main__':
    main()
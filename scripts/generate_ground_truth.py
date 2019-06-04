import pickle, IPython, os, math, shutil
import numpy as np
import matplotlib.pyplot as plt

from generate_depth_map import rotate_quadrilaterals, entire_depth_map

GRID_RESOLUTION = 0.05
ANGLE_RESOLUTION = 15.0

RESOLUTION = 0.025
SIDE = 1.5
GROUND_DEFAULT_DEPTH = -1.0
RADIUS = 1.0
WALL_DEFAULT_DEPTH = 2.0
WALL_MIN_HEIGHT = 1.0
WALL_MAX_HEIGHT = 2.0

# currently unused
NUM_ENVIRONMENT_TYPE = 24

def rotate_one_point(x, y, theta):
    """
    Inputs:
    "x" should be the x coordinate of a point.

    "y" should be the y coordinate of a point.

    "theta" should be the angle (in degree), by which the point will be rotated clockwise.

    Outputs:
    new coordinate after rotation
    """
    theta_in_radian = theta * np.pi / 180
    # rotating the coordinate system counter clockwise is the same as rotating the points clockwise
    rotation_matrix = np.array([[np.cos(-theta_in_radian), -np.sin(-theta_in_radian)],
                                [np.sin(-theta_in_radian), np.cos(-theta_in_radian)]])
    vector = np.array([x, y])
    new_vector = np.matmul(rotation_matrix, vector)
    return new_vector[0], new_vector[1]


def main():
    # if os.path.exists('../data/minimal'):
    #     shutil.rmtree('../data/minimal')
    # os.makedirs('../data/minimal')

    # os.makedirs('../data/minimal/ground_depth_maps')
    # os.makedirs('../data/minimal/wall_depth_maps')

    if os.path.exists('../data/test'):
        shutil.rmtree('../data/test')
    os.makedirs('../data/test')

    example_id = 0
    # final_status_list = []
    # minimal_ddyn_list = []

    for i in [0]:
        # file = open('../data/ground_truth/environments_' + str(i), 'r')
        # environments = pickle.load(file)

        file = open('../data/environ_pose_to_ddyn_' + str(i), 'r')
        environ_pose_to_ddyn = pickle.load(file)

        for environment_index in environ_pose_to_ddyn:
            pose_to_ddyn = environ_pose_to_ddyn[environment_index]
            for pose in pose_to_ddyn:
                # pose has six entries (in cell):
                # (init_x, init_y, init_theta, final_x, final_y, final_theta)
                assert(pose[0] == 0 and pose[1] == 0)
                # ground_patch_coordinates = rotate_quadrilaterals(environments[environment_index]['ground'], pose[2] * ANGLE_RESOLUTION)
                # ground_depth_map = entire_depth_map(ground_patch_coordinates, 'ground', RESOLUTION)
                # file = open('../data/minimal/ground_depth_maps/' + str(example_id), 'w')
                # pickle.dump(ground_depth_map, file)
                # wall_patch_coordinates = rotate_quadrilaterals(environments[environment_index]['others'], pose[2] * ANGLE_RESOLUTION)
                # wall_depth_map = entire_depth_map(wall_patch_coordinates, 'wall', RESOLUTION)
                # file = open('../data/minimal/wall_depth_maps/' + str(example_id), 'w')
                # pickle.dump(wall_depth_map, file)
                # rotated_final_x, rotated_final_y = rotate_one_point(pose[3]*GRID_RESOLUTION, pose[4]*GRID_RESOLUTION, pose[2]*ANGLE_RESOLUTION)
                # final_status_list.append([rotated_final_x, rotated_final_y, (pose[5] - pose[2])*ANGLE_RESOLUTION])
                # minimal_ddyn_list.append(min(pose_to_ddyn[pose]))

                clipped = np.clip(pose_to_ddyn[pose], 0, 8000)
                # IPython.embed()
                plt.figure(example_id)
                plt.hist(clipped, bins=range(-50, 8001, 50))
                plt.title('e: {} p1: {} p2: {}'.format(environment_index, pose[0:3], pose[3:]))
                plt.savefig('../data/test/{}.png'.format(example_id))

                example_id += 1
    
    # file = open('../data/minimal/final_status', 'w')
    # pickle.dump(np.array(final_status_list), file)
    # file = open('../data/minimal/minimal_ddyn', 'w')
    # pickle.dump(np.array(minimal_ddyn_list), file)

    
if __name__ == '__main__':
    main()


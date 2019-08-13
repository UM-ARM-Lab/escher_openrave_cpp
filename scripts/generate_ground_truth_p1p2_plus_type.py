"""
Generate data for model: (p1, p2) -> 25 percentile ddyn
Data will be saved in a dict.
example_id (string), p2, environment wall type (0: no wall, 1: one wall only, 2: two walls), 25 percentile ddyn (float)
"""

import pickle, IPython, os, math, shutil, getopt, sys, random
import numpy as np

# from generate_depth_map import rotate_quadrilaterals, entire_depth_map
from generate_boundary_and_depth_map import generate_combined_map, rotate_quadrilaterals

GRID_RESOLUTION = 0.15
ANGLE_RESOLUTION = 15.0
PERCENTILE = 25
# NUM_ENVIRONMENT_TYPE = 12
NUM_ENVIRONMENT_PER_TYPE = 100
DEPTH_MAP_RESOLUTION = 0.025
MAX_TRANSITIONS = 200
WALL_MIN_HEIGHT = 1.1
WALL_MAX_HEIGHT = 1.7


def calculate_altitude(arr):
    mask = arr > -0.99
    assert(np.sum(mask) > 0)
    return np.sum(arr * mask) / np.sum(mask)


def main():
    environment_type = None
    try:
        inputs, _ = getopt.getopt(sys.argv[1:], "e:", ['environment_type'])

        for opt, arg in inputs:
            if opt == '-e':
                environment_type = int(arg)

    except getopt.GetoptError:
        print('usage: -e: [environment_type]')
        exit(1)

    training = []
    validation = []
    test = []
    p2_ddyn_dict = {}

    total_transition = 0
    large_transition = 0

    p1_list = [(0,0,-5), (0,0,-4), (0,0,-3), (0,0,-2), (0,0,-1), (0,0,0), (0,0,1), (0,0,2), (0,0,3), (0,0,4), (0,0,5), (0,0,6)]

    for environment_index in range(0, 200):
        print('process data for environment type {} index {}'.format(environment_type, environment_index))
        with open('/mnt/big_narstie_data/chenxi/data/medium_dataset_normal_wall/environments_' + str(environment_type) + '_' + str(environment_index), 'r') as env_file:
            environment = pickle.load(env_file)
            ground_vertices = environment['ground_vertices']
            others_vertices = environment['others_vertices']
        # with open('/mnt/big_narstie_data/chenxi/data/medium_dataset_normal_wall/dynamic_cost_plus_type_' + str(environment_type) + '_' + str(environment_index), 'r') as file:
        #     data = pickle.load(file)

        # p1_list = sorted(data.keys(), key=lambda element: (element[0], element[1], element[2]))
        for p1 in p1_list:
            depth_map_id = str(environment_type) + '_' + str(environment_index) + '_' + str(p1[0]) + str(p1[1]) + str(p1[2])
            ground_patch_coordinates = rotate_quadrilaterals(ground_vertices, p1[2] * ANGLE_RESOLUTION)
            # ground_depth_map = entire_depth_map(ground_patch_coordinates, 'ground', DEPTH_MAP_RESOLUTION)
            ground_depth_map = generate_combined_map(ground_patch_coordinates, 'ground', DEPTH_MAP_RESOLUTION)
            with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/ground_depth_and_boundary_maps/' + depth_map_id, 'w') as depth_map_file:
                pickle.dump(np.expand_dims(ground_depth_map, axis=0).astype(np.float32), depth_map_file)

            altitude = calculate_altitude(np.array([ground_depth_map[27,27], ground_depth_map[27,33], ground_depth_map[27,39],
                                                    ground_depth_map[33,27], ground_depth_map[33,33], ground_depth_map[33,39],
                                                    ground_depth_map[39,27], ground_depth_map[39,33], ground_depth_map[39,39]]))
            altitude = np.round(altitude, 1)
            wall_patch_coordinates = rotate_quadrilaterals(others_vertices, p1[2] * ANGLE_RESOLUTION)
            # wall_depth_map = entire_depth_map(wall_patch_coordinates, 'wall', DEPTH_MAP_RESOLUTION, wall_min_height=altitude+WALL_MIN_HEIGHT, wall_max_height=altitude+WALL_MAX_HEIGHT)
            wall_depth_map = generate_combined_map(wall_patch_coordinates, 'wall', DEPTH_MAP_RESOLUTION, wall_min_height=altitude+WALL_MIN_HEIGHT, wall_max_height=altitude+WALL_MAX_HEIGHT)
            with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/wall_depth_and_boundary_maps/' + depth_map_id, 'w') as depth_map_file:
                pickle.dump(np.expand_dims(wall_depth_map, axis=0).astype(np.float32), depth_map_file)

    #         p2_list = sorted(data[p1].keys(), key=lambda element: (element[0], element[1], element[2]))
    #         for p2 in p2_list:

    #             if (6 in data[p1][p2] or 7 in data[p1][p2] or 8 in data[p1][p2] or 9 in data[p1][p2]):
    #                 environment_wall_type = 2
    #             elif (1 in data[p1][p2] or 2 in data[p1][p2] or 3 in data[p1][p2] or 4 in data[p1][p2] or 5 in data[p1][p2]):
    #                 environment_wall_type = 1
    #             else:
    #                 environment_wall_type = 0

    #             ddyn_list = []
    #             for transition_type in data[p1][p2]:
    #                 for transition in data[p1][p2][transition_type]:
    #                     ddyns = data[p1][p2][transition_type][transition][:, 6].tolist()
    #                     random.shuffle(ddyns)
    #                     total_transition += 1
    #                     if len(ddyns) > MAX_TRANSITIONS:
    #                         large_transition += 1
    #                     ddyn_list += ddyns[:min(len(ddyns), MAX_TRANSITIONS)]

    #             if ddyn_list:
    #                 example_id = depth_map_id + '_' + str(p2[0]) + str(p2[1]) + str(p2[2])
    #                 if environment_index < 80:
    #                     training.append(example_id)
    #                 elif environment_index < 90:
    #                     validation.append(example_id)
    #                 else:
    #                     test.append(example_id)

    #                 p2_ddyn_dict[example_id] = (depth_map_id, np.array(p2).astype(np.float32), environment_wall_type, np.percentile(np.array(ddyn_list), PERCENTILE).astype(np.float32))
    #     print(total_transition, large_transition)

    # with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/p2_ddyn_2' + str(environment_type), 'w') as file:
    #     pickle.dump(p2_ddyn_dict, file)

    # partition = {'training': training,
    #              'validation': validation,
    #              'test': test}
    # with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/partition_2' + str(environment_type), 'w') as file:
    #     pickle.dump(partition, file)
     

if __name__ == '__main__':
    main()


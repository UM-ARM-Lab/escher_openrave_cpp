"""
Generate data for model: (p1, p2) -> 25 percentile ddyn
Data will be saved in a dict.
example_id (string), 25 percentile ddyn (float)
"""

import pickle, IPython, os, math, shutil
import numpy as np

from generate_depth_map import rotate_quadrilaterals, entire_depth_map

GRID_RESOLUTION = 0.15
ANGLE_RESOLUTION = 15.0
PERCENTILE = 25
NUM_ENVIRONMENT_TYPE = 12
NUM_ENVIRONMENT_PER_TYPE = 50
DEPTH_MAP_RESOLUTION = 0.025


def main():
    p2_ddyn_dict = {}
    # for environment_type in range(NUM_ENVIRONMENT_TYPE):
    for environment_type in [9]:
        with open('../data/medium_dataset_normal_wall/environments_' + str(environment_type), 'r') as env_file:
            environments = pickle.load(env_file)
            # for environment_index in range(NUM_ENVIRONMENT_PER_TYPE):
            for environment_index in [0]:
                if os.path.exists('../data/medium_dataset_normal_wall/dynamic_cost_' + str(environment_type) + '_' + str(environment_index)):
                    print('process data in file dynamic_cost_{}_{}'.format(environment_type, environment_index))
                    ground_vertices = environments[environment_index]['ground_vertices']
                    others_vertices = environments[environment_index]['others_vertices']
                    with open('../data/medium_dataset_normal_wall/dynamic_cost_' + str(environment_type) + '_' + str(environment_index), 'r') as file:
                        data = pickle.load(file)
                        p1_list = sorted(data.keys(), key=lambda element: (element[0], element[1], element[2]))
                        for p1 in p1_list:
                            depth_map_id = str(environment_type) + '_' + str(environment_index) + '_' + str(p1[0]) + str(p1[1]) + str(p1[2])
                            ground_patch_coordinates = rotate_quadrilaterals(ground_vertices, p1[2] * ANGLE_RESOLUTION)
                            ground_depth_map = entire_depth_map(ground_patch_coordinates, 'ground', DEPTH_MAP_RESOLUTION)
                            with open('../data/ground_truth_p1p2/ground_depth_maps/' + depth_map_id, 'w') as depth_map_file:  
                                pickle.dump(ground_depth_map, depth_map_file)
                            wall_patch_coordinates = rotate_quadrilaterals(others_vertices, p1[2] * ANGLE_RESOLUTION)
                            wall_depth_map = entire_depth_map(wall_patch_coordinates, 'wall', DEPTH_MAP_RESOLUTION)
                            with open('../data/ground_truth_p1p2/wall_depth_maps/' + depth_map_id, 'w') as depth_map_file:
                                pickle.dump(wall_depth_map, depth_map_file)

                            p2_list = sorted(data[p1].keys(), key=lambda element: (element[0], element[1], element[2]))
                            for p2 in p2_list:
                                example_id = depth_map_id + '_' + str(p2[0]) + str(p2[1]) + str(p2[2])
                                print('\'{}\','.format(example_id))
                                p2_ddyn_dict[example_id] = (depth_map_id, np.array(p2), np.percentile(data[p1][p2], PERCENTILE).astype(np.float32))
    with open('../data/ground_truth_p1p2/p2_ddyn', 'w') as file:
        pickle.dump(p2_ddyn_dict, file)        

if __name__ == '__main__':
    main()


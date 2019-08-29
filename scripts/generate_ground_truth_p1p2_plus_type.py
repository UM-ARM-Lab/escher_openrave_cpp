"""
Generate data for 4 models
model_0
p1_theta: 0
at most 52 p2

model_1
p1_theta: 1
at most 49 p2

model_2
p1_theta: 2
at most 50 p2

model_3
p1_theta: 3
at most 49 p2

for each model:
ground map around p1, wall map around p1, p2 -> 25 percentile

Data will be saved in a dict.
key is example id
map_id (string), p2, environment wall type (0: no wall, 1: one wall only, 2: two walls), 25 percentile ddyn (float)
"""

import pickle, IPython, os, math, shutil, getopt, sys, random
import numpy as np

# from generate_depth_map import rotate_quadrilaterals, entire_depth_map
# from generate_boundary_and_depth_map import generate_combined_map, rotate_quadrilaterals

GRID_RESOLUTION = 0.15
ANGLE_RESOLUTION = 22.5
ANGLE_DIM = int(round(360 / ANGLE_RESOLUTION))
NUM_MODELS = int(round(90 / ANGLE_RESOLUTION))
PERCENTILE = 25
NUM_ENVIRONMENT_PER_TYPE = 200
# MAP_RESOLUTION = 0.025
MAX_TRANSITIONS_CHOSEN = 200
# WALL_MIN_HEIGHT = 1.1
# WALL_MAX_HEIGHT = 1.7


# def calculate_altitude(arr):
#     mask = arr > -0.99
#     assert(np.sum(mask) > 0)
#     return np.sum(arr * mask) / np.sum(mask)


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

    training = {}
    validation = {}
    test = {}
    data_dict = {}
    for i in range(NUM_MODELS): # 4 models
        training[i] = []
        validation[i] = []
        test[i] = []
        data_dict[i] = {}

    # total_transition = 0
    # large_transition = 0
    transition_model = {}
    transition_model[0] = {(0,0,0),(0,1,1),(2,-1,0),(3,0,-1),(-2,1,-1),(1,-1,-1),(-2,-1,-1),(1,0,1),(2,1,-1),(1,-1,0),
                           (-1,1,-1),(2,-1,-1),(-2,1,1),(-2,-1,0),(1,1,-1),(0,0,-1),(3,-1,-1),(0,-1,-1),(1,0,0),(-1,0,1),
                           (2,0,1),(-1,0,0),(-2,0,-1),(2,0,0),(-2,1,0),(-1,0,-1),(3,1,1),(2,0,-1),(3,1,0),(0,0,1),
                           (0,-1,1),(0,-1,0),(-2,0,0),(2,-1,1),(1,1,1),(-1,1,0),(1,1,0),(2,1,0),(-2,0,1),(3,-1,0),
                           (-1,1,1),(-1,-1,1),(2,1,1),(1,0,-1),(-1,-1,0),(-2,-1,1),(3,0,0),(1,-1,1),(3,0,1),(0,1,0),
                           (0,1,-1),(-1,-1,-1)}
    transition_model[1] = {(0,0,0),(0,1,1),(3,0,-1),(1,-1,-1),(-1,-1,-1),(1,0,1),(2,1,-1),(1,-1,0),(-1,1,-1),(-1,-1,0),
                           (-2,-1,0),(1,1,-1),(0,0,-1),(-2,0,0),(-2,0,1),(0,-1,-1),(1,0,0),(-1,0,1),(2,0,1),(3,1,-1),
                           (1,2,0),(2,0,0),(-2,-1,1),(1,2,1),(-1,0,-1),(3,1,1),(2,0,-1),(3,1,0),(0,0,1),(0,-1,1),
                           (0,-1,0),(-2,0,-1),(-1,0,0),(1,1,1),(-1,1,0),(1,1,0),(2,1,0),(-1,1,1),(2,2,1),(2,1,1),
                           (1,0,-1),(2,2,0),(-1,-1,1),(3,0,0),(1,-1,1),(0,1,0),(0,1,-1),(-2,-1,-1),(2,2,-1)}
    transition_model[2] = {(0,0,0),(0,1,1),(0,-1,0),(1,0,0),(2,2,-1),(1,0,1),(2,1,-1),(1,-1,0),(-1,1,-1),(-1,-1,0),
                           (0,2,1),(-2,-1,0),(0,2,0),(-1,-2,0),(1,1,-1),(0,-1,-1),(0,0,-1),(-1,0,1),(-2,0,0),(1,2,0),
                           (1,2,-1),(2,0,0),(-1,-2,1),(1,2,1),(1,-1,-1),(-1,0,-1),(-1,-1,-1),(2,0,-1),(-2,0,-1),(0,0,1),
                           (0,-1,1),(-1,-1,1),(0,-2,1),(-1,0,0),(1,1,1),(-1,1,0),(1,1,0),(2,1,0),(-1,1,1),(2,2,1),
                           (2,1,1),(1,0,-1),(2,2,0),(-1,-2,-1),(-2,-1,1),(1,-1,1),(0,1,0),(0,1,-1),(-2,-1,-1),(0,-2,0)}
    transition_model[3] = {(0,0,0),(0,1,1),(1,0,0),(1,3,1),(1,0,1),(2,1,-1),(1,3,0),(1,-1,0),(-1,1,-1),(0,2,1),
                           (0,-1,-1),(0,2,0),(1,1,-1),(0,-2,-1),(0,0,-1),(-1,0,1),(-1,0,0),(1,2,0),(1,2,-1),(2,2,1),
                           (-1,-2,1),(1,2,1),(1,-1,-1),(-1,-1,-1),(-1,-1,0),(0,3,0),(-1,0,-1),(0,0,1),(0,-1,1),(0,3,1),
                           (-1,-1,1),(0,-1,0),(-1,-2,0),(1,1,1),(-1,1,0),(1,1,0),(0,2,-1),(2,1,0),(-1,1,1),(0,-2,1),
                           (1,0,-1),(2,2,0),(-1,-2,-1),(1,3,-1),(1,-1,1),(0,1,0),(0,1,-1),(2,2,-1),(0,-2,0)}

    for environment_index in range(0, 50):
        print('process data for environment type {} index {}'.format(environment_type, environment_index))
        # with open('/mnt/big_narstie_data/chenxi/data/dataset_225/environments_' + str(environment_type) + '_' + str(environment_index), 'r') as env_file:
        #     environment = pickle.load(env_file)
        #     ground_vertices = environment['ground_vertices']
        #     others_vertices = environment['others_vertices']
        with open('/mnt/big_narstie_data/chenxi/data/dataset_225/dynamic_cost_plus_type_' + str(environment_type) + '_' + str(environment_index), 'r') as file:
            data = pickle.load(file)

        p1_list = sorted(data.keys(), key=lambda element: (element[0], element[1], element[2]))
        for p1 in p1_list:
            model_index = (p1[2] + ANGLE_DIM) % NUM_MODELS

            map_id = str(environment_type) + '_' + str(environment_index) + '_' + str(p1[0]) + str(p1[1]) + str(p1[2])
            # ground_patch_coordinates = rotate_quadrilaterals(ground_vertices, p1[2] * ANGLE_RESOLUTION)
            # ground_depth_map = entire_depth_map(ground_patch_coordinates, 'ground', DEPTH_MAP_RESOLUTION)
            # ground_depth_map = generate_combined_map(ground_patch_coordinates, 'ground', DEPTH_MAP_RESOLUTION)
            # with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/ground_depth_and_boundary_maps/' + map_id, 'w') as depth_map_file:
            #     pickle.dump(np.expand_dims(ground_depth_map, axis=0).astype(np.float32), depth_map_file)

            # altitude = calculate_altitude(np.array([ground_depth_map[27,27], ground_depth_map[27,33], ground_depth_map[27,39],
            #                                         ground_depth_map[33,27], ground_depth_map[33,33], ground_depth_map[33,39],
            #                                         ground_depth_map[39,27], ground_depth_map[39,33], ground_depth_map[39,39]]))
            # if abs(altitude) > 0.05:
            #     print(map_id)
            # altitude = np.round(altitude, 1)
            
            # wall_patch_coordinates = rotate_quadrilaterals(others_vertices, p1[2] * ANGLE_RESOLUTION)
            # # wall_depth_map = entire_depth_map(wall_patch_coordinates, 'wall', DEPTH_MAP_RESOLUTION, wall_min_height=altitude+WALL_MIN_HEIGHT, wall_max_height=altitude+WALL_MAX_HEIGHT)
            # wall_depth_map = generate_combined_map(wall_patch_coordinates, 'wall', DEPTH_MAP_RESOLUTION, wall_min_height=altitude+WALL_MIN_HEIGHT, wall_max_height=altitude+WALL_MAX_HEIGHT)
            # with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/wall_depth_and_boundary_maps/' + map_id, 'w') as depth_map_file:
            #     pickle.dump(np.expand_dims(wall_depth_map, axis=0).astype(np.float32), depth_map_file)

            p2_list = sorted(data[p1].keys(), key=lambda element: (element[0], element[1], element[2]))
            for p2 in p2_list:
                if (6 in data[p1][p2] or 7 in data[p1][p2] or 8 in data[p1][p2] or 9 in data[p1][p2]):
                    environment_wall_type = 2
                elif (1 in data[p1][p2] or 2 in data[p1][p2] or 3 in data[p1][p2] or 4 in data[p1][p2] or 5 in data[p1][p2]):
                    environment_wall_type = 1
                else:
                    environment_wall_type = 0

                ddyn_list = []
                for transition_type in data[p1][p2]:
                    for transition in data[p1][p2][transition_type]:
                        ddyns = data[p1][p2][transition_type][transition].tolist()
                        random.shuffle(ddyns)
                        # total_transition += 1
                        # if len(ddyns) > MAX_TRANSITIONS_CHOSEN:
                        #     large_transition += 1
                        ddyn_list += ddyns[:min(len(ddyns), MAX_TRANSITIONS_CHOSEN)]

                if ddyn_list:
                    example_id = map_id + '_' + str(p2[0]) + str(p2[1]) + str(p2[2])
                    if environment_index < 160:
                        training[model_index].append(example_id)
                    elif environment_index < 180:
                        validation[model_index].append(example_id)
                    else:
                        test[model_index].append(example_id)

                diff = p1[2] - model_index
                if diff == 4:
                    converted_p2 = (p2[1], -p2[0], p2[2])
                elif diff == 0:
                    converted_p2 = p2
                elif diff == -4:
                    converted_p2 = (-p2[1], p2[0], p2[2])
                elif diff == -8:
                    converted_p2 = (-p2[0], -p2[1], p2[2])
                else:
                    print('error')
                    exit(1)

                assert(converted_p2 in transition_model[model_index])

                data_dict[model_index][example_id] = (map_id, np.array(converted_p2).astype(np.float32), environment_wall_type, np.percentile(np.array(ddyn_list), PERCENTILE).astype(np.float32))
        # print(total_transition, large_transition)

    for i in range(NUM_MODELS):
        with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/data_' + str(environment_type) + '_model_' + str(i), 'w') as file:
            pickle.dump(data_dict[i], file)

        partition = {'training': training[i],
                    'validation': validation[i],
                    'test': test[i]}
        with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/partition_' + str(environment_type) + '_model_' + str(i), 'w') as file:
            pickle.dump(partition, file)
     

if __name__ == '__main__':
    main()


import pickle, IPython, math, sys, getopt
import numpy as np
# import mkl
# mkl.get_max_threads()
# import faiss
# import timeit

GRID_RESOLUTION = 0.15
ANGLE_RESOLUTION = 15.0


def adjust_p2(p1, p2):
    """
    p1: [p1x, p1y, p1yaw]
    p2: [p2x, p2y, p2yaw]
    """
    p1_yaw_in_radian = p1[2] / 180.0 * np.pi
    rotation_matrix = np.array([[np.cos(-p1_yaw_in_radian), -np.sin(-p1_yaw_in_radian)],
                                [np.sin(-p1_yaw_in_radian), np.cos(-p1_yaw_in_radian)]])
    p2_xy_after_adjustment = np.matmul(rotation_matrix, np.array([[p2[0] - p1[0]], [p2[1] - p1[1]]]))
    return [p2_xy_after_adjustment[0][0], p2_xy_after_adjustment[1][0], p2[2] - p1[2]]


def discretize_torso_pose(p):
    """
    px, py, pyaw
    """
    resolutions = [GRID_RESOLUTION, GRID_RESOLUTION, ANGLE_RESOLUTION]
    indices = [None] * len(resolutions)
    for i, v in enumerate(p):
        if abs(v) > resolutions[i] / 2.0:
            temp = v - np.sign(v) * resolutions[i] / 2.0
            indices[i] = int(np.sign(temp) * math.ceil(np.round(abs(temp) / resolutions[i], 1)))
        else:
            indices[i] = 0
    return indices


def main():
    environment_type = None
    try:
        inputs, _ = getopt.getopt(sys.argv[1:], "e:", ['environment_type'])

        for opt, arg in inputs:
            if opt == '-e':
                environment_type = arg

    except getopt.GetoptError:
        print('usage: -e: [environment_type]')
        exit(1)

    for environment_index in range(100, 200):
        # load sampled transitions
        transitions = None
        with open('../data/medium_dataset_normal_wall/transitions_' + environment_type + '_' + str(environment_index), 'r') as file:
            transitions = pickle.load(file)

        # info is a nested dictionary.
        # its first key is p1 (tuple)
        # its second key is p2 (tuple)
        # its third key is transition type
        # its value is transitions
        info = {}
        for transition in transitions:
            temp_p1 = transition['p1']
            discretized_p1 = tuple(discretize_torso_pose([temp_p1[0], temp_p1[1], temp_p1[5]]))
            if discretized_p1 not in info:
                info[discretized_p1] = {}
                
            temp_p2 = transition['p2']
            adjusted_p2 = adjust_p2([temp_p1[0], temp_p1[1], temp_p1[5]], [temp_p2[0], temp_p2[1], temp_p2[5]])
            discretized_p2 = tuple(discretize_torso_pose(adjusted_p2))
            if discretized_p2 not in info[discretized_p1]:
                info[discretized_p1][discretized_p2] = {}

            transition_type = transition['contact_transition_type']
            if transition_type in info[discretized_p1][discretized_p2]:
                info[discretized_p1][discretized_p2][transition_type].append(transition)
            else:
                info[discretized_p1][discretized_p2][transition_type] = [transition]   
        print('start save data to file transitions_dict_{}_{}'.format(environment_type, environment_index))
        with open('../data/medium_dataset_normal_wall/transitions_dict_' + str(environment_type) + '_' + str(environment_index), 'w') as file:
            pickle.dump(info, file)
        print('finish save data to file transitions_dict_{}_{}'.format(environment_type, environment_index)) 

if __name__ == "__main__":
    main()


import pickle, IPython, math, sys, getopt
import numpy as np
# import mkl
# mkl.get_max_threads()
# import faiss
# import timeit

GRID_RESOLUTION = 0.15
ANGLE_RESOLUTION = 22.5


def discretize_torso_pose(p):
    """
    px, py, pyaw
    """
    if p[2] > 180 - ANGLE_RESOLUTION / 2.0 + 1:
        p[2] -= 360
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
    # load sampled transitions
    transitions = None
    with open('specific_transitions', 'r') as file:
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
        discretized_p2 = tuple(discretize_torso_pose([temp_p2[0], temp_p2[1], temp_p2[5]]))
        temp = discretized_p2[2] - discretized_p1[2]
        if temp >= 2:
            temp -= int(round(360 / ANGLE_RESOLUTION))
        elif temp <= -2:
            temp += int(round(360 / ANGLE_RESOLUTION))
        increment = (discretized_p2[0], discretized_p2[1], temp)

        if increment not in info[discretized_p1]:
            info[discretized_p1][increment] = {}

        transition_type = transition['contact_transition_type']
        if transition_type in info[discretized_p1][increment]:
            info[discretized_p1][increment][transition_type].append(transition)
        else:
            info[discretized_p1][increment][transition_type] = [transition]

    with open('transitions_dict_specific_transitions', 'w') as file:
        pickle.dump(info, file)
    print('save data to file transitions_dict_specific_transitions')

if __name__ == "__main__":
    main()


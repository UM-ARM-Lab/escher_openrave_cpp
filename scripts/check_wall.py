"""
Check the distribution of dynamic cost in different environments:
each environment type is defined below:
(an environment is identified by a p1 and p2 pair)
a. environment without any wall: all the transitions in that environment is of transition type 0
b. environment with one wall only: there is some transition in that environment of transition type 1-5, but no 6-9
c. environment with two walls: there is some transition in that environment of transition type 6-9 
"""

import pickle, IPython, os, math, shutil, getopt, sys
import numpy as np
import matplotlib.pyplot as plt

# from generate_depth_map import rotate_quadrilaterals, entire_depth_map

# GRID_RESOLUTION = 0.15
# ANGLE_RESOLUTION = 15.0
# PERCENTILE = 25
NUM_ENVIRONMENT_TYPE = 12
NUM_ENVIRONMENT_PER_TYPE = 50
# DEPTH_MAP_RESOLUTION = 0.025


def main():
    environment_a_hist = np.zeros((600,), dtype=np.float)
    environment_b_hist = np.zeros((600,), dtype=np.float)
    environment_c_hist = np.zeros((600,), dtype=np.float)

    for environment_type in range(NUM_ENVIRONMENT_TYPE):
        for environment_index in range(NUM_ENVIRONMENT_PER_TYPE):
            if os.path.exists('../data/medium_dataset_normal_wall/dynamic_cost_plus_type_' + str(environment_type) + '_' + str(environment_index)):
                print('process data in file dynamic_cost_plus_type_{}_{}'.format(environment_type, environment_index))
                with open('../data/medium_dataset_normal_wall/dynamic_cost_plus_type_' + str(environment_type) + '_' + str(environment_index), 'r') as file:
                    data = pickle.load(file)
                    p1_list = sorted(data.keys(), key=lambda element: (element[0], element[1], element[2]))
                    for p1i, p1 in enumerate(p1_list):
                        p2_list = sorted(data[p1].keys(), key=lambda element: (element[0], element[1], element[2]))
                        for p2i, p2 in enumerate(p2_list):
                            if (6 in data[p1][p2].keys() or 7 in data[p1][p2].keys() or 8 in data[p1][p2].keys() or 9 in data[p1][p2].keys()):
                                for transition_type in data[p1][p2].keys():
                                    clipped = np.clip(data[p1][p2][transition_type][:,6], 0, 6000)
                                    hist, _ = np.histogram(clipped, bins=np.arange(0, 6010, 10))
                                    environment_c_hist = environment_c_hist + hist
                            elif (1 in data[p1][p2].keys() or 2 in data[p1][p2].keys() or 3 in data[p1][p2].keys() or 4 in data[p1][p2].keys() or 5 in data[p1][p2].keys()):
                                for transition_type in data[p1][p2].keys():
                                    clipped = np.clip(data[p1][p2][transition_type][:,6], 0, 6000)
                                    hist, _ = np.histogram(clipped, bins=np.arange(0, 6010, 10))
                                    environment_b_hist = environment_b_hist + hist
                            else:
                                for transition_type in data[p1][p2].keys():
                                    clipped = np.clip(data[p1][p2][transition_type][:,6], 0, 6000)
                                    hist, _ = np.histogram(clipped, bins=np.arange(0, 6010, 10))
                                    environment_a_hist = environment_a_hist + hist
 
    plt.figure()
    plt.plot(range(600), environment_a_hist / np.sum(environment_a_hist), '-o', label='no wall', color='red')
    plt.plot(range(600), environment_b_hist / np.sum(environment_b_hist), '-o', label='one wall only', color='green')
    plt.plot(range(600), environment_c_hist / np.sum(environment_c_hist) , '-o', label='two walls', color='blue')
    plt.title('distribution of dynamic cost')
    plt.xlabel('dynamic cost')
    plt.ylabel('number of transitions')
    plt.legend()
    plt.savefig('../data/test_plus_type/wall.png')


if __name__ == '__main__':
    main()


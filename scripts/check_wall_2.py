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
    environment_a_list = []
    environment_b_list = []
    environment_c_list = []

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
                            ddyn_list = []
                            for transition_type in data[p1][p2]:
                                ddyn_list += data[p1][p2][transition_type][:,6].tolist()
                            percentile = np.percentile(np.array(ddyn_list), 25)
                            if (6 in data[p1][p2].keys() or 7 in data[p1][p2].keys() or 8 in data[p1][p2].keys() or 9 in data[p1][p2].keys()):
                                environment_c_list.append(percentile)
                            elif (1 in data[p1][p2].keys() or 2 in data[p1][p2].keys() or 3 in data[p1][p2].keys() or 4 in data[p1][p2].keys() or 5 in data[p1][p2].keys()):
                                environment_b_list.append(percentile)
                            else:
                                environment_a_list.append(percentile)

    clipped_a = np.clip(np.array(environment_a_list), 0, 2000)
    hist_a, _ = np.histogram(clipped_a, bins=np.arange(0, 2010, 10))
    clipped_b = np.clip(np.array(environment_b_list), 0, 2000)
    hist_b, _ = np.histogram(clipped_b, bins=np.arange(0, 2010, 10))
    clipped_c = np.clip(np.array(environment_c_list), 0, 2000)
    hist_c, _ = np.histogram(clipped_c, bins=np.arange(0, 2010, 10))
 
    plt.figure()
    plt.plot(range(200), hist_a, '-o', label='no wall', color='red', alpha=0.5)
    plt.plot(range(200), hist_b, '-o', label='one wall only', color='green', alpha=0.5)
    plt.plot(range(200), hist_c, '-o', label='two walls', color='blue', alpha=0.5)
    plt.title('distribution of 25 percentiles')
    plt.xlabel('25 percentile')
    plt.ylabel('number of examples')
    plt.legend()
    plt.savefig('../data/percentile_accurate_subset.png')


if __name__ == '__main__':
    main()


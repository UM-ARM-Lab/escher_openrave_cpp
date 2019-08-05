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

# GRID_RESOLUTION = 0.15
# ANGLE_RESOLUTION = 15.0
# PERCENTILE = 25
NUM_ENVIRONMENT_TYPE = 12
NUM_ENVIRONMENT_PER_TYPE = 50
# DEPTH_MAP_RESOLUTION = 0.025


def main():
    type_dict = {0: [],
                 1: [],
                 2: []}

    with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/p2_ddyn', 'r') as file:
        p2_ddyn = pickle.load(file)
    for example_id in p2_ddyn:
        val = p2_ddyn[example_id]
        type_dict[val[2]].append(val[3])
        
    clipped_a = np.clip(np.array(type_dict[0]), 0, 2000)
    hist_a, _ = np.histogram(clipped_a, bins=np.arange(0, 2010, 10))
    environment_a_hist = hist_a
    clipped_b = np.clip(np.array(type_dict[1]), 0, 2000)
    hist_b, _ = np.histogram(clipped_b, bins=np.arange(0, 2010, 10))
    environment_b_hist = hist_b
    clipped_c = np.clip(np.array(type_dict[2]), 0, 2000)
    hist_c, _ = np.histogram(clipped_c, bins=np.arange(0, 2010, 10))
    environment_c_hist = hist_c
 
    plt.figure()
    plt.plot(range(0, 2000, 10), environment_a_hist, '-o', label='no wall', color='red')
    plt.plot(range(0, 2000, 10), environment_b_hist, '-o', label='one wall only', color='green')
    plt.plot(range(0, 2000, 10), environment_c_hist, '-o', label='two walls', color='blue')
    plt.title('distribution of ground truth')
    plt.xlabel('ground truth')
    plt.ylabel('number of transitions')
    plt.legend()
    plt.savefig('../data/dynamic_cost_accurate.png')


if __name__ == '__main__':
    main()


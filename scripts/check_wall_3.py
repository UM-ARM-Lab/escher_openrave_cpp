"""

"""

import pickle, IPython, os, math, shutil, getopt, sys
import numpy as np
import matplotlib.pyplot as plt

from generate_depth_map import rotate_quadrilaterals, entire_depth_map

GRID_RESOLUTION = 0.15
ANGLE_RESOLUTION = 15.0
PERCENTILE = 25
NUM_ENVIRONMENT_TYPE = 12
NUM_ENVIRONMENT_PER_TYPE = 50
DEPTH_MAP_RESOLUTION = 0.025


def arctan(x, y):
    if abs(x) < 0.1:
        if y > 0.1:
            return 90.0
        elif y < -0.1:
            return -90.0
        else:
            return 0
    else:
        return math.atan2(y, x) * 180.0 / math.pi


def main():
    p1_thetas = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

    wall_dict = {}
    
    for environment_type in range(NUM_ENVIRONMENT_TYPE):
        with open('../data/medium_dataset_normal_wall/environments_' + str(environment_type), 'r') as env_file:
            environments = pickle.load(env_file)
            for environment_index in range(NUM_ENVIRONMENT_PER_TYPE):
                print('process environment: {} index: {}'.format(environment_type, environment_index))
                others_vertices = environments[environment_index]['others_vertices']
                for p1_theta in p1_thetas:
                    wall_patch_coordinates = rotate_quadrilaterals(others_vertices, p1_theta * ANGLE_RESOLUTION)
                    wall_depth_map = entire_depth_map(wall_patch_coordinates, 'wall', DEPTH_MAP_RESOLUTION)
                    wall_dict[str(environment_type) + '_' + str(environment_index) + '_00' + str(p1_theta)] = wall_depth_map

    list_dict = {'a': [],
                 'bl': [],
                 'br': [],
                 'c': []}

    with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/p2_ddyn', 'r') as file:
        p2_ddyn = pickle.load(file)

    for i in p2_ddyn:
        p1 = p2_ddyn[i][0]
        p2_x = p2_ddyn[i][1][0]
        p2_y = p2_ddyn[i][1][1]
        ddyn = p2_ddyn[i][2]

        alpha = arctan(p2_x, p2_y)
        edge = int(math.ceil(1.0*176/360*(180-alpha)))
        wall_depth_map = wall_dict[p1]
        left = np.sum(np.ones((41, edge), dtype=float) * 0.7 - wall_depth_map[:, :edge])
        right = np.sum(np.ones((41, 176-edge), dtype=float) * 0.7 - wall_depth_map[:, edge:])
        if left > 0.01:
            if right > 0.01:
                list_dict['c'].append(ddyn)
            else:
                list_dict['bl'].append(ddyn)
        else:
            if right > 0.01:
                list_dict['br'].append(ddyn)
            else:
                list_dict['a'].append(ddyn)

    arra = np.clip(np.array(list_dict['a']), 0, 2000)
    hista, _ = np.histogram(arra, bins=np.arange(0, 2010, 10))
    arrbl = np.clip(np.array(list_dict['bl']), 0, 2000)
    histbl, _ = np.histogram(arrbl, bins=np.arange(0, 2010, 10))
    arrbr = np.clip(np.array(list_dict['br']), 0, 2000)
    histbr, _ = np.histogram(arrbr, bins=np.arange(0, 2010, 10))
    arrc = np.clip(np.array(list_dict['c']), 0, 2000)
    histc, _ = np.histogram(arrc, bins=np.arange(0, 2010, 10))
 
    plt.figure()
    plt.plot(range(200), hista, '-o', label='no wall', color='red')
    plt.plot(range(200), histbl, '-o', label='left wall only', color='green')
    plt.plot(range(200), histbr, '-o', label='right wall only', color='yellow')
    plt.plot(range(200), histc, '-o', label='two walls', color='blue')
    plt.title('distribution of 25 percentiles')
    plt.xlabel('25 percentile')
    plt.ylabel('number of examples')
    plt.legend()
    plt.savefig('../data/percentile_inaccurate.png')

     

if __name__ == '__main__':
    main()


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
ground map around p1, wall map around p1, p2 -> 10 percentile

Data will be saved in a dict.
key is example id
map_id (string), p2, environment wall type (0: no wall, 1: one wall only, 2: two walls), 25 percentile ddyn (float)
"""

import pickle, IPython, os, math, shutil, getopt, sys, random, pprint
import numpy as np
# import matplotlib.pyplot as plt

# from generate_boundary_and_depth_map import generateGroundDepthBoundaryMap, generateWallDepthBoundaryMap

GRID_RESOLUTION = 0.15
ANGLE_RESOLUTION = 22.5
PERCENTILE1 = 10
PERCENTILE2 = 25
MAX_TRANSITIONS_CHOSEN = 200

def main():
        with open('specific_dynamic_cost', 'r') as file:
            data = pickle.load(file)

        ddyn_list = []
        p1_list = sorted(data.keys(), key=lambda element: (element[0], element[1], element[2]))
        for p1 in p1_list:  
            p2_list = sorted(data[p1].keys(), key=lambda element: (element[0], element[1], element[2]))
            for p2 in p2_list:   
                for transition_type in data[p1][p2]:
                    for transition in data[p1][p2][transition_type]:
                        ddyns = data[p1][p2][transition_type][transition].tolist()
                        random.shuffle(ddyns)
                        # total_transition += 1
                        # if len(ddyns) > MAX_TRANSITIONS_CHOSEN:
                        #     large_transition += 1
                        ddyn_list += ddyns[:min(len(ddyns), MAX_TRANSITIONS_CHOSEN)]

        if not ddyn_list:
            print('no feasible contact transition')
        else:
            print(str(PERCENTILE1) + ' percentile is ' + str(np.percentile(np.array(ddyn_list), PERCENTILE1)))
            print(str(PERCENTILE2) + ' percentile is ' + str(np.percentile(np.array(ddyn_list), PERCENTILE2)))


if __name__ == '__main__':
    main()


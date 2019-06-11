"""
Generate training examples for model: (p1, p2) -> 25 percentile ddyn
"""

import pickle, IPython, os, math, shutil
import numpy as np

GRID_RESOLUTION = 0.15
ANGLE_RESOLUTION = 15.0
PERCENTILE = 25
NUM_ENVIRONMENT_TYPE = 12
NUM_ENVIRONMENT_PER_TYPE = 50

def discretize_torso_pose(arr, resolution):
    new_arr = np.zeros_like(arr, dtype=int)
    indices = np.argwhere(np.absolute(arr) > resolution/2.0).reshape(-1,)
    values = arr[indices]
    values -= np.sign(values) * resolution/2.0
    new_arr[indices] = np.sign(values) * np.ceil(np.round(np.absolute(values) / resolution, 1)).astype(int)
    return new_arr


def main():
    env_list = []
    p1_list = []
    p2_list = []
    ddyn_list = []
    for environment_type in range(NUM_ENVIRONMENT_TYPE):
        for environment_index in range(NUM_ENVIRONMENT_PER_TYPE):
            if os.path.exists('../data/medium_dataset/dynamic_cost_' + str(environment_type) + '_' + str(environment_index)):
                print('process data in file dynamic_cost_{}_{}'.format(environment_type, environment_index))
                with open('../data/medium_dataset/dynamic_cost_' + str(environment_type) + '_' + str(environment_index), 'r') as file:
                    data = pickle.load(file)
                    p1 = data['p1']
                    discrete_p1 = np.zeros_like(p1, dtype=int)
                    discrete_p1[:, 0] = discretize_torso_pose(p1[:, 0], GRID_RESOLUTION)
                    discrete_p1[:, 1] = discretize_torso_pose(p1[:, 1], GRID_RESOLUTION)
                    discrete_p1[:, 2] = discretize_torso_pose(p1[:, 2], ANGLE_RESOLUTION)
                    unique_discrete_p1 = np.unique(discrete_p1, axis=0)
                    p2 = data['p2']
                    discrete_p2 = np.zeros_like(p2, dtype=int)
                    discrete_p2[:, 0] = discretize_torso_pose(p2[:, 0], GRID_RESOLUTION)
                    discrete_p2[:, 1] = discretize_torso_pose(p2[:, 1], GRID_RESOLUTION)
                    discrete_p2[:, 2] = discretize_torso_pose(p2[:, 2], ANGLE_RESOLUTION)
                    unique_discrete_p2 = np.unique(discrete_p2, axis=0)
                    ddyn = data['ddyn']
                    for ip1 in range(unique_discrete_p1.shape[0]):
                        for ip2 in range(unique_discrete_p2.shape[0]):
                            mask = np.logical_and((discrete_p1 == unique_discrete_p1[ip1]).all(axis=1), (discrete_p2 == unique_discrete_p2[ip2]).all(axis=1))
                            if np.sum(mask) == 0:
                                continue
                            env_list.append((environment_type, environment_index))
                            p1_list.append(unique_discrete_p1[ip1])
                            p2_list.append(unique_discrete_p2[ip2])
                            ddyn_list.append(np.percentile(ddyn[np.argwhere(mask).reshape(-1,)], PERCENTILE))
                            indices = np.argwhere(mask == False).reshape(-1,)
                            discrete_p1 = discrete_p1[indices]
                            discrete_p2 = discrete_p2[indices]
                            ddyn = ddyn[indices]
                # IPython.embed()
    with open('../data/ground_truth_p1p2/env_list', 'w') as file:
        pickle.dump(env_list, file)
    with open('../data/ground_truth_p1p2/p1_list', 'w') as file:
        pickle.dump(p1_list, file)
    with open('../data/ground_truth_p1p2/p2_list', 'w') as file:
        pickle.dump(p2_list, file)
    with open('../data/ground_truth_p1p2/ddyn_list', 'w') as file:
        pickle.dump(ddyn_list, file)

    
if __name__ == '__main__':
    main()


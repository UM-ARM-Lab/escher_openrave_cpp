"""
Generate training examples for model: (p1, init x, p2, final x) -> 25 percentile ddyn
"""

import pickle, IPython, os, math, shutil
import numpy as np
import matplotlib.pyplot as plt
import timeit

GRID_RESOLUTION = 0.15
ANGLE_RESOLUTION = 15.0
PERCENTILE = 25
NUM_ENVIRONMENT_TYPE = 12
NUM_ENVIRONMENT_PER_TYPE = 50
X_BOUNDARY_1 = -0.2
X_BOUNDARY_2 = -0.1
X_TYPE = 3

def discretize_torso_pose(arr, resolution):
    new_arr = np.zeros_like(arr, dtype=int)
    indices = np.argwhere(np.absolute(arr) > resolution/2.0).reshape(-1,)
    values = arr[indices]
    values -= np.sign(values) * resolution/2.0
    new_arr[indices] = np.sign(values) * np.ceil(np.round(np.absolute(values) / resolution, 1)).astype(int)
    return new_arr
    

def discretize_x(X, x_boundary_1, x_boundary_2):
    """
    (-Inf, x_boundary_1): 0
    [x_boundary_1, x_boundary_2): 1
    [x_boundary_2, Inf): 2
    """
    new_X = np.zeros_like(X, dtype=int)
    new_X[np.argwhere(X >= x_boundary_1).reshape(-1,)] = 1
    new_X[np.argwhere(X >= x_boundary_2).reshape(-1,)] = 2
    return new_X


def main():
    env_list = []
    p1_list = []
    p2_list = []
    init_x_list = []
    final_x_list = []
    ddyn_list = []

    for environment_type in range(NUM_ENVIRONMENT_TYPE):
        # initial_x_arr = np.zeros((1,), dtype=float)
        # final_x_arr = np.zeros((1,), dtype=float)
        for environment_index in range(NUM_ENVIRONMENT_PER_TYPE):
            if os.path.exists('../data/medium_dataset/dynamic_cost_' + str(environment_type) + '_' + str(environment_index)):
                print('process data in file dynamic_cost_{}_{}'.format(environment_type, environment_index))
                with open('../data/medium_dataset/dynamic_cost_' + str(environment_type) + '_' + str(environment_index), 'r') as file:
                    data = pickle.load(file)
                    # initial_x_arr = np.concatenate((initial_x_arr, data['initial_com_position'][1:, 0]))
                    # final_x_arr = np.concatenate((final_x_arr, data['final_com_position'][1:, 0]))
                    ddyn = data['ddyn'][1:]
                    # p1, init x, p2, final x, ddyn
                    arr = np.zeros((ddyn.shape[0], 9), dtype=int)
                    p1 = data['p1'][1:]
                    arr[:, 0] = discretize_torso_pose(p1[:, 0], GRID_RESOLUTION)
                    arr[:, 1] = discretize_torso_pose(p1[:, 1], GRID_RESOLUTION)
                    arr[:, 2] = discretize_torso_pose(p1[:, 2], ANGLE_RESOLUTION)
                    arr[:, 3] = discretize_x(data['initial_com_position'][1:, 0], X_BOUNDARY_1, X_BOUNDARY_2)
                    p2 = data['p2'][1:]
                    arr[:, 4] = discretize_torso_pose(p2[:, 0], GRID_RESOLUTION)
                    arr[:, 5] = discretize_torso_pose(p2[:, 1], GRID_RESOLUTION)
                    arr[:, 6] = discretize_torso_pose(p2[:, 2], ANGLE_RESOLUTION)
                    arr[:, 7] = discretize_x(data['final_com_position'][1:, 0], X_BOUNDARY_1, X_BOUNDARY_2)
                    arr[:, 8] = ddyn.astype(int)
                    start = timeit.default_timer()
                    for ip1 in range(unique_discrete_p1.shape[0]):
                        for ip2 in range(unique_discrete_p2.shape[0]):
                            for iix in range(X_TYPE):
                                for ifx in range(X_TYPE):
                                    mask = np.logical_and((discrete_p1 == unique_discrete_p1[ip1]).all(axis=1), (discrete_p2 == unique_discrete_p2[ip2]).all(axis=1))
                                    mask = np.logical_and(mask, discrete_initial_x == iix)
                                    mask = np.logical_and(mask, discrete_final_x == ifx)
                                    if np.sum(mask) == 0:
                                        continue
                                    # env_list.append((environment_type, environment_index))
                                    # p1_list.append(unique_discrete_p1[ip1])
                                    # p2_list.append(unique_discrete_p2[ip2])
                                    # init_x_list.append(iix)
                                    # final_x_list.append(ifx)
                                    d = np.percentile(ddyn[np.argwhere(mask).reshape(-1,)], PERCENTILE)
                                    # ddyn_list.append(d)
                                    # print('p1: {} p2: {} init x: {} final x: {} ddyn: {} size {}'.format(unique_discrete_p1[ip1], unique_discrete_p2[ip2], iix, ifx, d, np.sum(mask)))
                                    indices = np.argwhere(mask == False).reshape(-1,)
                                    discrete_p1 = discrete_p1[indices]
                                    discrete_p2 = discrete_p2[indices]
                                    discrete_initial_x = discrete_initial_x[indices]
                                    discrete_final_x = discrete_final_x[indices]
                                    ddyn = ddyn[indices]
                    print(timeit.default_timer() - start)
                    # IPython.embed()
        # initial_x_clipped = np.clip(initial_x_arr[1:], -0.4, 0.3)
        # plt.figure(environment_type * 2)
        # plt.hist(initial_x_clipped, bins=np.arange(-0.5, 0.4, 0.01))
        # plt.title('environment type: {}, initial x'.format(environment_type))
        # plt.savefig('../data/ground_truth_p1p2x/{}_initial_x.png'.format(environment_type))

        # final_x_clipped = np.clip(final_x_arr[1:], -0.4, 0.3)
        # plt.figure(environment_type * 2 + 1)
        # plt.hist(final_x_clipped, bins=np.arange(-0.5, 0.4, 0.01))
        # plt.title('environment type: {}, final x'.format(environment_type))
        # plt.savefig('../data/ground_truth_p1p2x/{}_final_x.png'.format(environment_type))

    # with open('../data/ground_truth_p1p2x/env_list', 'w') as file:
    #     pickle.dump(env_list, file)
    # with open('../data/ground_truth_p1p2x/p1_list', 'w') as file:
    #     pickle.dump(p1_list, file)
    # with open('../data/ground_truth_p1p2x/p2_list', 'w') as file:
    #     pickle.dump(p2_list, file)
    # with open('../data/ground_truth_p1p2x/ddyn_list', 'w') as file:
    #     pickle.dump(ddyn_list, file)

    
if __name__ == '__main__':
    main()


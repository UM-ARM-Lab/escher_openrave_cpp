import pickle, IPython, math, sys, getopt, keras
import numpy as np
# import mkl
# mkl.get_max_threads()
# import faiss
from keras.models import load_model
import tensorflow as tf
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


def adjust_com(com_before_adjustment, original_frame, new_frame):
    """
    "original_frame" is mean_feet_pose, which has 6 entries

    "new_frame" is torse pose, which has 3 entries
    """
    original_x = original_frame[0]
    original_y = original_frame[1]
    original_z = original_frame[2]
    original_yaw = original_frame[5]
    original_yaw_in_radian = original_yaw / 180.0 * np.pi

    global_com = np.zeros_like(com_before_adjustment)
    global_com[:, 0] = original_x - com_before_adjustment[:, 1] * np.sin(original_yaw_in_radian) + com_before_adjustment[:, 0] * np.cos(original_yaw_in_radian)
    global_com[:, 1] = original_y + com_before_adjustment[:, 1] * np.cos(original_yaw_in_radian) + com_before_adjustment[:, 0] * np.sin(original_yaw_in_radian)
    global_com[:, 2] = original_z + com_before_adjustment[:, 2]

    new_x = new_frame[0]
    new_y = new_frame[1]
    new_z = 0
    new_yaw = new_frame[2]
    new_yaw_in_radian = new_yaw / 180.0 * np.pi
    rotation_matrix = np.array([[np.cos(-new_yaw_in_radian), -np.sin(-new_yaw_in_radian)],
                                [np.sin(-new_yaw_in_radian), np.cos(-new_yaw_in_radian)]])
    com_after_adjustment = np.copy(global_com)
    com_after_adjustment[:, 0:2] = np.matmul(rotation_matrix, (global_com - np.array([new_x, new_y, new_z]))[:, 0:2].T).T
    return com_after_adjustment


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
    device = None
    environment_type = None
    try:
        inputs, _ = getopt.getopt(sys.argv[1:], "d:e:", ['device', 'environment_type'])

        for opt, arg in inputs:
            if opt == '-d':
                device = arg

            if opt == '-e':
                environment_type = arg

    except getopt.GetoptError:
        print('usage: -d: [cpu / gpu] -e: [environment_type]')
        exit(1)

    if device == 'cpu':
        pass
    elif device == 'gpu':
        config = tf.ConfigProto(device_count={'GPU':1, 'CPU':3}, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95))
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)
    else:
        print('wrong device')
        exit(1)

    # load sampled COM combinations of all types
    com_combinations = {}
    for i in range(10):
        with open('../data/CoM/com_combinations_' + str(i), 'r') as file:
            com_combinations[i] = pickle.load(file)

    # load the normalize parameters for the classification model of all types
    classification_input_normalize_params = []
    for i in range(10):
        with open('../data/dynopt_result/feasibility_classification_nn_models/input_mean_std_' + str(i) + '_0.0001_256_0.1.txt', 'r') as file:
            strings = file.readline().strip().split(' ')
            params = np.zeros((2, len(strings) // 2), dtype=float)
            for j in range(len(strings) // 2):
                params[0, j] = float(strings[2 * j])
                params[1, j] = float(strings[2 * j + 1])
            classification_input_normalize_params.append(params)

    # load the classification models of all types
    classification_models = []
    for i in range(10):
        classification_models.append(load_model('../data/dynopt_result/feasibility_classification_nn_models/nn_model_' + str(i) + '_0.0001_256_0.1.h5'))

    # load the normalize parameters for the regression model of all types
    regression_input_normalize_params = []
    for i in range(10):
        with open('../data/dynopt_result/objective_regression_nn_models/input_mean_std_' + str(i) + '_0.0005_256_0.0.txt', 'r') as file:
            strings = file.readline().strip().split(' ')
            params = np.zeros((2, len(strings) // 2), dtype=float)
            for j in range(len(strings) // 2):
                params[0, j] = float(strings[2 * j])
                params[1, j] = float(strings[2 * j + 1])
            regression_input_normalize_params.append(params)

    # load the denormalize parameters for the regression model of all types
    regression_output_denormalize_params = []
    for i in range(10):
        with open('../data/dynopt_result/objective_regression_nn_models/output_mean_std_' + str(i) + '_0.0005_256_0.0.txt', 'r') as file:
            strings = file.readline().strip().split(' ')
            params = np.zeros((2, len(strings) // 2), dtype=float)
            for j in range(len(strings) // 2):
                params[0, j] = float(strings[2 * j])
                params[1, j] = float(strings[2 * j + 1])
            regression_output_denormalize_params.append(params)

    # load the regression models of all types
    regression_models = []
    for i in range(10):
        regression_models.append(load_model('../data/dynopt_result/objective_regression_nn_models/nn_model_' + str(i) + '_0.0005_256_0.0.h5'))

    # for distance to training data of classification model and regression model
    # classification_training_data_mean = {}
    # classification_training_data_std = {}
    # classification_training_data_tree = {}
    # regression_training_data_mean = {}
    # regression_training_data_std = {}
    # regression_training_data_tree = {}
    # for i in range(10):
    #     file = open('../data/dynopt_result/dataset/dynopt_total_data_' + str(i), 'r')
    #     original_X = pickle.load(file)[:, 1:-7]
    #     regression_training_data_mean[i] = np.mean(original_X, axis=0, dtype=np.float32)
    #     regression_training_data_std[i] = np.std(original_X, axis=0, dtype=np.float32)
    #     normalized_original_X = (original_X - regression_training_data_mean[i]) / regression_training_data_std[i]
    #     regression_training_data_tree[i] = faiss.IndexFlatL2(normalized_original_X.shape[1])
    #     regression_training_data_tree[i].add(np.float32(normalized_original_X))
    #
    #     file = open('../data/dynopt_result/dataset/dynopt_infeasible_total_data_' + str(i), 'r')
    #     infeasible_original_X = pickle.load(file)[:, 1:-1]
    #     all_original_X = np.concatenate((original_X, infeasible_original_X), axis=0)
    #     classification_training_data_mean[i] = np.mean(all_original_X, axis=0, dtype=np.float32)
    #     classification_training_data_std[i] = np.std(all_original_X, axis=0, dtype=np.float32)
    #     normalized_all_original_X = (all_original_X - classification_training_data_mean[i]) / classification_training_data_std[i]
    #     classification_training_data_tree[i] = faiss.IndexFlatL2(normalized_all_original_X.shape[1])
    #     classification_training_data_tree[i].add(np.float32(normalized_all_original_X))

    # load sampled transitions
    transitions = None
    with open('../data/medium_dataset_normal_wall/transitions_' + environment_type, 'r') as file:
        transitions = pickle.load(file)

    print("environment type: " + environment_type)

    # info is a nested dictionary.
    # its first key is p1 (tuple)
    # its second key is p2 (tuple)
    # its value is (initial_com_position, final_com_position, ddyn) (numpy array)
    info = {}

    prev_environment_index = 0

    for idx, transition in enumerate(transitions):
        environment_index = transition['environment_index']
        # if environment_index < 38:
        #     continue
        # if environment_index > 5:
        #     break
        if environment_index != prev_environment_index:
            print('start save data to file dynamic_cost_plus_type_{}_{}'.format(environment_type, prev_environment_index))
            with open('../data/medium_dataset_normal_wall/dynamic_cost_plus_type_' + str(environment_type) + '_' + str(prev_environment_index), 'w') as file:
                pickle.dump(info, file)
            print('finish save data to file dynamic_cost_plus_type_{}_{}'.format(environment_type, prev_environment_index))
            info = {}
            prev_environment_index = environment_index

        transition_type = transition['contact_transition_type']
        com = com_combinations[transition_type]

        X = np.zeros((len(com), len(transition['feature_vector_contact_part']) + 6), dtype=float)
        X[:, 0:-6] = np.tile(np.array(transition['feature_vector_contact_part']), (len(com), 1))
        X[:, -6:] = np.array(com)

        # the distance between com position and each foot contact should be in [0, 1.1]
        valid_com_indices = np.sum((np.array(transition['normalized_init_l_leg'][:3]) - X[:, -6:-3]) ** 2, axis=1) <= 1.1 ** 2
        if np.sum(valid_com_indices) == 0:
            continue
        X = X[np.argwhere(valid_com_indices == True).reshape(-1,)]
        valid_com_indices = np.sum((np.array(transition['normalized_init_r_leg'][:3]) - X[:, -6:-3]) ** 2, axis=1) <= 1.1 ** 2
        if np.sum(valid_com_indices) == 0:
            continue
        X = X[np.argwhere(valid_com_indices == True).reshape(-1,)]

        # the distance between com position and each palm contact (if exists) should be in [0, 0.8]
        if transition['normalized_init_l_arm']:
            valid_com_indices = np.sum((np.array(transition['normalized_init_l_arm'][:3]) - X[:, -6:-3]) ** 2, axis=1) <= 0.8 ** 2
            if np.sum(valid_com_indices) == 0:
                continue
            X = X[np.argwhere(valid_com_indices == True).reshape(-1,)]
        if transition['normalized_init_r_arm']:
            valid_com_indices = np.sum((np.array(transition['normalized_init_r_arm'][:3]) - X[:, -6:-3]) ** 2, axis=1) <= 0.8 ** 2
            if np.sum(valid_com_indices) == 0:
                continue
            X = X[np.argwhere(valid_com_indices == True).reshape(-1,)]

        # check the distance to the training data of classification model
        # dist, _ = classification_training_data_tree[transition_type].search(np.float32((X - classification_training_data_mean[transition_type]) / classification_training_data_std[transition_type]), 10)
        # dist = np.mean(np.sqrt(dist.clip(min=0)), axis=1)
        # valid_com_indices = dist < 3.0
        # if np.sum(valid_com_indices) == 0:
        #     continue
        # X = X[np.argwhere(valid_com_indices == True).reshape(-1,)]

        # query the classification model
        prediction = classification_models[transition_type].predict((X - classification_input_normalize_params[transition_type][0]) / classification_input_normalize_params[transition_type][1])
        valid_com_indices = prediction.reshape(-1,) > 0.5
        if np.sum(valid_com_indices) == 0:
            continue
        X = X[np.argwhere(valid_com_indices == True).reshape(-1,)]

        # check the distance to the training data of the regression model
        # dist, _ = regression_training_data_tree[transition_type].search(np.float32((X - regression_training_data_mean[transition_type]) / regression_training_data_std[transition_type]), 10)
        # dist = np.mean(np.sqrt(dist.clip(min=0)), axis=1)
        # valid_com_indices = dist < 3.0
        # if np.sum(valid_com_indices) == 0:
        #     continue
        # X = X[np.argwhere(valid_com_indices == True).reshape(-1,)]

        # query the regression model
        prediction = regression_models[transition_type].predict((X - regression_input_normalize_params[transition_type][0]) / regression_input_normalize_params[transition_type][1]) * regression_output_denormalize_params[transition_type][1] + regression_output_denormalize_params[transition_type][0]
       
        temp_p1 = transition['p1']
        discretized_p1 = tuple(discretize_torso_pose([temp_p1[0], temp_p1[1], temp_p1[5]]))
        if discretized_p1 not in info:
            info[discretized_p1] = {}
            
        temp_p2 = transition['p2']
        adjusted_p2 = adjust_p2([temp_p1[0], temp_p1[1], temp_p1[5]], [temp_p2[0], temp_p2[1], temp_p2[5]])
        discretized_p2 = tuple(discretize_torso_pose(adjusted_p2))
        if discretized_p2 not in info[discretized_p1]:
            info[discretized_p1][discretized_p2] = {}

        initial_com_after_adjustment = adjust_com(X[:, -6:-3], transition['mean_feet_pose'], np.array([temp_p1[0], temp_p1[1], temp_p1[5]]))
        final_com_after_adjustment = adjust_com(prediction[:, 0:3], transition['mean_feet_pose'], np.array([temp_p2[0], temp_p2[1], temp_p2[5]]))
        
        if transition_type in info[discretized_p1][discretized_p2]:
            info[discretized_p1][discretized_p2][transition_type] = np.concatenate((info[discretized_p1][discretized_p2][transition_type], np.concatenate((initial_com_after_adjustment, final_com_after_adjustment, prediction[:, 6:7]), axis=1)), axis=0)
        else:
            info[discretized_p1][discretized_p2][transition_type] = np.concatenate((initial_com_after_adjustment, final_com_after_adjustment, prediction[:, 6:7]), axis=1)
    print('start save data to file dynamic_cost_plus_type_{}_{}'.format(environment_type, prev_environment_index))
    with open('../data/medium_dataset_normal_wall/dynamic_cost_plus_type_' + str(environment_type) + '_' + str(prev_environment_index), 'w') as file:
        pickle.dump(info, file)
    print('finish save data to file dynamic_cost_plus_type_{}_{}'.format(environment_type, prev_environment_index)) 

if __name__ == "__main__":
    main()

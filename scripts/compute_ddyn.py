import pickle, IPython, math, sys, getopt, keras
import numpy as np
import mkl
mkl.get_max_threads()
import faiss
from keras.models import load_model
import tensorflow as tf
# import timeit

GRID_RESOLUTION = 0.15
ANGLE_RESOLUTION = 15.0


# # com_dict
# # x
# # 0: (-Inf, -0.3)
# # 1: [-0.3, -0.2)
# # 2: [-0.2, -0.1)
# # 3: [-0.1, 0.0)
# # 4: [0.0, 0.1)
# # 5: [0.1, 0.2)
# # 6: [0.2, Inf)
# # y
# # 0: (-Inf, -0.1)
# # 1: [-0.1, 0.0)
# # 2: [0.0, 0.1)
# # 3: [0.1, 0.2)
# # 4: [0.2, Inf)
# # z
# # 0: (-Inf, 0.8)
# # 1: [0.8, 0.9)
# # 2: [0.9, 1.0)
# # 3: [1.0, 1.1)
# # 4: [1.1, Inf)
# def com_index(x):
#     idxx = max(min(int(math.floor(x[-6] * 10) + 4), 6), 0)
#     idxy = max(min(int(math.floor(x[-5] * 10) + 2), 4), 0)
#     idxz = max(min(int(math.floor(x[-4] * 10) - 7), 4), 0)
#     return (idxx, idxy, idxz)

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
        config = tf.ConfigProto(device_count={'GPU':1, 'CPU':3}, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.45))
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)
    else:
        print('wrong device')
        exit(1)

    # load sampled COM combinations of all types
    com_combinations = {}
    for i in range(1):
        file = open('../data/CoM/com_combinations_' + str(i), 'r')
        com_combinations[i] = pickle.load(file)

    # load the normalize parameters for the classification model of all types
    classification_input_normalize_params = []
    for i in range(1):
        file = open('../data/dynopt_result/feasibility_classification_nn_models/input_mean_std_' + str(i) + '_0.0001_256_0.1.txt', 'r')
        strings = file.readline().strip().split(' ')
        params = np.zeros((2, len(strings) // 2), dtype=float)
        for j in range(len(strings) // 2):
            params[0, j] = float(strings[2 * j])
            params[1, j] = float(strings[2 * j + 1])
        classification_input_normalize_params.append(params)

    # load the classification models of all types
    classification_models = []
    for i in range(1):
        classification_models.append(load_model('../data/dynopt_result/feasibility_classification_nn_models/nn_model_' + str(i) + '_0.0001_256_0.1.h5'))

    # load the normalize parameters for the regression model of all types
    regression_input_normalize_params = []
    for i in range(1):
        file = open('../data/dynopt_result/objective_regression_nn_models/input_mean_std_' + str(i) + '_0.0005_256_0.0.txt', 'r')
        strings = file.readline().strip().split(' ')
        params = np.zeros((2, len(strings) // 2), dtype=float)
        for j in range(len(strings) // 2):
            params[0, j] = float(strings[2 * j])
            params[1, j] = float(strings[2 * j + 1])
        regression_input_normalize_params.append(params)

    # load the denormalize parameters for the regression model of all types
    regression_output_denormalize_params = []
    for i in range(1):
        file = open('../data/dynopt_result/objective_regression_nn_models/output_mean_std_' + str(i) + '_0.0005_256_0.0.txt', 'r')
        strings = file.readline().strip().split(' ')
        params = np.zeros((2, len(strings) // 2), dtype=float)
        for j in range(len(strings) // 2):
            params[0, j] = float(strings[2 * j])
            params[1, j] = float(strings[2 * j + 1])
        regression_output_denormalize_params.append(params)

    # load the regression models of all types
    regression_models = []
    for i in range(1):
        regression_models.append(load_model('../data/dynopt_result/objective_regression_nn_models/nn_model_' + str(i) + '_0.0005_256_0.0.h5'))

    # for distance to training data of classification model and regression model
    classification_training_data_mean = {}
    classification_training_data_std = {}
    classification_training_data_tree = {}
    regression_training_data_mean = {}
    regression_training_data_std = {}
    regression_training_data_tree = {}
    for i in range(1):
        file = open('../data/dynopt_result/dataset/dynopt_total_data_' + str(i), 'r')
        original_X = pickle.load(file)[:, 1:-7]
        regression_training_data_mean[i] = np.mean(original_X, axis=0, dtype=np.float32)
        regression_training_data_std[i] = np.std(original_X, axis=0, dtype=np.float32)
        normalized_original_X = (original_X - regression_training_data_mean[i]) / regression_training_data_std[i]
        regression_training_data_tree[i] = faiss.IndexFlatL2(normalized_original_X.shape[1])
        regression_training_data_tree[i].add(np.float32(normalized_original_X))

        file = open('../data/dynopt_result/dataset/dynopt_infeasible_total_data_' + str(i), 'r')
        infeasible_original_X = pickle.load(file)[:, 1:-1]
        all_original_X = np.concatenate((original_X, infeasible_original_X), axis=0)
        classification_training_data_mean[i] = np.mean(all_original_X, axis=0, dtype=np.float32)
        classification_training_data_std[i] = np.std(all_original_X, axis=0, dtype=np.float32)
        normalized_all_original_X = (all_original_X - classification_training_data_mean[i]) / classification_training_data_std[i]
        classification_training_data_tree[i] = faiss.IndexFlatL2(normalized_all_original_X.shape[1])
        classification_training_data_tree[i].add(np.float32(normalized_all_original_X))
    
    # environ_pose_to_ddyn is a nested dictionary.
    # the first key is environment index in each environment type;
    # the second key is initial pose and final pose;
    # the value is a vector of dynamic cost.
    environ_pose_to_ddyn = {}

    # load sampled transitions
    file = open('../data/large_dataset_resolution_015/large_transitions_' + environment_type, 'r')
    transitions = pickle.load(file)

    print("environment type: " + environment_type)

    for idx, transition in enumerate(transitions):
        environment_index = transition['environment_index']
        if idx % 1000 == 0:
            print("transition: " + str(idx))

        # start = timeit.default_timer()

        transition_type = transition['contact_transition_type']
        com = com_combinations[transition_type]

        X = np.zeros((len(com), len(transition['feature_vector_contact_part']) + 6), dtype=float)
        X[:, 0:-6] = np.tile(np.array(transition['feature_vector_contact_part']), (len(com), 1))
        X[:, -6:] = np.array(com)

        # print("construct X", timeit.default_timer() - start)

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

        # print("check position", timeit.default_timer() - start)

        # check the distance to the training data of classification model
        # dist, _ = classification_training_data_tree[transition_type].search(np.float32((X - classification_training_data_mean[transition_type]) / classification_training_data_std[transition_type]), 10)
        # dist = np.mean(np.sqrt(dist.clip(min=0)), axis=1)
        # valid_com_indices = dist < 3.0
        # if np.sum(valid_com_indices) == 0:
        #     continue
        # X = X[np.argwhere(valid_com_indices == True).reshape(-1,)]

        # print("check distance", timeit.default_timer() - start)

        # query the classification model
        prediction = classification_models[transition_type].predict((X - classification_input_normalize_params[transition_type][0]) / classification_input_normalize_params[transition_type][1])
        valid_com_indices = prediction.reshape(-1,) > 0.5
        if np.sum(valid_com_indices) == 0:
            continue
        X = X[np.argwhere(valid_com_indices == True).reshape(-1,)]

        # print("query classification model", timeit.default_timer() - start)

        # check the distance to the training data of the regression model
        # dist, _ = regression_training_data_tree[transition_type].search(np.float32((X - regression_training_data_mean[transition_type]) / regression_training_data_std[transition_type]), 10)
        # dist = np.mean(np.sqrt(dist.clip(min=0)), axis=1)
        # valid_com_indices = dist < 3.0
        # if np.sum(valid_com_indices) == 0:
        #     continue
        # X = X[np.argwhere(valid_com_indices == True).reshape(-1,)]

        # print("check distance", timeit.default_timer() - start)

        # query the regression model
        prediction = regression_models[transition_type].predict((X - regression_input_normalize_params[transition_type][0]) / regression_input_normalize_params[transition_type][1]) * regression_output_denormalize_params[transition_type][1] + regression_output_denormalize_params[transition_type][0]
        # ddyns = prediction[:, -1]

        # print("query regression model", timeit.default_timer() - start)

        if environment_index not in environ_pose_to_ddyn:
            environ_pose_to_ddyn[environment_index] = {}

        p1 = transition['p1']
        p2 = transition['p2']
        pose = tuple(p1 + p2)

        # [0, 1, 2]: initial CoM position, before rotation
        # [3, 4, 5]: final CoM position, before rotation
        # [6]: dynamic cost
        com_ddyn = np.concatenate((X[:, -6:-3], prediction[:, 0:3], prediction[:, 6:7]), axis=1)

        # if pose not in environ_pose_to_ddyn[environment_index]:
        #     environ_pose_to_ddyn[environment_index][pose] = {}

        # for j in range(X.shape[0]):
        #     com_idx = com_index(X[j])
        #     if com_idx in environ_pose_to_ddyn[environment_index][pose]:
        #         environ_pose_to_ddyn[environment_index][pose][com_idx].append(ddyns[j])
        #     else:
        #         environ_pose_to_ddyn[environment_index][pose][com_idx] = [ddyns[j]]

        if pose in environ_pose_to_ddyn[environment_index]:
            environ_pose_to_ddyn[environment_index][pose] = np.concatenate((environ_pose_to_ddyn[environment_index][pose], com_ddyn), axis=0)
        else:
            environ_pose_to_ddyn[environment_index][pose] = com_ddyn

    # IPython.embed()
    with open('../data/no_threshold_environ_pose_to_ddyn_' + environment_type, 'w') as file:
        pickle.dump(environ_pose_to_ddyn, file)


if __name__ == "__main__":
    main()

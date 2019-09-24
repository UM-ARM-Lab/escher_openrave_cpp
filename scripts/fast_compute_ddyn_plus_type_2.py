import pickle, IPython, math, sys, getopt, keras
import numpy as np
# import mkl
# mkl.get_max_threads()
# import faiss
from keras.models import load_model
import tensorflow as tf
# import timeit

GRID_RESOLUTION = 0.15
ANGLE_RESOLUTION = 22.5

def main():
    environment_type = None
    device = None
    try:
        inputs, _ = getopt.getopt(sys.argv[1:], "d:e:", ['device', 'environment_type'])

        for opt, arg in inputs:
            if opt == '-d':
                device = arg
            if opt == '-e':
                environment_type = arg

    except getopt.GetoptError:
        print('usage: -d: [cpu / gpu] -e: environment_type')
        exit(1)

    if device == 'cpu':
        pass
    elif device == 'gpu':
        # config = tf.ConfigProto(device_count={'GPU':1, 'CPU':3}, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
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

    for environment_index in range(56, 200):
        with open('/mnt/big_narstie_data/chenxi/data/dataset_225/transitions_dict_' + str(environment_type) + '_' + str(environment_index), 'r') as file:
            data = pickle.load(file)

        # info is a nested dictionary.
        # its first key is p1 (tuple)
        # its second key is p2 (tuple)
        # its third key is transition type
        # its value is (initial_com_position, final_com_position, ddyn) (numpy array)
        info = {}

        for p1 in data:
            info[p1] = {}

            for p2 in data[p1]:
                info[p1][p2] = {}

                for transition_type in data[p1][p2]:
                    info[p1][p2][transition_type] = {}

                    for index, transition in enumerate(data[p1][p2][transition_type]):
                        X = np.zeros((len(com_combinations[transition_type]), len(transition['feature_vector_contact_part']) + 6), dtype=float)
                        X[:, 0:-6] = np.tile(np.array(transition['feature_vector_contact_part']), (len(com_combinations[transition_type]), 1))
                        X[:, -6:] = np.array(com_combinations[transition_type])

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

                        # query the classification model
                        prediction = classification_models[transition_type].predict((X - classification_input_normalize_params[transition_type][0]) / classification_input_normalize_params[transition_type][1])
                        valid_com_indices = prediction.reshape(-1,) > 0.5
                        if np.sum(valid_com_indices) == 0:
                            continue
                        X = X[np.argwhere(valid_com_indices == True).reshape(-1,)]

                        # query the regression model
                        prediction = regression_models[transition_type].predict((X - regression_input_normalize_params[transition_type][0]) / regression_input_normalize_params[transition_type][1]) * regression_output_denormalize_params[transition_type][1] + regression_output_denormalize_params[transition_type][0]
                        info[p1][p2][transition_type][index] = prediction[:, 6]

        print('start save data to file dynamic_cost_plus_type_{}_{}'.format(environment_type, environment_index))
        with open('/mnt/big_narstie_data/chenxi/data/dataset_225/dynamic_cost_plus_type_' + str(environment_type) + '_' + str(environment_index), 'w') as file:
            pickle.dump(info, file)
        print('finish save data to file dynamic_cost_plus_type_{}_{}'.format(environment_type, environment_index))


if __name__ == "__main__":
    main()


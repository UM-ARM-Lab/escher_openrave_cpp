import pickle, IPython, math, sys, getopt, keras
import numpy as np
from keras.models import load_model

GRID_RESOLUTION = 0.05
ANGLE_RESOLUTION = 15.0

def main():
    device = None
    environment_type = None
    try:
        inputs, _ = getopt.getopt(sys.argv[1:], "d:t:", ['device', 'environment_type'])

        for opt, arg in inputs:
            if opt == '-d':
                device = arg

            if opt == '-t':
                environment_type = arg

    except getopt.GetoptError:
        print('usage: -d: [cpu / gpu] -t: [environment_type]')
        exit(1)

    if device == 'cpu':
        pass
    elif device == 'gpu':
        import tensorflow as tf
        config = tf.ConfigProto(device_count={'GPU':1, 'CPU':3}, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)
    else:
        print('wrong device')
        exit(1)

    # load sampled COM combinations of all types
    com_combinations = {}
    for i in range(10):
        file = open('../data/CoM/com_combinations_' + str(i), 'r')
        com_combinations[i] = pickle.load(file)

    # load the normalize parameters for the classification model of all types
    classification_input_normalize_params = []
    for i in range(10):
        file = open('../data/dynopt_result/feasibility_classification_nn_models/input_mean_std_' + str(i) + '_0.0001_256_0.1.txt', 'r')
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
        file = open('../data/dynopt_result/objective_regression_nn_models/input_mean_std_' + str(i) + '_0.0005_256_0.0.txt', 'r')
        strings = file.readline().strip().split(' ')
        params = np.zeros((2, len(strings) // 2), dtype=float)
        for j in range(len(strings) // 2):
            params[0, j] = float(strings[2 * j])
            params[1, j] = float(strings[2 * j + 1])
        regression_input_normalize_params.append(params)

    # load the denormalize parameters for the regression model of all types
    regression_output_denormalize_params = []
    for i in range(10):
        file = open('../data/dynopt_result/objective_regression_nn_models/output_mean_std_' + str(i) + '_0.0005_256_0.0.txt', 'r')
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

    X_zero = {}
    for i in range(10):
        X_zero[i] = []

    # load sampled transitions
    file = open('../data/ground_truth/transitions_' + environment_type, 'r')
    transitions = pickle.load(file)

    for transition in transitions:
        transition_type = transition['contact_transition_type']
        com = com_combinations[transition_type]

        # the distance between com position and each foot contact should be in [0, 1.1]
        valid_com_indices = np.sum((np.array(transition['normalized_init_l_leg'][:3]) - com[:, :3]) ** 2, axis=1) <= 1.1 ** 2
        valid_com_indices = np.logical_and(np.sum((np.array(transition['normalized_init_r_leg'][:3]) - com[:, :3]) ** 2, axis=1) <= 1.1 ** 2, valid_com_indices)
        
        # the distance between com position and each palm contact (if exists) should be in [0, 0.8]
        if transition['normalized_init_l_arm']:
            valid_com_indices = np.logical_and(np.sum((np.array(transition['normalized_init_l_arm'][:3]) - com[:, :3]) ** 2, axis=1) <= 0.8 ** 2, valid_com_indices)
        if transition['normalized_init_r_arm']:
            valid_com_indices = np.logical_and(np.sum((np.array(transition['normalized_init_r_arm'][:3]) - com[:, :3]) ** 2, axis=1) <= 0.8 ** 2, valid_com_indices)

        X = np.zeros((len(com), len(transition['feature_vector_contact_part']) + 6), dtype=float)
        X[:, 0:-6] = np.tile(np.array(transition['feature_vector_contact_part']), (len(com), 1))
        X[:, -6:] = np.array(com)

        # query the classification model
        prediction = classification_models[transition_type].predict((X - classification_input_normalize_params[transition_type][0]) / classification_input_normalize_params[transition_type][1])
        prediction = prediction.reshape(-1,)
        valid_com_indices = np.logical_and(prediction > 0.5, valid_com_indices)

        # query the regression model
        prediction = regression_models[transition_type].predict((X - regression_input_normalize_params[transition_type][0]) / regression_input_normalize_params[transition_type][1]) * regression_output_denormalize_params[transition_type][1] + regression_output_denormalize_params[transition_type][0]
        prediction = prediction[:, -1]
        # ddyns = prediction[np.argwhere(valid_com_indices == True).reshape(-1,)]

        zero_indices = np.logical_and(prediction < 1e-5, valid_com_indices)
        X_zero[transition_type] += X[np.argwhere(zero_indices == True).reshape(-1,)].tolist()


    with open('../data/zero_ddyn_' + environment_type, 'w') as file:
        pickle.dump(X_zero, file)


if __name__ == "__main__":
    main()
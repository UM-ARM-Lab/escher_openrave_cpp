import pickle, IPython, keras
import numpy as np
from keras.models import load_model
import tensorflow as tf

def main():
    config = tf.ConfigProto(device_count={'GPU':1, 'CPU':3}, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.6))
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    # # load the normalize parameters for the classification model of all types
    # classification_input_normalize_params = []
    # for i in range(10):
    #     file = open('../data/dynopt_result/feasibility_classification_nn_models/input_mean_std_' + str(i) + '_0.0001_256_0.1.txt', 'r')
    #     strings = file.readline().strip().split(' ')
    #     params = np.zeros((2, len(strings) // 2), dtype=float)
    #     for j in range(len(strings) // 2):
    #         params[0, j] = float(strings[2 * j])
    #         params[1, j] = float(strings[2 * j + 1])
    #     classification_input_normalize_params.append(params)

    # # load the classification models of all types
    # classification_models = []
    # for i in range(10):
    #     classification_models.append(load_model('../data/dynopt_result/feasibility_classification_nn_models/nn_model_' + str(i) + '_0.0001_256_0.1.h5'))

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

    for i in range(10):
        file = open('../data/dynopt_result/dataset/dynopt_total_data_' + str(i), 'r')
        data = pickle.load(file)

        ddyn_25 = np.percentile(data[:, -1], 25)
        indices = data[:, -1] < ddyn_25
        examples = data[np.argwhere(indices == True).reshape(-1,)]
        X = examples[:, 1:-7]
        y = examples[:, -1]
        predicted_y = regression_models[i].predict((X - regression_input_normalize_params[i][0]) / regression_input_normalize_params[i][1]) * regression_output_denormalize_params[i][1] + regression_output_denormalize_params[i][0]
        predicted_y = predicted_y[:, -1]
        mean_error = np.mean(np.absolute(predicted_y - y))
        print('\ntransition type: {}'.format(i))
        print('25 percentile: {}'.format(ddyn_25))
        print('mean error for examples below 25 percentile: {}'.format(mean_error))

        ddyn_50 = np.percentile(data[:, -1], 50)
        indices = data[:, -1] < ddyn_50
        examples = data[np.argwhere(indices == True).reshape(-1,)]
        X = examples[:, 1:-7]
        y = examples[:, -1]
        predicted_y = regression_models[i].predict((X - regression_input_normalize_params[i][0]) / regression_input_normalize_params[i][1]) * regression_output_denormalize_params[i][1] + regression_output_denormalize_params[i][0]
        predicted_y = predicted_y[:, -1]
        mean_error = np.mean(np.absolute(predicted_y - y))
        print('50 percentile: {}'.format(ddyn_50))
        print('mean error for examples below 50 percentile: {}'.format(mean_error))






if __name__ == "__main__":
    main()
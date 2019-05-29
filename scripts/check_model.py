import pickle, IPython
import numpy as np
from keras.models import load_model

def main():
    # load sampled COM combinations of all types
    # com_combinations = {}
    # for i in range(10):
    #     file = open('../data/CoM/com_combinations_' + str(i), 'r')
    #     com_combinations[i] = pickle.load(file)

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

    # environ_pose_to_ddyn is a nested dictionary.
    # the first key is environment index. The environments represented by environment indices are saved in the file "environments"
    # the second key is initial pose and final pose
    # the value is a vector of dynamic cost
    # environ_pose_to_ddyn = {}

    # load sampled transitions
    # file = open('../data/transitions', 'r')
    # transitions = pickle.load(file)

    for i in range(10):
        file = open('../data/dynopt_result/dataset/dynopt_total_data_' + str(i), 'r')
        data = pickle.load(file)
        feasible_X = data[:, 1:-7]
        feasible_y = data[:, -1]

        file = open('../data/dynopt_result/dataset/dynopt_infeasible_total_data_' + str(i), 'r')
        infeasible_X = pickle.load(file)[:, 1:-1]

         # query the classification model
        prediction_f = classification_models[i].predict((feasible_X - classification_input_normalize_params[i][0]) / classification_input_normalize_params[i][1])
        prediction_i = classification_models[i].predict((infeasible_X - classification_input_normalize_params[i][0]) / classification_input_normalize_params[i][1])
        accuracy = (np.sum(prediction_f.reshape(-1,) > 0.5) + np.sum(prediction_i.reshape(-1,) < 0.5)) * 100.0 / (feasible_X.shape[0] + infeasible_X.shape[0])

        indices = np.argwhere(feasible_y < 50000).reshape(-1,)
        feasible_X = feasible_X[indices]
        feasible_y = feasible_y[indices]
        prediction = regression_models[i].predict((feasible_X - regression_input_normalize_params[i][0]) / regression_input_normalize_params[i][1]) * regression_output_denormalize_params[i][1] + regression_output_denormalize_params[i][0]
        debug_ddyns = prediction[:, -1] 

        print('transition type: {}'.format(i))
        print('accuracy of the classification model: {:3.1f}%'.format(accuracy))

        print('percentiles of ground truth ddyns')
        print('25%: {:.2f}, 50%: {:.2f}, 75%: {:.2f}, mean: {:.2f}'.format(
            np.percentile(feasible_y, 25), np.percentile(feasible_y, 50), np.percentile(feasible_y, 75), np.mean(feasible_y)))
        print('percentiles of predicted ddyns')
        print('25%: {:.2f}, 50%: {:.2f}, 75%: {:.2f}, mean: {:.2f}'.format(
            np.percentile(debug_ddyns, 25), np.percentile(debug_ddyns, 50), np.percentile(debug_ddyns, 75), np.mean(debug_ddyns)))
        
    # IPython.embed()
    # file = open('../data/environ_pose_to_ddyn', 'w')
    # pickle.dump(environ_pose_to_ddyn, file)

    


if __name__ == "__main__":
    main()
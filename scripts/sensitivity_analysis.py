import pickle, random, os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

SAMPLE_SIZE = 5000

def main():
    for i in range(10):
        with open('../data/dynopt_result/dataset/dynopt_total_data_' + str(i), 'r') as file:
            data = pickle.load(file)
        X = data[:, 1:-7]
        ddyns = data[:, -1]

        # # load the normalize parameters for the classification model
        # with open('../data/dynopt_result/feasibility_classification_nn_models/input_mean_std_' + str(i) + '_0.0001_256_0.1.txt', 'r') as file:
        #     strings = file.readline().strip().split(' ')
        # classification_input_normalize_params = np.zeros((2, len(strings) // 2), dtype=float)
        # for j in range(len(strings) // 2):
        #     classification_input_normalize_params[0, j] = float(strings[2 * j])
        #     classification_input_normalize_params[1, j] = float(strings[2 * j + 1])

        # # load the classification model
        # classification_model = load_model('../data/dynopt_result/feasibility_classification_nn_models/nn_model_' + str(i) + '_0.0001_256_0.1.h5')

        # load the normalize parameters for the regression model
        with open('../data/dynopt_result/objective_regression_nn_models/input_mean_std_' + str(i) + '_0.0005_256_0.0.txt', 'r') as file:
            strings = file.readline().strip().split(' ')
        regression_input_normalize_params = np.zeros((2, len(strings) // 2), dtype=float)
        for j in range(len(strings) // 2):
            regression_input_normalize_params[0, j] = float(strings[2 * j])
            regression_input_normalize_params[1, j] = float(strings[2 * j + 1])

        # load the denormalize parameters for the regression model
        with open('../data/dynopt_result/objective_regression_nn_models/output_mean_std_' + str(i) + '_0.0005_256_0.0.txt', 'r') as file:
            strings = file.readline().strip().split(' ')
        regression_output_denormalize_params = np.zeros((2, len(strings) // 2), dtype=float)
        for j in range(len(strings) // 2):
            regression_output_denormalize_params[0, j] = float(strings[2 * j])
            regression_output_denormalize_params[1, j] = float(strings[2 * j + 1])

        # load the regression model
        regression_model = load_model('../data/dynopt_result/objective_regression_nn_models/nn_model_' + str(i) + '_0.0005_256_0.0.h5')

        random.seed(20190530)
        indices = random.sample(range(X.shape[0]), SAMPLE_SIZE)
        sampled_X = X[indices]

        variables_range = np.percentile(X, 99, axis=0) - np.percentile(X, 1, axis=0)
        variables_perturbation = variables_range / 98.0

        range_ddyn = np.percentile(ddyns, 99) - np.percentile(ddyns, 1)

        for j in range(X.shape[1]):
            predicted_ddyn = regression_model.predict((sampled_X - regression_input_normalize_params[0]) / regression_input_normalize_params[1]) * regression_output_denormalize_params[1] + regression_output_denormalize_params[0]
            predicted_ddyn = predicted_ddyn[:, -1]

            perturbed_X = np.copy(sampled_X)
            perturbed_X[:, j] += variables_perturbation[j]
            predicted_perturbed_ddyn = regression_model.predict((perturbed_X - regression_input_normalize_params[0]) / regression_input_normalize_params[1]) * regression_output_denormalize_params[1] + regression_output_denormalize_params[0]
            predicted_perturbed_ddyn = predicted_perturbed_ddyn[:, -1]

            diff = (predicted_perturbed_ddyn - predicted_ddyn) * 100.0 / range_ddyn
            np.clip(diff, -4, 2, out=diff)
            plt.figure(1000 * i + j)
            plt.hist(diff, bins=np.arange(-3.9, 2.1, 0.1))
            plt.title('transition type {}, dimension {}'.format(i, j))
            plt.savefig('../data/figures/t{}_d{}.png'.format(i, j))


if __name__ == "__main__":
    main()
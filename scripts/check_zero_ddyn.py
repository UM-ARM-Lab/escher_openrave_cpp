import pickle, IPython, math, sys, getopt, random
# import keras
import numpy as np
# from keras.models import load_model
from sklearn.neighbors import BallTree

SAMPLE_SIZE = 10000

def main():
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

    # # load the normalize parameters for the regression model of all types
    # regression_input_normalize_params = []
    # for i in range(10):
    #     file = open('../data/dynopt_result/objective_regression_nn_models/input_mean_std_' + str(i) + '_0.0005_256_0.0.txt', 'r')
    #     strings = file.readline().strip().split(' ')
    #     params = np.zeros((2, len(strings) // 2), dtype=float)
    #     for j in range(len(strings) // 2):
    #         params[0, j] = float(strings[2 * j])
    #         params[1, j] = float(strings[2 * j + 1])
    #     regression_input_normalize_params.append(params)

    # # load the denormalize parameters for the regression model of all types
    # regression_output_denormalize_params = []
    # for i in range(10):
    #     file = open('../data/dynopt_result/objective_regression_nn_models/output_mean_std_' + str(i) + '_0.0005_256_0.0.txt', 'r')
    #     strings = file.readline().strip().split(' ')
    #     params = np.zeros((2, len(strings) // 2), dtype=float)
    #     for j in range(len(strings) // 2):
    #         params[0, j] = float(strings[2 * j])
    #         params[1, j] = float(strings[2 * j + 1])
    #     regression_output_denormalize_params.append(params)

    # # load the regression models of all types
    # regression_models = []
    # for i in range(10):
    #     regression_models.append(load_model('../data/dynopt_result/objective_regression_nn_models/nn_model_' + str(i) + '_0.0005_256_0.0.h5'))

    file = open('../data/zero_ddyn_0_tiny', 'r')
    data = pickle.load(file)
    
    for transition_type in range(10):
        X = np.array(data[transition_type])
        if X.shape[0] != 0:
            # # check whether feasible
            # prediction = classification_models[transition_type].predict((X - classification_input_normalize_params[transition_type][0]) / classification_input_normalize_params[transition_type][1]).reshape(-1,)
            # assert(np.sum(prediction > 0.5) == prediction.shape[0])

            # # check dynamic cost
            # prediction = regression_models[transition_type].predict((X - regression_input_normalize_params[transition_type][0]) / regression_input_normalize_params[transition_type][1]) * regression_output_denormalize_params[transition_type][1] + regression_output_denormalize_params[transition_type][0]
            # prediction = prediction[:, -1]
            # assert(np.sum(prediction < 1e-5) == prediction.shape[0])
        
            file = open('../data/dynopt_result/dataset/dynopt_total_data_' + str(transition_type), 'r')
            original_data = pickle.load(file)
            
            original_X = original_data[:, 1:-7]

            mean = np.mean(original_X, axis=0)
            print(mean)
            std = np.std(original_X, axis=0)
            normalized_original_X = (original_X - mean) / std

            tree = BallTree(normalized_original_X)

            # original data
            # random.seed(20190602)
            # indices = random.sample(range(normalized_original_X.shape[0]), SAMPLE_SIZE)
            indices = np.argwhere(original_data[:, -1] < 30.0).reshape(-1,)
            dist, _ = tree.query(normalized_original_X[indices], k=10)
            avg_dist = np.mean(dist)

            # zero dynamic cost data
            normalized_X = (X - mean) / std
            dist, _ = tree.query(normalized_X, k=10)
            zero_avg_dist = np.mean(dist)

            print('original small dynamic cost X:')
            print(np.mean(normalized_original_X[indices], axis=0))
            print(avg_dist)
            print('zero dynamic cost X:')
            print(np.mean(normalized_X, axis=0))
            print(zero_avg_dist)







            
        


    

if __name__ == "__main__":
    main()


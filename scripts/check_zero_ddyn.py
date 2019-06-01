import pickle, IPython, math, sys, getopt
# import keras
import numpy as np
# from keras.models import load_model
from sklearn.neighbors import BallTree

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

    file = open('../data/zero_ddyn_0', 'r')
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

            mean = np.mean(original_data, axis=0)
            std = np.std(original_data, axis=0)
            IPython.embed()
            normalized_original_data = (original_data - mean) / std

            tree = BallTree(normalized_original_data)
            dist, indices = tree.query(normalized_original_data[0:5], k=10)
            IPython.embed()




            
        


    

if __name__ == "__main__":
    main()


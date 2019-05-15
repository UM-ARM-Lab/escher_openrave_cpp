import pickle
import numpy as np
from keras.models import load_model

def main():

    for i in range(2):
        transitions_handle = open('../data/transitions/transitions_type_' + str(i), 'r')
        transitions = pickle.load(transitions_handle)
        print(transitions[0])

        # com_handle = open('../data/COM/com_combinations_' + str(i), 'r')
        # com_combinations = pickle.load(com_handle)

        # prepare for the classification model
        # feasibility_input_normalize_param_handle = open('../data/dynopt_result/feasibility_classification_nn_models/input_mean_std_' + str(i) + '_0.0001_256_0.1.txt', 'r')
        # line = feasibility_input_normalize_param_handle.readline().strip()
        # feasibility_input_normalize_param_str = line.split(' ')
        # feasibility_input_normalize_param_float = [float(s) for s in feasibility_input_normalize_param_str]
        # feasibility_input_normalize_param = np.array(feasibility_input_normalize_param_float)

        # classification_model = load_model('../data/dynopt_result/feasibility_classification_nn_models/nn_model_' + str(i) + '_0.0001_256_0.1.h5')
        
        # prepare for the regression model
        # regression_input_normalize_param_handle = open('../data/dynopt_result/objective_regression_nn_models/input_mean_std_' + str(i) + '_0.0005_256_0.0.txt', 'r')
        # line = regression_input_normalize_param_handle.readline().strip()
        # regression_input_normalize_param_str = line.split(' ')
        # regression_input_normalize_param_float = [float(s) for s in regression_input_normalize_param_str]
        # regression_input_normalize_param = np.array(regression_input_normalize_param_float)
        
        # regression_output_normalize_param_handle = open('../data/dynopt_result/objective_regression_nn_models/output_mean_std_' + str(i) + '_0.0005_256_0.0.txt', 'r')
        # line = regression_output_normalize_param_handle.readline().strip()
        # regression_output_normalize_param_str = line.split(' ')
        # regression_output_normalize_param_float = [float(s) for s in regression_output_normalize_param_str]
        # regression_output_normalize_param = np.array(regression_output_normalize_param_float)

        # regression_model = load_model('../data/dynopt_result/objective_regression_nn_models/nn_model_' + str(i) + '_0.0005_256_0.0.h5')

        #for transition in transitions:


        


        


if __name__ == "__main__":
    main()
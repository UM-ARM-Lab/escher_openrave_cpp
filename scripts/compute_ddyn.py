import pickle, IPython
import numpy as np
from keras.models import load_model

def main():
    for i in range(10):
        transitions_file = open('../data/transitions/transitions_type_' + str(i), 'r')
        transitions = pickle.load(transitions_file)

        com_file = open('../data/COM/com_combinations_' + str(i), 'r')
        com_combinations = pickle.load(com_file)

        # prepare for the classification model
        classification_input_normalize_param_file = open('../data/dynopt_result/feasibility_classification_nn_models/input_mean_std_' + str(i) + '_0.0001_256_0.1.txt', 'r')
        line = classification_input_normalize_param_file.readline().strip()
        classification_input_normalize_param_str = line.split(' ')
        classification_input_normalize_param = np.zeros((2, len(classification_input_normalize_param_str) // 2), dtype=float)
        for j in range(len(classification_input_normalize_param_str) // 2):
            classification_input_normalize_param[0, j] = float(classification_input_normalize_param_str[2 * j])
            classification_input_normalize_param[1, j] = float(classification_input_normalize_param_str[2 * j + 1])

        classification_model = load_model('../data/dynopt_result/feasibility_classification_nn_models/nn_model_' + str(i) + '_0.0001_256_0.1.h5')
        
        # prepare for the regression model
        regression_input_normalize_param_file = open('../data/dynopt_result/objective_regression_nn_models/input_mean_std_' + str(i) + '_0.0005_256_0.0.txt', 'r')
        line = regression_input_normalize_param_file.readline().strip()
        regression_input_normalize_param_str = line.split(' ')
        regression_input_normalize_param = np.zeros((2, len(regression_input_normalize_param_str) // 2), dtype=float)
        for j in range(len(regression_input_normalize_param_str) // 2):
            regression_input_normalize_param[0, j] = float(regression_input_normalize_param_str[2 * j])
            regression_input_normalize_param[1, j] = float(regression_input_normalize_param_str[2 * j + 1])

        regression_output_denormalize_param_file = open('../data/dynopt_result/objective_regression_nn_models/output_mean_std_' + str(i) + '_0.0005_256_0.0.txt', 'r')
        line = regression_output_denormalize_param_file.readline().strip()
        regression_output_denormalize_param_str = line.split(' ')
        regression_output_denormalize_param = np.zeros((2, len(regression_output_denormalize_param_str) // 2), dtype=float)
        for j in range(len(regression_output_denormalize_param_str) // 2):
            regression_output_denormalize_param[0, j] = float(regression_output_denormalize_param_str[2 * j])
            regression_output_denormalize_param[1, j] = float(regression_output_denormalize_param_str[2 * j + 1])

        regression_model = load_model('../data/dynopt_result/objective_regression_nn_models/nn_model_' + str(i) + '_0.0005_256_0.0.h5')

        for transition in transitions:
            # query the classification model
            X = np.zeros((len(com_combinations), len(classification_input_normalize_param_str) // 2), dtype=float)
            X[:, 0:-6] = np.tile(np.array(transition['feature_vector_contact_part']), (len(com_combinations), 1))
            X[:, -6:] = np.array(com_combinations)
            X = X - classification_input_normalize_param[0]
            X = X / classification_input_normalize_param[1]
            prediction = classification_model.predict(X)
            prediction = (prediction > 0.5).astype(int)

            # query the regression model
            X = np.zeros((np.sum(prediction), len(classification_input_normalize_param_str) // 2), dtype=float)
            X[:, 0:-6] = np.tile(np.array(transition['feature_vector_contact_part']), (len(com_combinations), 1))
            k = 0
            for j in range(prediction.shape[0]):
                if prediction[j] == 1:
                    X[k, -6:] = com_combinations[j]
                    k += 1
            ddyn_prediction = regression_model.predict(X)[:, -1]
            print(ddyn_prediction)
            

            

            assert(False)
       


            



        


        


if __name__ == "__main__":
    main()
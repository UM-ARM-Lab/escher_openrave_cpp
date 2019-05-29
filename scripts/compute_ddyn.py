import pickle, IPython
import numpy as np
from keras.models import load_model

def main():
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

    # environ_pose_to_ddyn is a nested dictionary.
    # the first key is environment index. The environments represented by environment indices are saved in the file "environments"
    # the second key is initial pose and final pose
    # the value is a vector of dynamic cost
    environ_pose_to_ddyn = {}

    # load sampled transitions
    file = open('../data/transitions', 'r')
    transitions = pickle.load(file)

    debug_all_sampled_com = {}
    debug_com_after_position_check = {}
    debug_com_after_feasibility_check = {}
    debug_ddyns = {}
    for i in range(10):
        debug_all_sampled_com[i] = 0
        debug_com_after_position_check[i] = 0
        debug_com_after_feasibility_check[i] = 0
        debug_ddyns[i] = []

    for transition in transitions:
        transition_type = transition['contact_transition_type']
        com = com_combinations[transition_type]

        debug_all_sampled_com[transition_type] += com.shape[0]

        # the distance between com position and each foot contact should be in [0, 1.1]
        valid_com_indices = np.sum((np.array(transition['normalized_init_l_leg'][:3]) - com[:, :3]) ** 2, axis=1) <= 1.1 ** 2
        valid_com_indices = np.logical_and(np.sum((np.array(transition['normalized_init_r_leg'][:3]) - com[:, :3]) ** 2, axis=1) <= 1.1 ** 2, valid_com_indices)
        
        # the distance between com position and each palm contact (if exists) should be in [0, 0.8]
        if transition['normalized_init_l_arm']:
            valid_com_indices = np.logical_and(np.sum((np.array(transition['normalized_init_l_arm'][:3]) - com[:, :3]) ** 2, axis=1) <= 0.8 ** 2, valid_com_indices)
        if transition['normalized_init_r_arm']:
            valid_com_indices = np.logical_and(np.sum((np.array(transition['normalized_init_r_arm'][:3]) - com[:, :3]) ** 2, axis=1) <= 0.8 ** 2, valid_com_indices)

        debug_com_after_position_check[transition_type] += np.sum(valid_com_indices)

        X = np.zeros((len(com), len(transition['feature_vector_contact_part']) + 6), dtype=float)
        X[:, 0:-6] = np.tile(np.array(transition['feature_vector_contact_part']), (len(com), 1))
        X[:, -6:] = np.array(com)

        # query the classification model
        prediction = classification_models[transition_type].predict((X - classification_input_normalize_params[transition_type][0]) / classification_input_normalize_params[transition_type][1])
        prediction = prediction.reshape(-1,)
        valid_com_indices = np.logical_and(prediction > 0.5, valid_com_indices)

        debug_com_after_feasibility_check[transition_type] += np.sum(valid_com_indices)

        # query the regression model
        prediction = regression_models[transition_type].predict((X - regression_input_normalize_params[transition_type][0]) / regression_input_normalize_params[transition_type][1]) * regression_output_denormalize_params[transition_type][1] + regression_output_denormalize_params[transition_type][0]
        prediction = prediction[:, -1]
        ddyns = prediction[np.argwhere(valid_com_indices == True).reshape(-1,)]

        if ddyns.shape[0] != 0:
            ddyns = ddyns.tolist()

            debug_ddyns[transition_type] += ddyns

            environ = transition['environment_index']

            p1 = transition['p1']
            pose1 = []
            pose1.append(round(p1[0], 2))
            pose1.append(round(p1[1], 2))
            pose1.append(round(p1[5], 0))
            p2 = transition['p2']
            pose2 = []
            pose2.append(round(p2[0], 2))
            pose2.append(round(p2[1], 2))
            pose2.append(round(p2[5], 0))
            pose = tuple(pose1 + pose2)

            if environ not in environ_pose_to_ddyn:
                environ_pose_to_ddyn[environ] = {}

            if pose not in environ_pose_to_ddyn[environ]:
                environ_pose_to_ddyn[environ][pose] = ddyns
            else:
                environ_pose_to_ddyn[environ][pose] += ddyns

    for i in range(10):
        print('\ntransition type: {}'.format(i))
        print('all sampled com: {}'.format(debug_all_sampled_com[i]))
        print('com after position check: {}, {:6.2f} percent of all coms are valid'.format(debug_com_after_position_check[i], debug_com_after_position_check[i] * 100.0 / debug_all_sampled_com[i]))
        print('com after feasibility check: {}, {:6.2f} percent of all coms are feasible'.format(debug_com_after_feasibility_check[i], debug_com_after_feasibility_check[i] * 100.0 / debug_all_sampled_com[i]))
        file = open('../data/dynopt_result/dataset/dynopt_total_data_' + str(i), 'r')
        data = pickle.load(file)
        all_ddyn = data[:, -1]
        print('percentiles of all ddyns in data')
        print('min: {}, 25%: {}, 50%: {}, 75%: {}, max: {}'.format(
            np.min(all_ddyn), np.percentile(all_ddyn, 25), np.percentile(all_ddyn, 50), np.percentile(all_ddyn, 75), np.max(all_ddyn)))
        if len(debug_ddyns[i]) != 0:
            debug_ddyns[i] = np.array(debug_ddyns[i])
            print('percentiles of all ddyns in this program')
            print('min: {}, 25%: {}, 50%: {}, 75%: {}, max: {}'.format(
                np.min(debug_ddyns[i]), np.percentile(debug_ddyns[i], 25), np.percentile(debug_ddyns[i], 50), np.percentile(debug_ddyns[i], 75), np.max(debug_ddyns[i])))
        
    # IPython.embed()
    # file = open('../data/environ_pose_to_ddyn', 'w')
    # pickle.dump(environ_pose_to_ddyn, file)

    


if __name__ == "__main__":
    main()
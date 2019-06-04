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

    # debug_all_sampled_com = {}
    # debug_com_after_position_check = {}
    # debug_com_after_feasibility_check = {}
    # debug_ddyns = {}
    # debug_distance = {}
    # for i in range(10):
    #     debug_all_sampled_com[i] = 0
    #     debug_com_after_position_check[i] = 0
    #     debug_com_after_feasibility_check[i] = 0
    #     debug_ddyns[i] = []
    #     debug_distance[i] = []
    
    # environ_pose_to_ddyn is a nested dictionary.
    # the first key is environment index in each environment type;
    # the second key is initial pose and final pose;
    # the value is a vector of dynamic cost.
    environ_pose_to_ddyn = {}

    # load sampled transitions
    file = open('../data/ground_truth/transitions_' + environment_type, 'r')
    transitions = pickle.load(file)[:10]

    for transition in transitions:
        transition_type = transition['contact_transition_type']
        com = com_combinations[transition_type]

        # debug_all_sampled_com[transition_type] += com.shape[0]

        # # the distance between com position and each foot contact should be in [0, 1.1]
        # valid_com_indices = np.sum((np.array(transition['normalized_init_l_leg'][:3]) - com[:, :3]) ** 2, axis=1) <= 1.1 ** 2
        # valid_com_indices = np.logical_and(np.sum((np.array(transition['normalized_init_r_leg'][:3]) - com[:, :3]) ** 2, axis=1) <= 1.1 ** 2, valid_com_indices)
        
        # # the distance between com position and each palm contact (if exists) should be in [0, 0.8]
        # if transition['normalized_init_l_arm']:
        #     valid_com_indices = np.logical_and(np.sum((np.array(transition['normalized_init_l_arm'][:3]) - com[:, :3]) ** 2, axis=1) <= 0.8 ** 2, valid_com_indices)
        # if transition['normalized_init_r_arm']:
        #     valid_com_indices = np.logical_and(np.sum((np.array(transition['normalized_init_r_arm'][:3]) - com[:, :3]) ** 2, axis=1) <= 0.8 ** 2, valid_com_indices)

        # debug_com_after_position_check[transition_type] += np.sum(valid_com_indices)

        X = np.zeros((len(com), len(transition['feature_vector_contact_part']) + 6), dtype=float)
        X[:, 0:-6] = np.tile(np.array(transition['feature_vector_contact_part']), (len(com), 1))
        X[:, -6:] = np.array(com)

        # # query the classification model
        # prediction = classification_models[transition_type].predict((X - classification_input_normalize_params[transition_type][0]) / classification_input_normalize_params[transition_type][1])
        # prediction = prediction.reshape(-1,)
        # valid_com_indices = np.logical_and(prediction > 0.5, valid_com_indices)

        # debug_com_after_feasibility_check[transition_type] += np.sum(valid_com_indices)

        # query the regression model
        prediction = regression_models[transition_type].predict((X - regression_input_normalize_params[transition_type][0]) / regression_input_normalize_params[transition_type][1]) * regression_output_denormalize_params[transition_type][1] + regression_output_denormalize_params[transition_type][0]
        prediction = prediction[:, -1]
        # ddyns = prediction[np.argwhere(valid_com_indices == True).reshape(-1,)]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ddyns = prediction

        if ddyns.shape[0] != 0:
            ddyns = ddyns.tolist()

            # debug_ddyns[transition_type] += ddyns

            environment_index = transition['environment_index']

            p1 = transition['p1']
            p2 = transition['p2']
            pose = tuple(p1 + p2)

            # distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) * GRID_RESOLUTION
            # debug_distance[transition_type].append(distance)

            if environment_index not in environ_pose_to_ddyn:
                environ_pose_to_ddyn[environment_index] = {}

            if pose not in environ_pose_to_ddyn[environment_index]:
                environ_pose_to_ddyn[environment_index][pose] = ddyns
            else:
                environ_pose_to_ddyn[environment_index][pose] += ddyns

    # for i in range(10):
    #     if debug_all_sampled_com[i] != 0:
    #         print('\ntransition type: {}'.format(i))
    #         print('all sampled com: {}'.format(debug_all_sampled_com[i]))
    #         print('com after position check: {}, {:4.2f} percent of all coms are valid'.format(debug_com_after_position_check[i], debug_com_after_position_check[i] * 100.0 / debug_all_sampled_com[i]))
    #         print('com after feasibility check: {}, {:4.2f} percent of all coms are feasible'.format(debug_com_after_feasibility_check[i], debug_com_after_feasibility_check[i] * 100.0 / debug_all_sampled_com[i]))
    #         file = open('../data/dynopt_result/dataset/dynopt_total_data_' + str(i), 'r')
    #         data = pickle.load(file)
    #         all_ddyn = data[:, -1]
    #         print('percentiles of all ddyns in data')
    #         print('min: {:4.2f}, 25%: {:4.2f}, 50%: {:4.2f}, 75%: {:4.2f}, max: {:4.2f}'.format(
    #             np.min(all_ddyn), np.percentile(all_ddyn, 25), np.percentile(all_ddyn, 50), np.percentile(all_ddyn, 75), np.max(all_ddyn)))
    #         if len(debug_ddyns[i]) != 0:
    #             debug_ddyns[i] = np.array(debug_ddyns[i])
    #             print('percentiles of all ddyns in this program')
    #             print('min: {:4.2f}, 25%: {:4.2f}, 50%: {:4.2f}, 75%: {:4.2f}, max: {:4.2f}'.format(
    #                 np.min(debug_ddyns[i]), np.percentile(debug_ddyns[i], 25), np.percentile(debug_ddyns[i], 50), np.percentile(debug_ddyns[i], 75), np.max(debug_ddyns[i])))
    #             print('average distance between initial status and final status: {:4.2f}'.format(np.mean(np.array(debug_distance[i]))))

    # IPython.embed()
    with open('../data/test_environ_pose_to_ddyn_' + environment_type, 'w') as file:
        pickle.dump(environ_pose_to_ddyn, file)


if __name__ == "__main__":
    main()
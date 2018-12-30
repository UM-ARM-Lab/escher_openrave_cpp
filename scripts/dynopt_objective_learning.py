# imports
from sklearn import svm
from sklearn import multioutput

import tensorflow as tf
from tensorflow import keras

import sys
import numpy as np
import IPython
import pickle
import matplotlib.pyplot as plt
import os
import scipy
import random

data_index_map = {}
data_label = []
feature_label = []
contact_status_label = ['motion_code']

prediction_label = []

contact_status_indices_list = []
feature_indices_list = []
prediction_indices_list = []

training_data_path = '../data/dynopt_result/'
model_path = '../data/dynopt_result/objective_regression_nn_models/'

# put here as reference
# enum ContactTransitionCode
# {
#     FEET_ONLY_MOVE_FOOT,                // 0
#     FEET_ONLY_ADD_HAND,                 // 1
#     FEET_AND_ONE_HAND_MOVE_INNER_FOOT,  // 2
#     FEET_AND_ONE_HAND_MOVE_OUTER_FOOT,  // 3
#     FEET_AND_ONE_HAND_BREAK_HAND,       // 4
#     FEET_AND_ONE_HAND_MOVE_HAND,        // 5
#     FEET_AND_ONE_HAND_ADD_HAND,         // 6
#     FEET_AND_TWO_HANDS_MOVE_FOOT,        // 7
#     FEET_AND_TWO_HANDS_BREAK_HAND,       // 8
#     FEET_AND_TWO_HANDS_MOVE_HAND         // 9
# };

# enum ZeroStepCaptureCode
# {
#     ONE_FOOT,                   // 0
#     TWO_FEET,                   // 1
#     ONE_FOOT_AND_INNER_HAND,    // 2
#     ONE_FOOT_AND_OUTER_HAND,    // 3
#     ONE_FOOT_AND_TWO_HANDS,     // 4
#     FEET_AND_ONE_HAND           // 5
# };

# enum OneStepCaptureCode
# {
#     ONE_FOOT_ADD_FOOT,                  // 0
#     ONE_FOOT_ADD_INNER_HAND,            // 1
#     ONE_FOOT_ADD_OUTER_HAND,            // 2
#     TWO_FEET_ADD_HAND,                  // 3
#     ONE_FOOT_AND_INNER_HAND_ADD_FOOT,   // 4
#     ONE_FOOT_AND_INNER_HAND_ADD_HAND,   // 5
#     ONE_FOOT_AND_OUTER_HAND_ADD_FOOT,   // 6
#     ONE_FOOT_AND_OUTER_HAND_ADD_HAND,   // 7
#     ONE_FOOT_AND_TWO_HANDS_ADD_FOOT,    // 8
#     TWO_FEET_AND_ONE_HAND_ADD_HAND      // 9
# };

contact_transition_code_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
contact_transition_code_manip_map = {0: ['prev_lf', 'prev_rf', 'curr_lf'],
                                     1: ['prev_lf', 'prev_rf', 'curr_lh'],
                                     2: ['prev_lf', 'prev_rf', 'prev_lh', 'curr_lf'],
                                     3: ['prev_lf', 'prev_rf', 'prev_rh', 'curr_lf'],
                                     4: ['prev_lf', 'prev_rf', 'prev_lh'],
                                     5: ['prev_lf', 'prev_rf', 'prev_lh', 'curr_lh'],
                                     6: ['prev_lf', 'prev_rf', 'prev_rh', 'curr_lh'],
                                     7: ['prev_lf', 'prev_rf', 'prev_lh', 'prev_rh', 'curr_lf'],
                                     8: ['prev_lf', 'prev_rf', 'prev_lh', 'prev_rh'],
                                     9: ['prev_lf', 'prev_rf', 'prev_lh', 'prev_rh', 'curr_lh']}

zero_step_capture_code_list = [0, 1, 2, 3, 4, 5]
zero_step_capture_code_manip_map = {0: ['curr_rf'],
                                    1: ['curr_lf', 'curr_rf'],
                                    2: ['curr_rf', 'curr_rh'],
                                    3: ['curr_rf', 'curr_lh'],
                                    4: ['curr_rf', 'curr_lh', 'curr_rh'],
                                    5: ['curr_lf', 'curr_rf', 'curr_rh']}

one_step_capture_code_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
one_step_capture_code_manip_map = {0: ['prev_rf', 'curr_lf'],
                                   1: ['prev_lf', 'curr_lh'],
                                   2: ['prev_rf', 'curr_lh'],
                                   3: ['prev_lf', 'prev_rf', 'curr_lh'],
                                   4: ['prev_rf', 'prev_rh', 'curr_lf'],
                                   5: ['prev_rf', 'prev_rh', 'curr_lh'],
                                   6: ['prev_rf', 'prev_lh', 'curr_lf'],
                                   7: ['prev_lf', 'prev_rh', 'curr_lh'],
                                   8: ['prev_rf', 'prev_lh', 'prev_rh', 'curr_lf'],
                                   9: ['prev_lf', 'prev_rf', 'prev_rh', 'curr_lh']}

motion_code_list = []
motion_code_manip_map = {}
motion_type_string = ''

prev_move_manip_list = ['0', '1', '2', '3'] # left_leg, right_leg, left_arm, right_arm

total_data_dict = {}

start_index = 0
end_index = 100

def plot_history(history):

    try:
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.plot(history.epoch, np.array(history.history['loss']), label='MSE Train Loss')
        plt.plot(history.epoch, np.array(history.history['val_loss']), label='MSE Val loss')
        plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='MAE Train Loss')
        plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label='MAE Val loss')
        plt.legend()
        # plt.ylim([0,5])
    except Exception:
        print 'Error in plot_history. Skip it.'

def construct_data_label(desired_motion_code, motion_type):
    global prediction_label

    se3_list = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

    data_label.append('motion_code')

    # prev_lf and prev_rf are guaranteed
    for manip in motion_code_manip_map[desired_motion_code]:
        for j in range(6):
            feature_label.append(manip + '_' + se3_list[j])
            data_label.append(manip + '_' + se3_list[j])

    feature_label.extend(['start_com_x', 'start_com_y', 'start_com_z',
                          'start_com_dot_x', 'start_com_dot_y', 'start_com_dot_z'])

    # data_label.extend(['start_com_x', 'start_com_y', 'start_com_z',
    #                    'start_com_dot_x', 'start_com_dot_y', 'start_com_dot_z',
    #                    'final_com_x', 'final_com_y', 'final_com_z',
    #                    'final_com_dot_x', 'final_com_dot_y', 'final_com_dot_z',
    #                    'objective_value'])

    if motion_type == 'contact_transition' or motion_type == 'one_step_capture':
        data_label.extend(['start_com_x', 'start_com_y', 'start_com_z',
                        'start_com_dot_x', 'start_com_dot_y', 'start_com_dot_z',
                        'final_com_x', 'final_com_y', 'final_com_z',
                        'final_com_dot_x', 'final_com_dot_y', 'final_com_dot_z',
                        'objective_value', 'timing'])

        prediction_label = ['final_com_x', 'final_com_y', 'final_com_z',
                            'final_com_dot_x', 'final_com_dot_y', 'final_com_dot_z',
                            'objective_value', 'timing']

    elif motion_type == 'zero_step_capture':
        data_label.extend(['start_com_x', 'start_com_y', 'start_com_z',
                           'start_com_dot_x', 'start_com_dot_y', 'start_com_dot_z',
                           'objective_value', 'timing'])

        prediction_label = ['objective_value', 'timing']

def combine_data():
    total_data_dict = {}

    try:
        file_stream = open(training_data_path + motion_type_string + '_dynopt_total_data_dict','r')
        total_data_dict = pickle.load(file_stream)
        file_stream.close()
    except Exception:
        total_data_dict = {}

    i = 0

    while(True):

        try:
            file_stream = open(training_data_path + motion_type_string + '_dynopt_total_data_dict_' + str(i*100) + '_' + str((i+1)*100),'r')
            data_dict = pickle.load(file_stream)
            file_stream.close()
        except Exception:
            break

        for motion_code, data in data_dict.iteritems():
            if motion_code in total_data_dict:
                total_data_dict[motion_code] = np.vstack((total_data_dict[motion_code], data_dict[motion_code]))
            else:
                total_data_dict[motion_code] = np.copy(data_dict[motion_code])

        print motion_type_string + '_dynopt_total_data_dict_' + str(i*100) + '_' + str((i+1)*100)
        i += 1

    for motion_code in motion_code_list:
        if motion_code in total_data_dict:
            print 'Contact transition code: ', motion_code, ', Original dataset size: ', total_data_dict[motion_code].shape[0]
            print 'Find unique entries of the total dataset'
            total_data_dict[motion_code] = np.unique(total_data_dict[motion_code], axis=0)
            print 'Unique dataset size: ', total_data_dict[motion_code].shape[0]

    file_stream = open(training_data_path + motion_type_string + '_dynopt_total_data_dict','w')
    pickle.dump(total_data_dict, file_stream)
    file_stream.close()

    for motion_code, data in total_data_dict.iteritems():
        file_stream = open(training_data_path + motion_type_string + '_dynopt_total_data_' + str(motion_code),'w')
        pickle.dump(data, file_stream)
        file_stream.close()


def evaluate_data_complexity(data, data_mean_std):

    feature_data = data[:,feature_indices_list]
    feature_data_mean_std = data_mean_std[:,feature_indices_list]
    normalized_feature_data = normalize_data(feature_data, feature_data_mean_std)

    tree = scipy.spatial.KDTree(normalized_feature_data, leafsize=int(normalized_feature_data.shape[0]/10))

    sample_num = 100
    max_sample_radius = 20
    radius_range = range(1,max_sample_radius+1)
    sample_std_mean_by_radius = np.zeros(max_sample_radius)
    sample_std_std_by_radius = np.zeros(max_sample_radius)
    counter = 0

    for radius in radius_range:
        print 'radius: ', radius
        sample_std = np.zeros(sample_num)
        for sample in range(sample_num):
            sample_index = random.randint(0, normalized_feature_data.shape[0]-1)

            sample_std[sample] = np.std(data[tree.query_ball_point(normalized_feature_data[sample_index], radius), data_index_map['objective_value']])

        sample_std_mean_by_radius[counter] = np.mean(sample_std)
        sample_std_std_by_radius[counter] = np.std(sample_std)
        counter += 1

    plt.errorbar(radius_range, sample_std_mean_by_radius, sample_std_std_by_radius)
    plt.show()

    IPython.embed()

def nearest_neighbor_test(training_data, testing_data, data_mean_std):

    training_feature_data = training_data[:,feature_indices_list]
    training_ground_truth_data = training_data[:,prediction_indices_list]
    training_feature_data_mean_std = data_mean_std[:,feature_indices_list]
    normalized_training_feature_data = normalize_data(training_feature_data, training_feature_data_mean_std)

    tree = scipy.spatial.KDTree(normalized_training_feature_data, leafsize=int(normalized_training_feature_data.shape[0]/10))

    testing_feature_data = testing_data[:,feature_indices_list]
    testing_ground_truth_data = testing_data[:,prediction_indices_list]
    normalized_testing_feature_data = normalize_data(testing_feature_data, training_feature_data_mean_std)

    IPython.embed()

    feature_dists, indices = tree.query(normalized_testing_feature_data)
    error = training_ground_truth_data[indices,:] - testing_ground_truth_data

    abs_error = np.abs(error)

    # mean absolute error
    mean_abs_error = np.mean(abs_error, axis=0)
    print mean_abs_error

    IPython.embed()



# data loading and processing
def load_data(desired_motion_code, load_from_txt=True):

    total_data_dict = {}

    if load_from_txt:
        # file_name_counter = 0

        total_data = None
        training_data = None
        testing_data = None

        # counter = 0

        for file_name_counter in range(start_index,end_index):

            file_name = training_data_path + motion_type_string + '_dynopt_result_' + str(file_name_counter) + '.txt'

            if os.path.isfile(file_name):
                with open(file_name,'r') as f:
                    for line in f:
                        content = [x.strip() for x in line.split(' ')]
                        data = np.array([float(datum) for datum in content])
                        motion_code = int(data[0])

                        if motion_code in total_data_dict:
                            total_data_dict[motion_code] = np.vstack((total_data_dict[motion_code],data))
                        else:
                            total_data_dict[motion_code] = data

                        # counter += 1

                        # if counter % 10000 == 0:
                        #     for motion_code in motion_code_list:
                        #         if motion_code in total_data_dict:
                        #             print 'Contact transition code: ', motion_code, ', Original dataset size: ', total_data_dict[motion_code].shape[0]
                        #             print 'Find unique entries of the total dataset'
                        #             total_data_dict[motion_code] = np.unique(total_data_dict[motion_code], axis=0)
                        #             print 'Unique dataset size: ', total_data_dict[motion_code].shape[0]

                        #     file_stream = open(training_data_path + 'dynopt_total_data_dict_' + str(counter/100-100) + '_' + str(counter/100),'w')
                        #     pickle.dump(total_data_dict, file_stream)
                        #     file_stream.close()

                        #     total_data_dict = {}


                print 'Loaded ' + motion_type_string + '_dynopt_result_'+str(file_name_counter)+'.txt'


        for motion_code in motion_code_list:
            if motion_code in total_data_dict:
                print 'Contact transition code: ', motion_code, ', Original dataset size: ', total_data_dict[motion_code].shape[0]
                print 'Find unique entries of the total dataset'
                total_data_dict[motion_code] = np.unique(total_data_dict[motion_code], axis=0)
                print 'Unique dataset size: ', total_data_dict[motion_code].shape[0]

        file_stream = open(training_data_path + motion_type_string + '_dynopt_total_data_dict_' + str(start_index) + '_' + str(end_index),'w')
        pickle.dump(total_data_dict, file_stream)
        file_stream.close()

        # total_data = total_data_dict[desired_motion_code]

        return

    else:
        print 'Unpickling dynopt_total_data...'

        file_stream = open(training_data_path + motion_type_string + '_dynopt_total_data_' + str(desired_motion_code),'r')
        total_data = pickle.load(file_stream)
        file_stream.close()

    # print 'Extract the data with specified contact statuses'

    # condition = total_data[:, contact_status_indices_list] == target_contact_status
    # total_data = total_data[condition[:,0],:]

    total_data = total_data[total_data[:,data_index_map['objective_value']] < 50000, :]

    print 'Random shuffle the dataset.'

    np.random.shuffle(total_data)

    total_data_num = total_data.shape[0]

    training_data_num = int(np.floor(0.8 * total_data_num))

    training_data = total_data[0:training_data_num,:]
    testing_data = total_data[training_data_num:,:]

    return training_data, testing_data

def get_data_mean_std(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    for i in contact_status_indices_list:
        mean[i] = 0.0
        std[i] = 1.0

    mean[data_index_map['objective_value']] = 0.0

    return np.stack((mean, std))

def normalize_data(data, data_mean_std):
    data_num = data.shape[0]

    mean_mat = np.repeat(np.atleast_2d(data_mean_std[0,:]), data_num, axis=0)
    std_mat = np.repeat(np.atleast_2d(data_mean_std[1,:]), data_num, axis=0)

    normalized_data = np.divide((data - mean_mat), std_mat)

    return normalized_data

def unnormalize_data(normalized_data, data_mean_std):
    data_num = normalized_data.shape[0]

    mean_mat = np.repeat(np.atleast_2d(data_mean_std[0,:]), data_num, axis=0)
    std_mat = np.repeat(np.atleast_2d(data_mean_std[1,:]), data_num, axis=0)

    data = np.multiply(normalized_data, std_mat) + mean_mat

    return data

def predict(tf_nn_model, input_data, input_data_mean_std, output_data_mean_std):

    normalized_input_data = normalize_data(np.atleast_2d(input_data), input_data_mean_std)
    normalized_output_data = tf_nn_model.predict(normalized_input_data)
    output_data = unnormalize_data(normalized_output_data, output_data_mean_std)

    print output_data

    return output_data



# tensorflow interface
def build_NeuralNetwork_model(learning_rate, layer_size, drop_out_rate):

    # relu output layer for all outputs
    # model = keras.Sequential([
    #     keras.layers.Dense(layer_size, activation=tf.nn.relu, input_shape=(len(feature_label),)),
    #     # keras.layers.Dropout(drop_out_rate),
    #     keras.layers.Dense(layer_size, activation=tf.nn.relu),
    #     # keras.layers.Dropout(drop_out_rate),
    #     keras.layers.Dense(layer_size, activation=tf.nn.relu),
    #     # keras.layers.Dropout(drop_out_rate),
    #     keras.layers.Dense(len(prediction_label))
    # ])

    # relu output layer for the objective value
    # input_layer = keras.layers.Input(shape=(len(feature_label),))
    # hidden_layer_1 = keras.layers.Dense(layer_size, activation='relu')(input_layer)
    # hidden_layer_2 = keras.layers.Dense(layer_size, activation='relu')(hidden_layer_1)
    # hidden_layer_3 = keras.layers.Dense(layer_size, activation='relu')(hidden_layer_2)
    # output1 = keras.layers.Dense(len(prediction_label)-1)(hidden_layer_3)
    # output2 = keras.layers.Dense(1, activation='relu')(hidden_layer_3)
    # output_layer = keras.layers.concatenate([output1, output2])

    # relu output layer for the objective value and timing
    input_layer = keras.layers.Input(shape=(len(feature_label),))
    hidden_layer_1 = keras.layers.Dense(layer_size, activation='relu')(input_layer)
    hidden_layer_2 = keras.layers.Dense(layer_size, activation='relu')(hidden_layer_1)
    hidden_layer_3 = keras.layers.Dense(layer_size, activation='relu')(hidden_layer_2)
    output1 = keras.layers.Dense(len(prediction_label)-2)(hidden_layer_3)
    output2 = keras.layers.Dense(2, activation='relu')(hidden_layer_3)
    output_layer = keras.layers.concatenate([output1, output2])

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # optimizer = tf.train.RMSPropOptimizer(learning_rate)
    optimizer = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    # optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])

    return model

def learning_NeuralNetwork_model(normalized_training_data, learning_rate, layer_size, drop_out_rate):

    print 'Learning Neural Network model...'

    tf_nn_model = build_NeuralNetwork_model(learning_rate, layer_size, drop_out_rate)

    normalized_training_feature = normalized_training_data[:,feature_indices_list]
    normalized_training_ground_truth = normalized_training_data[:,prediction_indices_list]

    # Display training progress by printing a single dot for each completed epoch.
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs):
            if epoch % 100 == 0:
                print ''
                print '.'

    # Store training stats
    history = tf_nn_model.fit(normalized_training_feature, normalized_training_ground_truth, epochs=100, validation_split=0.2, verbose=0, callbacks=[PrintDot()])
    plot_history(history)

    return tf_nn_model, history

def evaluating_NeuralNetwork_model(tf_nn_model, normalized_testing_data, data_mean_std, desired_motion_code, learning_rate, layer_size, drop_out_rate, history):

    print 'Evaluating Neural Network model...'

    normalized_testing_feature = normalized_testing_data[:,feature_indices_list]
    normalized_testing_ground_truth = normalized_testing_data[:,prediction_indices_list]
    testing_ground_truth = unnormalize_data(normalized_testing_ground_truth, data_mean_std)

    [loss, mae] = tf_nn_model.evaluate(normalized_testing_feature, normalized_testing_ground_truth, verbose=0)

    normalized_prediction = tf_nn_model.predict(normalized_testing_feature)

    prediction = unnormalize_data(normalized_prediction, data_mean_std)

    error = prediction - testing_ground_truth
    abs_error = np.abs(error)

    # mean absolute error
    mean_abs_error = np.mean(abs_error, axis=0)
    print mean_abs_error

    # mean testing ground truth
    mean_testing_ground_truth = np.mean(testing_ground_truth, axis=0)
    print mean_testing_ground_truth

    # mae gap
    # TODO: the index needs a fix
    mae_gap = abs(np.array(history.history['mean_absolute_error'])[-1] - np.array(history.history['val_mean_absolute_error'])[-1])

    file_stream = open(training_data_path +  motion_type_string + '_test_dynopt_learning_result_' + str(desired_motion_code) + '_' + str(learning_rate) + '_' + str(layer_size) + '_' + str(drop_out_rate) + '.txt','w')

    for value in mean_testing_ground_truth:
        file_stream.write('%5.3f '%(value))

    for error in mean_abs_error:
        file_stream.write('%5.3f '%(error))

    file_stream.write('%5.3f'%(mae_gap))
    file_stream.close()
    IPython.embed()

def save_NeuralNetwork_model(tf_nn_model, data_mean_std, desired_motion_code, learning_rate, layer_size, drop_out_rate):

    tf_nn_model.save(training_data_path + motion_type_string + '_nn_model_' + str(desired_motion_code) + '_' + str(learning_rate) + '_' + str(layer_size) + '_' + str(drop_out_rate) + '.h5')
    input_mean_std_file_stream = open(training_data_path + motion_type_string + '_input_mean_std_' + str(desired_motion_code) + '_' + str(learning_rate) + '_' + str(layer_size) + '_' + str(drop_out_rate) + '.txt','w')
    output_mean_std_file_stream = open(training_data_path + motion_type_string + '_output_mean_std_' + str(desired_motion_code) + '_' + str(learning_rate) + '_' + str(layer_size) + '_' + str(drop_out_rate) + '.txt','w')

    input_data_mean_std = data_mean_std[:,feature_indices_list]
    output_data_mean_std = data_mean_std[:,prediction_indices_list]

    for i in range(input_data_mean_std.shape[1]):
        input_mean_std_file_stream.write('%5.3f %5.3f '%(input_data_mean_std[0,i],input_data_mean_std[1,i]))

    for i in range(output_data_mean_std.shape[1]):
        output_mean_std_file_stream.write('%5.3f %5.3f '%(output_data_mean_std[0,i],output_data_mean_std[1,i]))

    input_mean_std_file_stream.close()
    output_mean_std_file_stream.close()

# scikit-learn interface
def learning_SVR_model(normalized_training_data):

    print 'Learning SVR model...'

    normalized_training_feature = normalized_training_data[:,feature_indices_list]
    normalized_training_ground_truth = normalized_training_data[:,prediction_indices_list]

    # clf = svm.SVR(epsilon=0.01, kernel='rbf', C=500)
    clf = svm.SVR(kernel='rbf', verbose=True)
    multi_clf = multioutput.MultiOutputRegressor(clf, n_jobs=-1)
    multi_clf.fit(normalized_training_feature, normalized_training_ground_truth)

    IPython.embed()

    return multi_clf

def evaluating_SVR_model(svr_model, normalized_testing_data, data_mean_std):

    print 'Evaluating SVR model...'

    normalized_testing_feature = normalized_testing_data[:,feature_indices_list]
    normalized_testing_ground_truth = normalized_testing_data[:,prediction_indices_list]
    testing_ground_truth = unnormalize_data(normalized_testing_ground_truth, data_mean_std)

    normalized_prediction = np.zeros(normalized_testing_feature.shape)
    for i in range(normalized_testing_feature.shape[0]):
        normalized_prediction[i,:] = svr_model.predict(normalized_testing_feature[i,:])

    prediction = unnormalize_data(normalized_prediction, data_mean_std)

    error = prediction - testing_ground_truth
    abs_error = np.abs(error)

    # mean absolute error percentage
    mean_abs_error_percentage = np.mean(np.divide(abs_error, testing_ground_truth), axis=0) * 100
    print mean_abs_error_percentage

    # mean absolute error
    mean_abs_error = np.mean(abs_error, axis=0)
    print mean_abs_error

    IPython.embed()


# main
def main(learning_rate, layer_size, drop_out_rate, mode, motion_type, desired_motion_code):

    global motion_code_manip_map, motion_code_list, motion_type_string

    if motion_type == 'contact_transition':
        motion_code_list = contact_transition_code_list
        motion_code_manip_map = contact_transition_code_manip_map
    elif motion_type == 'zero_step_capture':
        motion_code_list = zero_step_capture_code_list
        motion_code_manip_map = zero_step_capture_code_manip_map
    elif motion_type == 'one_step_capture':
        motion_code_list = one_step_capture_code_list
        motion_code_manip_map = one_step_capture_code_manip_map

    motion_type_string = motion_type

    construct_data_label(desired_motion_code, motion_type)

    # construct the data index map
    for i in range(len(data_label)):
        data_index_map[data_label[i]] = i

    for label in feature_label:
        feature_indices_list.append(data_index_map[label])

    for label in prediction_label:
        prediction_indices_list.append(data_index_map[label])

    for label in contact_status_label:
        contact_status_indices_list.append(data_index_map[label])

    if mode == 'learning':
        training_method = 'NeuralNetwork'

        [training_data, testing_data] = load_data(desired_motion_code, load_from_txt=False)

        training_data = training_data[0:500000,:]
        # training_data = training_data[0:50,:]
        # training_data = training_data[0:6000,:]

        training_data_mean_std = get_data_mean_std(training_data)

        normalized_training_data = normalize_data(training_data, training_data_mean_std)
        normalized_testing_data = normalize_data(testing_data, training_data_mean_std)

        if training_method == 'SVR':
            # Using SVR
            svr_model = learning_SVR_model(normalized_training_data)
            evaluating_SVR_model(svr_model, normalized_testing_data, training_data_mean_std[:,prediction_indices_list])

        elif training_method == 'NeuralNetwork':
            # Using NN
            tf_nn_model, history = learning_NeuralNetwork_model(normalized_training_data,learning_rate,layer_size,drop_out_rate)
            evaluating_NeuralNetwork_model(tf_nn_model, normalized_testing_data, training_data_mean_std[:,prediction_indices_list], desired_motion_code, learning_rate, layer_size, drop_out_rate, history)
            save_NeuralNetwork_model(tf_nn_model, training_data_mean_std, desired_motion_code, learning_rate, layer_size, drop_out_rate)

            # IPython.embed()

        elif training_method == 'NearestNeighbor':
            nearest_neighbor_test(training_data, testing_data, training_data_mean_std)

    elif mode == 'prediction':

        model_code = str(desired_motion_code) + '_' + str(learning_rate) + '_' + str(layer_size) + '_' + str(drop_out_rate)

        tf_nn_model = keras.models.load_model(model_path + motion_type_string + '_nn_model_' + model_code + '.h5')

        tmp_input_data_mean_std = np.loadtxt(model_path + motion_type_string + '_input_mean_std_' + model_code + '.txt')
        tmp_output_data_mean_std = np.loadtxt(model_path + motion_type_string + '_output_mean_std_' + model_code + '.txt')

        input_data_mean_std = np.zeros((2,tmp_input_data_mean_std.shape[0]/2))
        input_data_mean_std[0,:] = tmp_input_data_mean_std[::2]
        input_data_mean_std[1,:] = tmp_input_data_mean_std[1::2]

        output_data_mean_std = np.zeros((2,tmp_output_data_mean_std.shape[0]/2))
        output_data_mean_std[0,:] = tmp_output_data_mean_std[0::2]
        output_data_mean_std[1,:] = tmp_output_data_mean_std[1::2]

        # testing_feature = np.array([[0,0.1,0,0,0,0, 0,-0.1,0,0,0,0, 0,0.1,0,0,0,0, 0,0,1.0, 0,0,0],
        #                             [0,0.1,0,0,0,0, 0,-0.1,0,0,0,0, 0,0.2,0,0,0,0, 0,0,1.0, 0,0,0],
        #                             [0,0.1,0,0,0,0, 0,-0.1,0,0,0,0, 0,0.3,0,0,0,0, 0,0,1.0, 0,0,0],
        #                             [0,0.1,0,0,0,0, 0,-0.1,0,0,0,0, 0,0.4,0,0,0,0, 0,0,1.0, 0,0,0],
        #                             [0,0.1,0,0,0,0, 0,-0.1,0,0,0,0, 0,0.5,0,0,0,0, 0,0,1.0, 0,0,0],
        #                             [0,0.1,0,0,0,0, 0,-0.1,0,0,0,0, 0,0.6,0,0,0,0, 0,0,1.0, 0,0,0],
        #                             [0,0.1,0,0,0,0, 0,-0.1,0,0,0,0, 0,0.7,0,0,0,0, 0,0,1.0, 0,0,0],
        #                             [0,0.1,0,0,0,0, 0,-0.1,0,0,0,0, 0,0.8,0,0,0,0, 0,0,1.0, 0,0,0],
        #                             [0,0.1,0,0,0,0, 0,-0.1,0,0,0,0, 0,0.9,0,0,0,0, 0,0,1.0, 0,0,0],
        #                             [0,0.1,0,0,0,0, 0,-0.1,0,0,0,0, 0,1.0,0,0,0,0, 0,0,1.0, 0,0,0]])

        # normalized_testing_feature = normalize_data(testing_feature, input_data_mean_std)
        # normalized_prediction = tf_nn_model.predict(normalized_testing_feature)
        # prediction = unnormalize_data(normalized_prediction, output_data_mean_std)

        IPython.embed()

    elif mode == 'read_data':
        load_data(desired_motion_code, load_from_txt=True)
        return

    elif mode == 'combine_data':
        combine_data()
        return


if __name__ == "__main__":

    learning_rate = float(sys.argv[1])
    layer_size = int(sys.argv[2])
    drop_out_rate = float(sys.argv[3])
    mode = sys.argv[4]
    motion_type = sys.argv[5]
    desired_motion_code = int(sys.argv[6])

    if len(sys.argv) > 7:
        start_index = int(sys.argv[7])
        end_index = int(sys.argv[8])

    main(learning_rate=learning_rate,
         layer_size=layer_size,
         drop_out_rate=drop_out_rate,
         mode=mode,
         motion_type=motion_type,
         desired_motion_code=desired_motion_code)
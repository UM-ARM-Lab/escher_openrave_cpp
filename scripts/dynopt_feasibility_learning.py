# imports
from sklearn import svm, multioutput
from sklearn.metrics import roc_curve, auc

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

data_label = []

contact_status_label = ['motion_code']

feature_label = []

prediction_label = ['feasibility']

contact_status_indices_list = []
feature_indices_list = []
prediction_indices_list = []

training_data_path = '../data/dynopt_result/'
model_path = '../data/dynopt_result/feasibility_classification_nn_models/'

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
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    # plt.plot(history.epoch, np.array(history.history['loss']), label='Train Loss')
    # plt.plot(history.epoch, np.array(history.history['val_loss']), label='Val Loss')
    plt.plot(history.epoch, np.array(history.history['acc']), label='Train Accuracy')
    plt.plot(history.epoch, np.array(history.history['val_acc']), label='Val Accuracy')
    plt.legend()
    # plt.ylim([0,5])

def test_data_feasibility_gap(data):
    data_mean_std = get_data_mean_std(data)
    feature_data = data[:,feature_indices_list]
    feature_data_mean_std = data_mean_std[:,feature_indices_list]
    normalized_feature_data = normalize_data(feature_data, feature_data_mean_std)

    normalized_feasible_feature_data = normalized_feature_data[data[:,prediction_indices_list[0]] == 1, :]
    normalized_infeasible_feature_data = normalized_feature_data[data[:,prediction_indices_list[0]] == 0, :]

    normalized_feasible_feature_data_tree = scipy.spatial.KDTree(normalized_feasible_feature_data, leafsize=int(normalized_feasible_feature_data.shape[0]/10))
    normalized_infeasible_feature_data_tree = scipy.spatial.KDTree(normalized_infeasible_feature_data, leafsize=int(normalized_infeasible_feature_data.shape[0]/10))

    sample_num = 3000

    cross_class_dist = np.zeros(sample_num)
    in_infeasible_class_dist = np.zeros(sample_num)
    in_feasible_class_dist = np.zeros(sample_num)

    # get the distance from infeasible data to feasible data
    for i in range(sample_num):
        if i%100 == 0:
            print i
        dist, index = normalized_feasible_feature_data_tree.query(normalized_infeasible_feature_data[i,:])
        cross_class_dist[i] = dist

    print 'Mean cross class distance: ', np.mean(cross_class_dist)
    IPython.embed()

    # get the distance from infeasible data to feasible data
    for i in range(sample_num):
        if i%100 == 0:
            print i
        dists, indices = normalized_infeasible_feature_data_tree.query(normalized_infeasible_feature_data[i,:],2)
        in_infeasible_class_dist[i] = dists[1]

    print 'Mean in-infeasible class distance: ', np.mean(in_infeasible_class_dist)
    IPython.embed()

    # get the distance from feasible data to infeasible data
    for i in range(sample_num):
        if i%100 == 0:
            print i
        dists, indices = normalized_feasible_feature_data_tree.query(normalized_feasible_feature_data[i,:],2)
        in_feasible_class_dist[i] = dists[1]

    print 'Mean in-feasible class distance: ', np.mean(in_feasible_class_dist)
    IPython.embed()

def test_data_repeat_condition(data):
    np.random.shuffle(data)

    feature_data = data[:,feature_indices_list]
    ground_truth_data = data[:,prediction_indices_list]

    feature_data_tree = scipy.spatial.KDTree(feature_data, leafsize=int(data.shape[0]/5))

    sample_num = 3000
    num_close_samples = np.zeros(sample_num)
    mean_objective_mae = np.zeros(sample_num)

    for i in range(sample_num):
        if i%100 == 0:
            print i
        indices = feature_data_tree.query_ball_point(feature_data[i,:], 0.01)
        num_close_samples[i] = len(indices)-1

    print 'Mean number of close samples: ', np.mean(num_close_samples)
    IPython.embed()

def test_closest_feasible_data_objective(data):
    data_mean_std = get_data_mean_std(data)
    feature_data = data[:,feature_indices_list]
    feature_data_mean_std = data_mean_std[:,feature_indices_list]
    normalized_feature_data = normalize_data(feature_data, feature_data_mean_std)

    normalized_feasible_feature_data = normalized_feature_data[data[:,prediction_indices_list[0]] == 1, :]
    normalized_infeasible_feature_data = normalized_feature_data[data[:,prediction_indices_list[0]] == 0, :]

    normalized_feasible_feature_data_tree = scipy.spatial.KDTree(normalized_feasible_feature_data, leafsize=int(normalized_feasible_feature_data.shape[0]/10))

    sample_num = 3000

    for i in range(sample_num):
        if i%100 == 0:
            print i
        dist, index = normalized_feasible_feature_data_tree.query(normalized_infeasible_feature_data[i,:])
        cross_class_dist[i] = dist

    print 'Mean cross class distance: ', np.mean(cross_class_dist)
    IPython.embed()


def construct_data_label(desired_motion_code, motion_type):
    se3_list = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

    data_label.append('motion_code')

    # prev_lf and prev_rf are guaranteed
    for manip in motion_code_manip_map[desired_motion_code]:
        for j in range(len(se3_list)):
            feature_label.append(manip + '_' + se3_list[j])
            data_label.append(manip + '_' + se3_list[j])

    feature_label.extend(['start_com_x', 'start_com_y', 'start_com_z',
                          'start_com_dot_x', 'start_com_dot_y', 'start_com_dot_z'])

    data_label.extend(['start_com_x', 'start_com_y', 'start_com_z',
                       'start_com_dot_x', 'start_com_dot_y', 'start_com_dot_z'])

    data_label.append('feasibility')

def combine_data():

    infeasible_total_data_dict = {}

    try:
        file_stream = open(training_data_path + motion_type_string + '_dynopt_infeasible_total_data_dict','r')
        infeasible_total_data_dict = pickle.load(file_stream)
        file_stream.close()
    except Exception:
        infeasible_total_data_dict = {}

    i = 0

    while(True):

        try:
            file_stream = open(training_data_path + motion_type_string + '_dynopt_infeasible_total_data_dict_' + str(i*100) + '_' + str((i+1)*100),'r')
            data_dict = pickle.load(file_stream)
            file_stream.close()
        except Exception:
            break

        for motion_code, data in data_dict.iteritems():
            if motion_code in infeasible_total_data_dict:
                try:
                    infeasible_total_data_dict[motion_code] = np.vstack((infeasible_total_data_dict[motion_code], data_dict[motion_code]))
                except Exception:
                    continue
            else:
                infeasible_total_data_dict[motion_code] = np.copy(data_dict[motion_code])

        print motion_type_string + '_dynopt_infeasible_total_data_dict_' + str(i*100) + '_' + str((i+1)*100)

        i += 1

    for motion_code in motion_code_list:
        if motion_code in infeasible_total_data_dict:
            print 'Contact transition code: ', motion_code, ', Original dataset size: ', infeasible_total_data_dict[motion_code].shape[0]
            print 'Find unique entries of the total dataset'
            infeasible_total_data_dict[motion_code] = np.unique(infeasible_total_data_dict[motion_code], axis=0)
            print 'Unique dataset size: ', infeasible_total_data_dict[motion_code].shape[0]

    file_stream = open(training_data_path + motion_type_string + '_dynopt_infeasible_total_data_dict','w')
    pickle.dump(infeasible_total_data_dict, file_stream)
    file_stream.close()

    for motion_code, data in infeasible_total_data_dict.iteritems():
        file_stream = open(training_data_path + motion_type_string + '_dynopt_infeasible_total_data_' + str(motion_code),'w')
        pickle.dump(data, file_stream)
        file_stream.close()


# def evaluate_data_complexity(data, data_mean_std):

#     feature_data = data[:,feature_indices_list]
#     feature_data_mean_std = data_mean_std[:,feature_indices_list]
#     normalized_feature_data = normalize_data(feature_data, feature_data_mean_std)

#     tree = scipy.spatial.KDTree(normalized_feature_data, leafsize=int(normalized_feature_data.shape[0]/10))

#     sample_num = 100
#     max_sample_radius = 20
#     radius_range = range(1,max_sample_radius+1)
#     sample_std_mean_by_radius = np.zeros(max_sample_radius)
#     sample_std_std_by_radius = np.zeros(max_sample_radius)
#     counter = 0

#     for radius in radius_range:
#         print 'radius: ', radius
#         sample_std = np.zeros(sample_num)
#         for sample in range(sample_num):
#             sample_index = random.randint(0, normalized_feature_data.shape[0]-1)

#             sample_std[sample] = np.std(data[tree.query_ball_point(normalized_feature_data[sample_index], radius), data_index_map['objective_value']])

#         sample_std_mean_by_radius[counter] = np.mean(sample_std)
#         sample_std_std_by_radius[counter] = np.std(sample_std)
#         counter += 1

#     plt.errorbar(radius_range, sample_std_mean_by_radius, sample_std_std_by_radius)
#     plt.show()

#     IPython.embed()

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

    infeasible_total_data_dict = {}

    if load_from_txt:
        # file_name_counter = 0
        total_data = None
        training_data = None
        testing_data = None
        # counter = 0

        # Load infeasible entries
        for file_name_counter in range(start_index,end_index):
            file_name = training_data_path + motion_type_string + '_dynopt_result_infeasible_'+str(file_name_counter)+'.txt'
            if os.path.isfile(file_name):
                with open(file_name,'r') as f:
                    for line in f:
                        content = [x.strip() for x in line.split(' ')]
                        content.remove('')
                        data = np.array([float(datum) for datum in content])
                        data = np.hstack((data,0))
                        motion_code = int(data[0])

                        if motion_code in infeasible_total_data_dict:
                            infeasible_total_data_dict[motion_code] = np.vstack((infeasible_total_data_dict[motion_code],data))
                        else:
                            infeasible_total_data_dict[motion_code] = data

                        # counter += 1

                        # if counter % 10000 == 0:
                        #     for motion_code in contact_transition_code_list:
                        #         if motion_code in infeasible_total_data_dict:
                        #             print 'Contact transition code: ', motion_code, ', Original dataset size: ', infeasible_total_data_dict[motion_code].shape[0]
                        #             print 'Find unique entries of the total dataset'
                        #             infeasible_total_data_dict[motion_code] = np.unique(infeasible_total_data_dict[motion_code], axis=0)
                        #             print 'Unique dataset size: ', infeasible_total_data_dict[motion_code].shape[0]

                        #     file_stream = open(training_data_path + 'dynopt_infeasible_total_data_dict_' + str(counter/100-100) + '_' + str(counter/100),'w')
                        #     pickle.dump(infeasible_total_data_dict, file_stream)
                        #     file_stream.close()

                        #     infeasible_total_data_dict = {}

                print 'Loaded ' + motion_type_string + '_dynopt_result_infeasible_' + str(file_name_counter)+'.txt'


        for motion_code in contact_transition_code_list:
            if motion_code in infeasible_total_data_dict:
                print 'Contact transition code: ', motion_code, ', Original dataset size: ', infeasible_total_data_dict[motion_code].shape[0]
                print 'Find unique entries of the total dataset'
                infeasible_total_data_dict[motion_code] = np.unique(infeasible_total_data_dict[motion_code], axis=0)
                print 'Unique dataset size: ', infeasible_total_data_dict[motion_code].shape[0]

        file_stream = open(training_data_path + motion_type_string + '_dynopt_infeasible_total_data_dict_' + str(start_index) + '_' + str(end_index),'w')
        pickle.dump(infeasible_total_data_dict, file_stream)
        file_stream.close()

        # infeasible_total_data = infeasible_total_data_dict[desired_motion_code]

        return

    else:
        print 'Unpickling dynopt_infeasible_total_data...'

        file_stream = open(training_data_path + motion_type_string + '_dynopt_infeasible_total_data_' + str(desired_motion_code),'r')
        infeasible_total_data = pickle.load(file_stream)
        file_stream.close()


    # Load feasible entries
    print 'Unpickling dynopt_total_data...'

    file_stream = open(training_data_path + motion_type_string + '_dynopt_total_data_' + str(desired_motion_code),'r')
    feasible_total_data = pickle.load(file_stream)
    file_stream.close()

    feasible_total_data = np.hstack((feasible_total_data[:,contact_status_indices_list], feasible_total_data[:,feature_indices_list], np.ones((feasible_total_data.shape[0],1))))

    print 'Infeasible entries: ', infeasible_total_data.shape[0], ', Feasible entries: ', feasible_total_data.shape[0]

    np.random.shuffle(infeasible_total_data)
    np.random.shuffle(feasible_total_data)

    testing_data = np.vstack((infeasible_total_data[0:500,:],feasible_total_data[0:500,:]))
    infeasible_total_data = infeasible_total_data[500:50500,:]
    feasible_total_data = feasible_total_data[500:50500,:]

    # balance the dataset
    infeasible_data_num = infeasible_total_data.shape[0]
    feasible_data_num = feasible_total_data.shape[0]
    infeasible_total_data = infeasible_total_data[0:min(feasible_data_num,infeasible_data_num)+1,:]
    feasible_total_data = feasible_total_data[0:min(feasible_data_num,infeasible_data_num)+1,:]

    total_data = np.vstack((infeasible_total_data, feasible_total_data))

    print 'Random shuffle the dataset.'

    np.random.shuffle(total_data)

    training_data = total_data

    # total_data_num = total_data.shape[0]
    # training_data_num = int(np.floor(0.8 * total_data_num))
    # training_data = total_data[0:training_data_num,:]
    # testing_data = total_data[training_data_num:,:]

    return training_data, testing_data

def get_data_mean_std(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    for i in contact_status_indices_list:
        mean[i] = 0.0
        std[i] = 1.0

    for i in range(std.shape[0]):
        if std[i] == 0:
            std[i] = 1.0

    mean[data_index_map['feasibility']] = 0.0
    std[data_index_map['feasibility']] = 1.0

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

    input_layer = keras.layers.Input(shape=(len(feature_label),))
    hidden_layer_1 = keras.layers.Dense(layer_size, activation='relu')(input_layer)
    dropout_layer_1 = keras.layers.Dropout(drop_out_rate)(hidden_layer_1)
    hidden_layer_2 = keras.layers.Dense(layer_size, activation='relu')(dropout_layer_1)
    dropout_layer_2 = keras.layers.Dropout(drop_out_rate)(hidden_layer_2)
    hidden_layer_3 = keras.layers.Dense(layer_size, activation='relu')(dropout_layer_2)
    dropout_layer_3 = keras.layers.Dropout(drop_out_rate)(hidden_layer_3)
    output_layer = keras.layers.Dense(1, activation='sigmoid')(dropout_layer_3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # optimizer = tf.train.RMSPropOptimizer(learning_rate)
    optimizer = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    # optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

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
                print '. '

    # Store training stats
    history = tf_nn_model.fit(normalized_training_feature, normalized_training_ground_truth, epochs=100, validation_split=0.2, verbose=0, callbacks=[PrintDot()])
    plot_history(history)

    return tf_nn_model, history

def evaluating_NeuralNetwork_model(tf_nn_model, normalized_testing_data, data_mean_std, desired_motion_code, learning_rate, layer_size, drop_out_rate, history):

    print 'Evaluating Neural Network model...'

    normalized_testing_feature = normalized_testing_data[:,feature_indices_list]
    normalized_testing_ground_truth = normalized_testing_data[:,prediction_indices_list]
    testing_ground_truth = unnormalize_data(normalized_testing_ground_truth, data_mean_std)

    [loss, accuracy] = tf_nn_model.evaluate(normalized_testing_feature, normalized_testing_ground_truth, verbose=0)

    normalized_prediction = tf_nn_model.predict(normalized_testing_feature)

    prediction = unnormalize_data(normalized_prediction, data_mean_std)

    error = prediction - testing_ground_truth
    abs_error = np.abs(error)

    print 'accuracy: ', accuracy

    # # mean absolute error percentage
    # mean_abs_error_percentage = np.mean(np.divide(abs_error, testing_ground_truth), axis=0) * 100
    # print mean_abs_error_percentage

    # # mean absolute error
    # mean_abs_error = np.mean(abs_error, axis=0)
    # print mean_abs_error

    # # mae gap
    file_stream = open(training_data_path + motion_type_string + '_test_dynopt_feasibility_result_' + str(desired_motion_code) + '_' + str(learning_rate) + '_' + str(layer_size) + '_' + str(drop_out_rate) + '.txt','w')
    file_stream.write('%5.3f'%(accuracy))

    # ROC and AUC
    normalized_prediction.ravel()
    fpr, tpr, thresholds = roc_curve(normalized_testing_ground_truth, normalized_prediction)
    auc_score = auc(fpr, tpr)
    file_stream.write(' %5.3f'%(auc_score))
    print 'auc: ', auc_score

    file_stream.close()
    # IPython.embed()

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

        if training_method == 'NeuralNetwork':
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

        data_mean_std = np.hstack((np.array([[0],[1]]), input_data_mean_std, output_data_mean_std))

        file_stream = open(training_data_path + motion_type_string + '_dynopt_infeasible_test_data_' + str(desired_motion_code),'r')
        testing_infeasible_data = pickle.load(file_stream)
        file_stream.close()

        file_stream = open(training_data_path + motion_type_string + '_dynopt_test_data_' + str(desired_motion_code),'r')
        testing_feasible_data = pickle.load(file_stream)
        testing_feasible_data = np.hstack((testing_feasible_data[:,contact_status_indices_list], testing_feasible_data[:,feature_indices_list], np.ones((testing_feasible_data.shape[0],1))))
        file_stream.close()

        IPython.embed()

        testing_data = np.vstack((testing_infeasible_data, testing_feasible_data))
        normalized_testing_data = normalize_data(testing_data, data_mean_std)

        evaluating_NeuralNetwork_model(tf_nn_model, normalized_testing_data, input_data_mean_std, desired_motion_code, learning_rate, layer_size, drop_out_rate, None)

    elif mode == 'testing':
        [training_data, testing_data] = load_data(desired_motion_code, load_from_txt=False)
        # test_data_feasibility_gap(np.vstack((training_data, testing_data)))
        test_data_repeat_condition(np.vstack((training_data, testing_data)))
        # IPython.embed()

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
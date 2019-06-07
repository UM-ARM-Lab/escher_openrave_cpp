from __future__ import print_function

__author__ = 'yu-chi'

# System Imports

# 3rd-Party Imports
import numpy as np
import scipy.spatial as sp
import math
import openravepy as rave
import copy
import random
import time
import pickle
import sys
import IPython
import getopt

# Local Imports
# import load_escher
import load_athena
# import load_hermes_full

# from config_parameter import *
from transformation_conversion import *
from environment_handler_2 import environment_handler
from map_grid import map_grid_dim
from node import *
from contact_projection import *
from draw import DrawStance

SAMPLE_SIZE_EACH_ENVIRONMENT = 50
GRID_RESOLUTION = 0.15
ANGLE_RESOLUTION = 15.0

# save all transitions to this list
transitions = []

# save all environments to this list
environments = []


def contact_degree_to_radian(long_list):
    """
    Input:
    long_list should be a list
    """
    new_list = list(long_list)
    for i in range(len(long_list) // 6):
        for j in range(3, 6):
            new_list[6 * i + j] = long_list[6 * i + j] * np.pi / 180
    return new_list


def position_to_cell_index(position,resolution):
    resolutions = [resolution, resolution, ANGLE_RESOLUTION] # resolution is the resolution of x and y, which is 0.15m in this case
    adjusted_position = [position[0], position[1], first_terminal_angle(position[2])] # first_terminal_angle is defined in transformation_conversion.py
    cell_index = [None] * len(position)

    for i, v in enumerate(adjusted_position):
        if abs(v) > resolutions[i]/2.0:
            v += -np.sign(v) * resolutions[i]/2.0
            cell_index[i] = int(np.sign(v) * math.ceil(abs(v)/resolutions[i]))
        else:
            cell_index[i] = 0

    return cell_index


class contact_transition:
    def __init__(self,init_node,final_node,grid_resolution):
        self.init_node = init_node
        self.final_node = final_node

        init_virtual_body_pose = init_node.get_virtual_body_pose()
        final_virtual_body_pose = final_node.get_virtual_body_pose()
        init_virtual_body_pose_SE2 = init_virtual_body_pose[0:2] + [init_virtual_body_pose[5]]
        final_virtual_body_pose_SE2 = final_virtual_body_pose[0:2] + [final_virtual_body_pose[5]]

        self.init_virtual_body_cell = position_to_cell_index(init_virtual_body_pose_SE2, grid_resolution)
        self.final_virtual_body_cell = position_to_cell_index(final_virtual_body_pose_SE2, grid_resolution)
        self.move_manip = final_node.prev_move_manip

        self.contact_transition_type = None
        self.feature_vector_contact_part = []
        self.normalized_init_l_leg = None
        self.normalized_init_r_leg = None
        self.normalized_init_l_arm = None
        self.normalized_init_r_arm = None

    def get_contact_transition_type(self):
        # please fill in this part
        # only consider the left side
        if self.init_node.get_contact_manip_num() == 2:
            if self.final_node.get_contact_manip_num() == 2:
                self.contact_transition_type = 0
            elif self.final_node.get_contact_manip_num() == 3:
                self.contact_transition_type = 1
            else:
                raw_input('Invalid Transition')

        elif self.init_node.get_contact_manip_num() == 3:
            if self.final_node.get_contact_manip_num() == 2:
                self.contact_transition_type = 4
            elif self.final_node.get_contact_manip_num() == 3:
                if self.final_node.prev_move_manip == LEFT_LEG:
                    if self.init_node.manip_in_contact('l_arm'):
                        self.contact_transition_type = 2
                    elif self.init_node.manip_in_contact('r_arm'):
                        self.contact_transition_type = 3
                    else:
                        raw_input('Invalid Transition')
                elif self.final_node.prev_move_manip == LEFT_ARM:
                    self.contact_transition_type = 5
                else:
                    raw_input('Invalid Transition')
            elif self.final_node.get_contact_manip_num() == 4:
                self.contact_transition_type = 6
            else:
                raw_input('Invalid Transition')

        elif self.init_node.get_contact_manip_num() == 4:
            if self.final_node.get_contact_manip_num() == 3:
                self.contact_transition_type = 8
            elif self.final_node.get_contact_manip_num() == 4:
                if self.final_node.prev_move_manip == LEFT_LEG:
                    self.contact_transition_type = 7
                elif self.final_node.prev_move_manip == LEFT_ARM:
                    self.contact_transition_type = 9
                else:
                    raw_input('Invalid Transition')
            else:
                raw_input('Invalid Transition')

        else:
            raw_input('Invalid Transition')

        return self.contact_transition_type


    def get_feature_vector_contact_part(self):
        self.get_contact_transition_type()

        # center the poses about the mean feet pose
        inv_mean_feet_transform = xyzrpy_to_inverse_SE3(self.init_node.get_mean_feet_xyzrpy())

        init_left_leg = SE3_to_xyzrpy(np.dot(inv_mean_feet_transform, xyzrpy_to_SE3(self.init_node.left_leg)))
        init_right_leg = SE3_to_xyzrpy(np.dot(inv_mean_feet_transform, xyzrpy_to_SE3(self.init_node.right_leg)))
        final_left_leg = SE3_to_xyzrpy(np.dot(inv_mean_feet_transform, xyzrpy_to_SE3(self.final_node.left_leg)))

        self.normalized_init_l_leg = init_left_leg
        self.normalized_init_r_leg = init_right_leg

        if self.init_node.manip_in_contact('l_arm'):
            init_left_arm = SE3_to_xyzrpy(np.dot(inv_mean_feet_transform, xyzrpy_to_SE3(self.init_node.left_arm)))
            self.normalized_init_l_arm = init_left_arm
        
        if self.init_node.manip_in_contact('r_arm'):
            init_right_arm = SE3_to_xyzrpy(np.dot(inv_mean_feet_transform, xyzrpy_to_SE3(self.init_node.right_arm)))
            self.normalized_init_r_arm = init_right_arm

        if self.final_node.manip_in_contact('l_arm'):
            final_left_arm = SE3_to_xyzrpy(np.dot(inv_mean_feet_transform, xyzrpy_to_SE3(self.final_node.left_arm)))

        # construct the feature vector
        if self.contact_transition_type == 0:
            self.feature_vector_contact_part = contact_degree_to_radian(init_left_leg + init_right_leg + final_left_leg)

        elif self.contact_transition_type == 1:
            self.feature_vector_contact_part = contact_degree_to_radian(init_left_leg + init_right_leg + final_left_arm)

        elif self.contact_transition_type == 2:
            self.feature_vector_contact_part = contact_degree_to_radian(init_left_leg + init_right_leg + init_left_arm + final_left_leg)

        elif self.contact_transition_type == 3:
            self.feature_vector_contact_part = contact_degree_to_radian(init_left_leg + init_right_leg + init_right_arm + final_left_leg)

        elif self.contact_transition_type == 4:
            self.feature_vector_contact_part = contact_degree_to_radian(init_left_leg + init_right_leg + init_left_arm)

        elif self.contact_transition_type == 5:
            self.feature_vector_contact_part = contact_degree_to_radian(init_left_leg + init_right_leg + init_left_arm + final_left_arm)

        elif self.contact_transition_type == 6:
            self.feature_vector_contact_part = contact_degree_to_radian(init_left_leg + init_right_leg + init_right_arm + final_left_arm)

        elif self.contact_transition_type == 7:
            self.feature_vector_contact_part = contact_degree_to_radian(init_left_leg + init_right_leg + init_left_arm + init_right_arm + final_left_leg)

        elif self.contact_transition_type == 8:
            self.feature_vector_contact_part = contact_degree_to_radian(init_left_leg + init_right_leg + init_left_arm + init_right_arm)

        elif self.contact_transition_type == 9:
            self.feature_vector_contact_part = contact_degree_to_radian(init_left_leg + init_right_leg + init_left_arm + init_right_arm + final_left_arm)

        else:
            raw_input('Wrong Type.')

        # sample the initial CoM position and CoM velocity from data/dynopt_result/dataset
        # remember to normalize the feature vector using information in data/dynopt_result/*nn_models before sending to the network.
        return self.feature_vector_contact_part


# def extract_env_feature():

def sample_contact_transitions(env_handler,robot_obj,hand_transition_model,foot_transition_model,structures,grid_resolution,environment_index):
    # assume the robot is at (x,y) = (0,0), we sample 12 orientation (-75,-60,-45,-30,-15,0,15,30,45,60,75,90)
    # other orientations are just these 12 orientations plus 180*n, so we do not need to sample them.
    handles = []
    init_node_list = []
    for orientation in range(-75,91,15):
        rave.raveLogInfo('Orientation: ' + repr(orientation) + ' degrees.')
        orientation_rad = orientation * DEG2RAD
        orientation_rotation_matrix = rpy_to_SO3([0, 0, orientation])

        # first check what are the available hand contacts for left and right hand
        # make a dummy node to start sampling all possible hand contacts
        dummy_init_left_leg = [-0.1*math.sin(orientation_rad),0.1*math.cos(orientation_rad),0,0,0,orientation]
        dummy_init_right_leg = [0.1*math.sin(orientation_rad),-0.1*math.cos(orientation_rad),0,0,0,orientation]
        dummy_init_left_arm = copy.copy(no_contact)
        dummy_init_right_arm = copy.copy(no_contact)
        dummy_init_node = node(dummy_init_left_leg, dummy_init_right_leg, dummy_init_left_arm, dummy_init_right_arm)

        # all possible hand contacts from (0, 0, theta)
        init_left_hand_pose_lists = [copy.copy(no_contact)] # ?????
        init_right_hand_pose_lists = [copy.copy(no_contact)] # ?????

        for arm_orientation in hand_transition_model:
            if arm_orientation[0] != -99.0:
                if hand_projection(robot_obj, LEFT_ARM, arm_orientation, dummy_init_node, structures):
                    init_left_hand_pose_lists.append(copy.copy(dummy_init_node.left_arm))
                if hand_projection(robot_obj, RIGHT_ARM, arm_orientation, dummy_init_node, structures):
                    init_right_hand_pose_lists.append(copy.copy(dummy_init_node.right_arm))

        rave.raveLogInfo(repr(len(init_left_hand_pose_lists)-1) + ' initial left hand contact poses.')
        rave.raveLogInfo(repr(len(init_right_hand_pose_lists)-1) + ' initial right hand contact poses.')

        # for each combination of hand contacts, find all the foot combinations to make the torso pose to be (0,0,theta)
        rave.raveLogInfo('Collect initial nodes...')
        for init_left_hand_pose in init_left_hand_pose_lists:
            for init_right_hand_pose in init_right_hand_pose_lists:
                dummy_init_node.left_arm = copy.copy(init_left_hand_pose)
                dummy_init_node.right_arm = copy.copy(init_right_hand_pose)

                # check if the hand contacts are too far away or too close
                if not dummy_init_node.node_feasibile(robot_obj):
                    continue

                # based on the dummy init virtual body pose, recenter the node to be at (0,0,orientation)
                dummy_init_virtual_body_pose = dummy_init_node.get_virtual_body_pose()
                mean_feet_position_offset = np.atleast_2d(np.array(dummy_init_virtual_body_pose[0:2])).T * (dummy_init_node.get_contact_manip_num()/2.0)

                # sample the initial foot step combinations
                for foot_transition in foot_transition_model:
                    init_node = copy.deepcopy(dummy_init_node)

                    init_left_foot_position = np.dot(orientation_rotation_matrix[0:2,0:2], np.array([[-foot_transition[0]/2],[foot_transition[1]/2]])) - mean_feet_position_offset
                    init_right_foot_position = np.dot(orientation_rotation_matrix[0:2,0:2], np.array([[foot_transition[0]/2],[-foot_transition[1]/2]])) - mean_feet_position_offset

                    init_node.left_leg = [init_left_foot_position[0,0], init_left_foot_position[1,0], LARGE_NUMBER, 0, 0, foot_transition[2]/2 + orientation]
                    init_node.right_leg = [init_right_foot_position[0,0], init_right_foot_position[1,0], LARGE_NUMBER, 0, 0, -foot_transition[2]/2 + orientation]

                    # init_node.left_leg = [init_left_foot_position[0,0], init_left_foot_position[1,0], 0.1, 0, 0, foot_transition[2]/2 + orientation]
                    # init_node.right_leg = [init_right_foot_position[0,0], init_right_foot_position[1,0], 0.1, 0, 0, -foot_transition[2]/2 + orientation]

                    # DrawStance(init_node, robot_obj, handles)
                    # if not foot_projection(robot_obj, init_node, structures):
                    #     IPython.embed()
                    # handles = []

                    # contact is in polygon
                    if foot_projection(robot_obj, init_node, structures):
                        init_node_list.append(init_node)
                        # print('----------------')
                        # print(mean_feet_position_offset.T)
                        # print(dummy_init_node.get_virtual_body_pose())
                        # print(init_node.get_virtual_body_pose())
                        # DrawStance(init_node, robot_obj, handles)
                        # raw_input()
                        # handles = []

    # here we get a set of initial nodes(contact combinations) that are with torso pose (0,0,theta)
    # branch contacts for left arm and leg, and record the torso pose transition
    rave.raveLogInfo('Collected ' + repr(len(init_node_list)) + ' initial nodes.')
    contact_transition_list = []
    for init_node in init_node_list:
        child_node_list = branching(init_node, foot_transition_model, hand_transition_model, structures, robot_obj)

        for child_node in child_node_list:
            # ??????????
            one_contact_transition = contact_transition(init_node, child_node, grid_resolution)
            contact_transition_list.append(one_contact_transition)
            temp_dict = {}
            temp_dict['environment_index'] = environment_index
            temp_dict['p1'] = one_contact_transition.init_virtual_body_cell
            temp_dict['p2'] = one_contact_transition.final_virtual_body_cell
            temp_dict['contact_transition_type'] = one_contact_transition.get_contact_transition_type()
            temp_dict['feature_vector_contact_part'] = one_contact_transition.get_feature_vector_contact_part()
            temp_dict['normalized_init_l_leg'] = one_contact_transition.normalized_init_l_leg
            temp_dict['normalized_init_r_leg'] = one_contact_transition.normalized_init_r_leg
            temp_dict['normalized_init_l_arm'] = one_contact_transition.normalized_init_l_arm
            temp_dict['normalized_init_r_arm'] = one_contact_transition.normalized_init_r_arm

            transitions.append(temp_dict)

            # print(contact_transition_list[-1].init_virtual_body_cell)
            # print(init_node.get_virtual_body_pose())
            # print(contact_transition_list[-1].final_virtual_body_cell)
            # print(child_node.get_virtual_body_pose())
            # print('previous move manipulator: ' + str(child_node.prev_move_manip))
            # DrawStance(init_node, robot_obj, handles)
            # DrawStance(child_node, robot_obj, handles)
            # raw_input()
            # handles = []

    rave.raveLogInfo('Collected ' + repr(len(contact_transition_list)) + ' contact transitions.')
    return contact_transition_list


def sample_env(env_handler, robot_obj, surface_source):
    env_handler.update_environment(robot_obj, surface_source=surface_source)
    return env_handler.structures


def main(robot_name='athena'): # for test
    environment_type = None
    try:
        inputs, _ = getopt.getopt(sys.argv[1:], "e:", ['environment_type'])

        for opt, arg in inputs:
            if opt == '-e' or opt == '--environment_type':
                environment_type = arg

    except getopt.GetoptError:
        print('usage: -e: [environment_type]')
        exit(1)

    environment_file = "large_environments_" + environment_type
    transition_file = "large_transitions_" + environment_type

    ### Initialize the environment handler
    rave.raveLogInfo('Load the Environment Handler.')
    env_handler = environment_handler()
    env = env_handler.env
    structures = env_handler.structures

    ### Set up the collision checker
    fcl = rave.RaveCreateCollisionChecker(env, "fcl_")
    if fcl is not None:
        env.SetCollisionChecker(fcl)
    else:
        print("FCL Not installed, falling back to ode")
        env.SetCollisionChecker(rave.RaveCreateCollisionChecker(env, "ode"))

    ### Construct the hand transition model
    hand_transition_model = []
    hand_pitch = [-60.0,-50.0,-40.0,-30.0,-20.0,-10.0,0.0,10.0,20.0,30.0,40.0,50.0,60.0] # horizontal
    hand_yaw = [-20.0,0.0,20.0] # vertical
    for pitch in hand_pitch:
        for yaw in hand_yaw:
            hand_transition_model.append((pitch,yaw))
    hand_transition_model.append((-99.0,-99.0))
  

    ### Load the foot transition model
    try:
        print('Load foot transition model...', end='')
        
        f = open('../data/escher_motion_planning_data/step_transition_model_mid_range_symmetric.txt','r')
        line = ' '
        foot_transition_model = []

        while(True):
            line = f.readline()
            if(line == ''):
                break
            foot_transition_model.append((float(line[0:5]),float(line[6:11]),float(line[12:17])))

        f.close()
        print('Done.')
    except Exception:
        raw_input('Not Found.')


    ### Load and initialize the robot
    rave.raveLogInfo('Load and Initialize the Robot.')

    if robot_name == 'escher':
        robot_obj = load_escher.escher(env)
    elif robot_name == 'athena': # use athena as your robot
        robot_obj = load_athena.athena(env)
    elif robot_name == 'hermes_full':
        robot_obj = load_hermes_full.hermes_full(env)

    robot_obj.robot.SetDOFValues(robot_obj.GazeboOriginalDOFValues)

    # sample environments and contact transitions
    for i in range(SAMPLE_SIZE_EACH_ENVIRONMENT):
        structures = sample_env(env_handler, robot_obj, 'one_step_env_' + environment_type)

        # save the environment
        env = {}
        env['ground_vertices'] = []
        env['ground_normals'] = []
        env['others_vertices'] = []
        env['others_normals'] = []
        for s in structures:
            if s.type == 'ground':
                env['ground_vertices'].append(s.vertices)
                env['ground_normals'].append(s.get_normal())
            elif s.type == 'others':
                env['others_vertices'].append(s.vertices)
                env['others_normals'].append(s.get_normal())
            else:
                print('invalid structure type')
                exit(1)
        environments.append(env)
                
        sample_contact_transitions(env_handler, robot_obj, hand_transition_model, foot_transition_model, structures, GRID_RESOLUTION, i)
        # IPython.embed()

    with open('../data/' + transition_file, 'w') as file:
        pickle.dump(transitions, file)
    with open('../data/' + environment_file, 'w') as file:
        pickle.dump(environments, file)
    # IPython.embed()
    rave.raveLogInfo('Sampling finished!!')
    print('data is saved in file: {} and {}'.format(environment_file, transition_file))


if __name__ == "__main__":
    main()

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

class contact_transition:
    def __init__(self,init_node,final_node):
        self.init_node = init_node
        self.final_node = final_node
        self.move_manip = final_node.prev_move_manip
        self.contact_transition_type = None
        self.feature_vector = []

    # def get_contact_transition_type(self):
        # please fill in this part
    
    # def get_feature_vector(self):
        # please fill in this part

# def extract_env_feature():

def sample_contact_transitions(env_handler, robot_obj, hand_transition_model, foot_transition_model, structures):
    # assume the robot is at (x,y) = (0,0), we sample 3 kinds of orientation (0, 30, 60)
    # other orientations are just those 3 orientations plus 90*n, so we do not need to sample them.
    for orientation in range(0,61,30):
        rave.raveLogInfo('Orientation: ' + repr(orientation) + ' degrees.')
        orientation_rad = orientation * DEG2RAD
        orientation_rotation_matrix = rpy_to_SO3([0, 0, orientation])
        init_node_list = []        
        
        # first check what are the available hand contacts for left and right hand
        # make a dummy node to start sampling all possible hand contacts
        dummy_init_left_leg = [-0.1*math.sin(orientation_rad),0.1*math.cos(orientation_rad),0,0,0,orientation]
        dummy_init_right_leg = [0.1*math.sin(orientation_rad),-0.1*math.cos(orientation_rad),0,0,0,orientation]
        dummy_init_left_arm = copy.copy(no_contact)
        dummy_init_right_arm = copy.copy(no_contact)
        dummy_init_node = node(dummy_init_left_leg, dummy_init_right_leg, dummy_init_left_arm, dummy_init_right_arm)

        init_left_hand_pose_lists = [copy.copy(no_contact)]
        init_right_hand_pose_lists = [copy.copy(no_contact)]

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
                dummy_init_virtual_body_pose = dummy_init_node.get_virtual_body_pose()

                # based on the dummy init virtual body pose, recenter the node to be at (0,0,orientation)
                mean_feet_position_offset = np.atleast_2d(np.array(dummy_init_virtual_body_pose[0:2])).T

                # check if the hand contacts are too far away or too close
                # TODO: need to find the shoulder point
                if dummy_init_node.manip_in_contact('l_arm'):
                    left_hand_mean_feet_dist = np.linalg.norm(np.array(init_left_hand_pose[0:2])+mean_feet_position_offset.T)
                    if left_hand_mean_feet_dist < robot_obj.min_arm_length or left_hand_mean_feet_dist > robot_obj.max_arm_length:
                        continue

                if dummy_init_node.manip_in_contact('r_arm'):
                    right_hand_mean_feet_dist = np.linalg.norm(np.array(init_right_hand_pose[0:2])+mean_feet_position_offset.T)
                    if right_hand_mean_feet_dist < robot_obj.min_arm_length or right_hand_mean_feet_dist > robot_obj.max_arm_length:
                        continue

                print('ahhh!!!!!')

                # sample the initial foot step combinations
                for foot_transition in foot_transition_model:
                    init_node = copy.deepcopy(dummy_init_node)

                    init_left_foot_position = np.dot(orientation_rotation_matrix[0:2,0:2], np.array([[-foot_transition[0]/2],[foot_transition[1]/2]])) - mean_feet_position_offset
                    init_right_foot_position = np.dot(orientation_rotation_matrix[0:2,0:2], np.array([[foot_transition[0]/2],[-foot_transition[1]/2]])) - mean_feet_position_offset

                    init_node.left_leg = [init_left_foot_position[0], init_left_foot_position[1], sys.float_info.max, 0, 0, foot_transition[2]/2 + orientation]
                    init_node.right_leg = [init_right_foot_position[0], init_right_foot_position[1], sys.float_info.max, 0, 0, -foot_transition[2]/2 + orientation]

                    # IPython.embed()

                    if foot_projection(robot_obj, init_node, structures):
                        init_node_list.append(init_node)
    
        # here we get a set of initial nodes(contact combinations) that are with torso pose (0,0,theta)
        # branch contacts for left arm and leg, and record the torso pose transition
        rave.raveLogInfo('Collected ' + repr(len(init_node_list)) + ' initial nodes.')
        child_node_list = []
        for init_node in init_node_list:
            print(init_node.get_virtual_body_pose())
            child_node_list += branching(init_node, foot_transition_model, hand_transition_model, structures, robot_obj)

def sample_env(env_handler, robot_obj, surface_source):
    env_handler.update_environment(robot_obj, surface_source=surface_source)
    return env_handler.structures

def main(robot_name='athena'): # for test

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
    hand_pitch = [-60.0,-50.0,-40.0,-30.0,-20.0,-10.0,0.0,10.0,20.0,30.0,40.0,50.0,60.0]
    hand_yaw = [-20.0,0.0,20.0]
    for pitch in hand_pitch:
        for yaw in hand_yaw:
            hand_transition_model.append((pitch,yaw))
    hand_transition_model.append((-99.0,-99.0))

    ### Load the step transition model
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


    ### sample environments, and contact transition
    # while(True):
    #     sample_env(env_handler, robot_obj, 'dynopt_test_env_1')

    structures = sample_env(env_handler, robot_obj, 'dynopt_test_env_6')
    sample_contact_transitions(env_handler, robot_obj, hand_transition_model, foot_transition_model, structures)


if __name__ == "__main__":
    main()
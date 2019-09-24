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
import pprint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Local Imports
# import load_escher
import load_athena
# import load_hermes_full

# from config_parameter import *
from transformation_conversion import *
from environment_handler_2 import environment_handler
# from map_grid import map_grid_dim
from node import *
from contact_projection import *
from draw import DrawStance

GRID_RESOLUTION = 0.15
ANGLE_RESOLUTION = 22.5


def position_to_cell_index(position,resolution):
    resolutions = [resolution, resolution, ANGLE_RESOLUTION] # resolution is the resolution of x and y, which is 0.05m in this case
    
    while position[2] < -180 - ANGLE_RESOLUTION / 2.0 or position[2] >= 180 - ANGLE_RESOLUTION / 2.0:
        if position[2] < -180 - ANGLE_RESOLUTION / 2.0:
            position[2] = position[2] + 360
        elif position[2] >= 180 - ANGLE_RESOLUTION / 2.0:
            position[2] = position[2] - 360 
    
    adjusted_position = [position[0], position[1], position[2]] # first_terminal_angle is defined in transformation_conversion.py
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


def sample_contact_transitions(env_handler,robot_obj,hand_transition_model,foot_transition_model1, foot_transition_model2,structures,grid_resolution, environment_index):
    # assume the robot is at (x,y) = (0,0), we sample 12 orientation (-75,-60,-45,-30,-15,0,15,30,45,60,75,90)
    # other orientations are just these 12 orientations plus 180*n, so we do not need to sample them.
    for orientation in np.arange(-180, 180, 22.5):
        nested_dict = {}
        handles = []
        init_node_list = []
        
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

        # rave.raveLogInfo(repr(len(init_left_hand_pose_lists)-1) + ' initial left hand contact poses.')
        # rave.raveLogInfo(repr(len(init_right_hand_pose_lists)-1) + ' initial right hand contact poses.')

        # for each combination of hand contacts, find all the foot combinations to make the torso pose to be (0,0,theta)
        # rave.raveLogInfo('Collect initial nodes...')
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
                for foot_transition in foot_transition_model1:
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
        # rave.raveLogInfo('Collected ' + repr(len(init_node_list)) + ' initial nodes.')
        contact_transition_list = []
        for init_node in init_node_list:
            child_node_list = branching(init_node, foot_transition_model2, hand_transition_model, structures, robot_obj)
            
            for child_node in child_node_list:
                # ??????????
                one_contact_transition = contact_transition(init_node, child_node, grid_resolution)
                contact_transition_list.append(one_contact_transition)

                final_cell = one_contact_transition.final_virtual_body_cell 

                if final_cell[0] not in nested_dict:
                    nested_dict[final_cell[0]] = {}
                if final_cell[1] not in nested_dict[final_cell[0]]:
                    nested_dict[final_cell[0]][final_cell[1]] = {}
                if final_cell[2] not in nested_dict[final_cell[0]][final_cell[1]]:
                    nested_dict[final_cell[0]][final_cell[1]][final_cell[2]] = 1
                else:
                    nested_dict[final_cell[0]][final_cell[1]][final_cell[2]] += 1

                # print(contact_transition_list[-1].init_virtual_body_cell)
                # print(init_node.get_virtual_body_pose())
                # print(contact_transition_list[-1].final_virtual_body_cell)
                # print(child_node.get_virtual_body_pose())
                # print('previous move manipulator: ' + str(child_node.prev_move_manip))
                # DrawStance(init_node, robot_obj, handles)
                # DrawStance(child_node, robot_obj, handles)
                # raw_input()
                # handles = []
        # rave.raveLogInfo('Sampling finished!!')
        # pprint.pprint(nested_dict)

        # IPython.embed()

        for ix in nested_dict.keys():
            for iy in nested_dict[ix].keys():
                for itheta in nested_dict[ix][iy]:
                    # print(str(ix) + ' ' + str(iy) + ' ' + str(itheta))
                    temp = int(itheta - round(orientation / ANGLE_RESOLUTION))
                    if temp > 10:
                        temp -= 16
                    elif temp < -10:
                        temp += 16
                    print(str(ix) + ' ' + str(iy) + ' ' + str(temp))

        # fig = plt.figure(orientation)
        # ax = fig.add_subplot(111, projection='3d')

        # _x = np.arange(min(nested_dict.keys()), max(nested_dict.keys()) + 1, 1)
        # _y = np.arange(min(nested_dict[0].keys()), max(nested_dict[0].keys()) + 1, 1)
        # _xx, _yy = np.meshgrid(_x, _y)
        # x, y = _xx.ravel(), _yy.ravel()
        
        # top = np.zeros((x.shape[0],), dtype=int)
        # for i in range(x.shape[0]):
        #     if y[i] in nested_dict[x[i]]:
        #         for j in nested_dict[x[i]][y[i]]:
        #             top[i] += nested_dict[x[i]][y[i]][j]

        # bottom = np.zeros_like(top)
        # width = depth = 1
        # ax.bar3d(x, y, bottom, width, depth, top)

        # plt.show()


        # rave.raveLogInfo('Collected ' + repr(len(contact_transition_list)) + ' contact transitions.')
    return contact_transition_list


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
    hand_pitch = [-60.0,-50.0,-40.0,-30.0,-20.0,-10.0,0.0,10.0,20.0,30.0,40.0,50.0,60.0] # horizontal
    hand_yaw = [-20.0,0.0,20.0] # vertical
    for pitch in hand_pitch:
        for yaw in hand_yaw:
            hand_transition_model.append((pitch,yaw))
    hand_transition_model.append((-99.0,-99.0))
  

    ### Load the foot transition model
    try:
        print('Load foot transition model 1...', end='')
        
        f = open('../data/escher_motion_planning_data/step_transition_model_mid_range_symmetric.txt','r')
        line = ' '
        foot_transition_model1 = []

        while(True):
            line = f.readline()
            if(line == ''):
                break
            foot_transition_model1.append((float(line[0:5]),float(line[6:11]),float(line[12:17])))

        f.close()
        print('Done.')
    except Exception:
        raw_input('Not Found.')

    try:
        print('Load foot transition model 2...', end='')
        
        f = open('../data/escher_motion_planning_data/step_transition_model_mid_range.txt','r')
        line = ' '
        foot_transition_model2 = []

        while(True):
            line = f.readline()
            if(line == ''):
                break
            foot_transition_model2.append((float(line[0:5]),float(line[6:11]),float(line[12:17])))

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
    for i in [0]:
        structures = sample_env(env_handler, robot_obj, 'large_flat_ground_env')
                
        sample_contact_transitions(env_handler, robot_obj, hand_transition_model, foot_transition_model1, foot_transition_model2, structures, GRID_RESOLUTION, i)
        # IPython.embed()

    # IPython.embed()
    

if __name__ == "__main__":
    main()
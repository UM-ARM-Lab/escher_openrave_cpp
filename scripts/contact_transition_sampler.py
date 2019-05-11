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
from node import node, manip_dict


# def extract_env_feature():

def sample_contact_transitions(env_handler, robot_obj):
    # assume the robot is at (x,y) = (0,0), we sample 3 kinds of orientation (0, 30, 60)
    # other orientations are just those 3 orientations plus 90*n, so we do not need to sample them.

    # first check what are the available hand contacts for left and right hand

    # for each combination of hand contacts, find all the foot projections available to make the torso pose to be (0,0,theta)
    # here we get a set of initial contact combinations that are with torso pose (0,0,theta)

    # branch contacts for each one of them, and record the torso pose transition

def sample_env(robot_obj, surface_source):
    env_handler.update_environment(robot_obj, surface_source)

def main(robot_name='athena'): # for test

    ### Initialize the environment handler
    rave.raveLogInfo('Load the Environment Handler.')
    env_handler = environment_handler()
    env = env_handler.env
    strctures = env_handler.structures

    ### Set up the collision checker
    fcl = rave.RaveCreateCollisionChecker(env, "fcl_")
    if fcl is not None:
        env.SetCollisionChecker(fcl)
    else:
        print("FCL Not installed, falling back to ode")
        env.SetCollisionChecker(rave.RaveCreateCollisionChecker(env, "ode"))

    ### Construct the hand transition model
    hand_transition_model = []
    hand_pitch = [-30.0,-20.0,-10.0,0.0,10.0,20.0,30.0,40.0,50.0,60.0]
    hand_yaw = [-20.0,0.0,20.0]
    for pitch in hand_pitch:
        for yaw in hand_yaw:
            hand_transition_model.append((pitch,yaw))
    hand_transition_model.append((-99.0,-99.0))

    ### Load the step transition model
    try:
        print('Load step_transition_model...', end='')
        f = open('../data/escher_motion_planning_data/step_transition_model_mid_range_2.txt','r')
        line = ' '
        step_transition_model = []

        while(True):
            line = f.readline()
            if(line == ''):
                break
            step_transition_model.append((float(line[0:5]),float(line[6:11]),float(line[12:17])))

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
    while(True):
        sample_env(robot_obj, 'dynopt_test_6')


if __name__ == "__main__":
    main()

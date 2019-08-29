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
import multiprocessing

# Local Imports
import load_escher
import load_athena
import load_hermes_full

# from config_parameter import *
from transformation_conversion import *
from environment_handler_2 import environment_handler
from escher_openrave_cpp_wrapper import escher_openrave_cpp_wrapper

def main(start_env_id=0,
         end_env_id=9999,
         robot_name='hermes_full'):

    escher_planning_data_path = '../data/escher_motion_planning_data/'

    ### Initialize the environment handler
    rave.raveLogInfo('Load the Environment Handler.')

    env = rave.Environment()  # create openrave environment
    env.SetViewer('qtcoin')  # attach viewer (optional)

    fcl = rave.RaveCreateCollisionChecker(env, "fcl_")
    if fcl is not None:
        env.SetCollisionChecker(fcl)
    else:
        print("FCL Not installed, falling back to ode")
        env.SetCollisionChecker(rave.RaveCreateCollisionChecker(env, "ode"))

    ### Construct the hand transition model
    hand_transition_model = []
    # hand_pitch = [-100.0,-90.0,-80.0,-70.0,-60.0,-50.0,-40.0,-30.0,-20.0,-10.0,0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0]
    # hand_pitch = [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0]
    hand_pitch = [10.0,20.0,30.0,40.0,50.0,60.0]
    # hand_yaw = [0.0]
    hand_yaw = [-20.0,0.0,20.0]
    for pitch in hand_pitch:
        for yaw in hand_yaw:
            hand_transition_model.append((pitch,yaw))
    hand_transition_model.append((-99.0,-99.0))

    ### Load the step transition model
    try:
        print('Load step_transition_model...', end='')
        # f = open(escher_planning_data_path + 'step_transition_model_ik_verified.txt','r')
        # f = open(escher_planning_data_path + 'step_transition_model_wide_range.txt','r')
        f = open(escher_planning_data_path + 'step_transition_model_mid_range_2.txt','r')
        # f = open(escher_planning_data_path + 'step_transition_model_test.txt','r')
        # f = open(escher_planning_data_path + 'step_transition_model_mid_range_straight.txt','r')
        # f = open(escher_planning_data_path + 'step_transition_model_straight_v3.txt','r')
        # f = open(escher_planning_data_path + 'step_transition_model_straight_dynopt_test.txt','r')
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

    ########################################################################
    # load and initialize the robot
    rave.raveLogInfo('Load and Initialize the Robot.')

    if robot_name == 'escher':
        escher = load_escher.escher(env)
    elif robot_name == 'athena':
        escher = load_athena.athena(env)
    elif robot_name == 'hermes_full':
        escher = load_hermes_full.hermes_full(env)

    # Initialize Escher C++ interface
    escher_cpp = escher_openrave_cpp_wrapper(env)

    env_id = start_env_id

    # IPython.embed()

    while (env_id <= end_env_id):
        rave.raveLogInfo('Initialize the Robot and the C++ Interface.')
        # iniitialize the robot
        escher.robot.SetDOFValues(escher.OriginalDOFValues)

        rave.raveLogInfo('Start running C++ interface.')

        escher_cpp.SendStartCollectDynamicsOptimizationData(robot_name=robot_name,
                                                            escher=escher,
                                                            foot_transition_model=step_transition_model,
                                                            hand_transition_model=hand_transition_model,
                                                            check_zero_step_capturability=False,
                                                            check_one_step_capturability=True,
                                                            specified_motion_code=8,
                                                            check_contact_transition_feasibility=False,
                                                            sample_feet_only_state=True,
                                                            sample_feet_and_one_hand_state=True,
                                                            sample_feet_and_two_hands_state=True,
                                                            thread_num=1,
                                                            contact_sampling_iteration=200,
                                                            # thread_num=multiprocessing.cpu_count(),
                                                            planning_id=env_id,
                                                            printing=True)

        env_id += 1


if __name__ == "__main__":

    start_env_id = 0
    end_env_id = 9999


    i = 1
    while i < len(sys.argv):

        command = sys.argv[i]

        i += 1

        if command == 'start_env_id':
            start_env_id = int(sys.argv[i])
        elif command == 'end_env_id':
            end_env_id = int(sys.argv[i])
        else:
            print('Invalid command: %s. Abort.'%(command))
            sys.exit()

        i += 1

    print('Escher Motion Sampler Command:')
    print('start_env_id: %d'%(start_env_id))
    print('end_env_id: %d'%(end_env_id))

    main(start_env_id = start_env_id,
         end_env_id = end_env_id)


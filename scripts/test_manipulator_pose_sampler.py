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

# from config_parameter import *
from transformation_conversion import *
from environment_handler_2 import environment_handler
from escher_openrave_cpp_wrapper import escher_openrave_cpp_wrapper

def main():

    ### Initialize the environment handler
    rave.raveLogInfo('Load the Environment Handler.')
    env_handler = environment_handler()
    env = env_handler.env

    # load and initialize the robot
    rave.raveLogInfo('Load and Initrialize the Robot.')

    escher = load_athena.athena(env)
    # escher = load_escher.escher(env)

    escher.body_collision_box.SetTransform(np.eye(4, dtype=float))
    escher.move_body_to_collision_box_transform(np.eye(4, dtype=float))
    env_handler.DrawOrientation(np.eye(4, dtype=float), size=0.5)

    escher.body_collision_box.SetTransform([[1,0,0,100],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

    # escher.robot.SetDOFValues(np.zeros(escher.robot.GetDOF()))

    # initialize the IK solver for each manipulator
    escher.robot.SetActiveManipulator('l_leg')
    ikmodel = rave.databases.inversekinematics.InverseKinematicsModel(escher.robot, iktype = rave.IkParameterizationType.Transform6D)
    # ikmodel = rave.databases.inversekinematics.InverseKinematicsModel(escher.robot, iktype = rave.IkParameterizationType.Translation3D)

    if not ikmodel.load():
        ikmodel.autogenerate()

    manip = escher.robot.GetManipulator('l_leg')

    counter = 0;
    total_time = 0;

    with env:
        while(counter < 10000):
            sampled_config = sample_configuration(escher.robot, 'l_leg')
            dof_values = escher.robot.GetDOFValues()
            for i in range(manip.GetArmDOF()):
                dof_values[manip.GetArmIndices()[i]] = sampled_config[i]
            escher.robot.SetDOFValues(dof_values)
            # ikparam = IkParameterization(manip.GetTransform()[0:3,3],ikmodel.iktype)

            start = time.time()

            config = manip.FindIKSolution(manip.GetTransform(), rave.IkFilterOptions.IgnoreSelfCollisions)

            end = time.time()

            total_time += (end-start)
            if config is not None:
                print(config)
                print('oh~~~~~')
                IPython.embed()

            counter += 1

    print(total_time*1000/10000)
    IPython.embed()

    # for ix in range(-100,100):
    #     for iy in range(-100,100):
    #         for iz in range(-100,100):

    #             target_pose = np.array([[1,0,0,ix*0.01],[0,1,0,iy*0.01],[0,0,1,iz*0.01],[0,0,0,1]])

    #             config = manip.FindIKSolution(target_pose, rave.IkFilterOptions.IgnoreSelfCollisions)


    #             if config is not None:
    #                 IPython.embed()



    IPython.embed()

def sample_configuration(robot, manip_name):
    manip = robot.GetManipulator(manip_name)
    manip_joint_indices = manip.GetArmIndices()

    lower_limits = robot.GetDOFLimits()[0][manip_joint_indices]
    higher_limits = robot.GetDOFLimits()[1][manip_joint_indices]

    sampled_config = [0] * manip.GetArmDOF()

    for i in range(manip.GetArmDOF()):
        sampled_config[i] = random.uniform(lower_limits[i], higher_limits[i])

    return sampled_config



if __name__ == "__main__":
    main()
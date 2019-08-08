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
from optim_config_generator import *
from environment_handler_2 import environment_handler
from escher_openrave_cpp_wrapper import escher_openrave_cpp_wrapper

manip_dict = {0:'l_leg',1:'r_leg',2:'l_arm',3:'r_arm'}

class node:
    def __init__(self,left_leg,right_leg,left_arm,right_arm):
        self.left_leg = [None] * len(left_leg)
        self.right_leg = [None] * len(right_leg)
        self.left_arm = [None] * len(left_arm)
        self.right_arm = [None] * len(right_arm)

        self.left_leg[0:3] = [round(i,3) for i in left_leg[0:3]]
        self.right_leg[0:3] = [round(i,3) for i in right_leg[0:3]]
        self.left_arm[0:3] = [round(i,3) for i in left_arm[0:3]]
        self.right_arm[0:3] = [round(i,3) for i in right_arm[0:3]]

        self.left_leg[3:6] = [round(i,1) for i in left_leg[3:6]]
        self.right_leg[3:6] = [round(i,1) for i in right_leg[3:6]]
        self.left_arm[3:6] = [round(i,1) for i in left_arm[3:6]]
        self.right_arm[3:6] = [round(i,1) for i in right_arm[3:6]]

    def get_left_horizontal_yaw(self):
        l_leg_rotation = rpy_to_SO3(self.left_leg[3:6])
        cy = l_leg_rotation[0:3,1]
        nx = np.cross(cy,np.array([0,0,1]))
        return (round(math.atan2(nx[1],nx[0]) * RAD2DEG,1))

    def get_right_horizontal_yaw(self):
        r_leg_rotation = rpy_to_SO3(self.right_leg[3:6])
        cy = r_leg_rotation[0:3,1]
        nx = np.cross(cy,np.array([0,0,1]))
        return (round(math.atan2(nx[1],nx[0]) * RAD2DEG,1))

    def get_virtual_body_yaw(self):
        left_horizontal_yaw = self.get_left_horizontal_yaw()
        right_horizontal_yaw = self.get_right_horizontal_yaw()

        # return ((left_horizontal_yaw + right_horizontal_yaw)/2.0)
        return angle_mean(left_horizontal_yaw,right_horizontal_yaw)

    def get_manip_pose(self, manip_descriptor):
        if isinstance(manip_descriptor, int):
            if manip_descriptor >= len(manip_dict):
                rave.raveLogError('Invalid manipulator index: %d'%(manip_descriptor))
                raw_input()

            pose_list = [self.left_leg, self.right_leg, self.left_arm, self.right_arm]
            return pose_list[manip_descriptor]

        elif isinstance(manip_descriptor, basestring):
            if manip_descriptor is 'l_leg':
                return self.left_leg
            elif manip_descriptor is 'r_leg':
                return self.right_leg
            elif manip_descriptor is 'l_arm':
                return self.left_arm
            elif manip_descriptor is 'r_arm':
                return self.right_arm
            else:
                rave.raveLogError('Invalid manipulator name: %s'%(manip_descriptor))
                raw_input()

        rave.raveLogError('Unknown manipulator descriptor type.')

    def get_mean_feet_xyzrpy(self):
        mean_yaw = self.get_virtual_body_yaw()
        mean_x = (self.left_leg[0] + self.right_leg[0])/2.0
        mean_y = (self.left_leg[1] + self.right_leg[1])/2.0
        mean_z = (self.left_leg[2] + self.right_leg[2])/2.0

        return [mean_x, mean_y, mean_z, 0, 0, mean_yaw]

class map_grid_dim:
    def __init__(self,min_x,max_x,min_y,max_y,resolution):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.resolution = resolution

    def update_grid_boundary(self,structures):
        env_min_x = sys.maxint
        env_max_x = -sys.maxint
        env_min_y = sys.maxint
        env_max_y = -sys.maxint

        for struct in structures:
            for vertex in struct.vertices:
                env_min_x = min(vertex[0], env_min_x)
                env_max_x = max(vertex[0], env_max_x)
                env_min_y = min(vertex[1], env_min_y)
                env_max_y = max(vertex[1], env_max_y)

        self.min_x = math.floor(env_min_x/self.resolution)*self.resolution - self.resolution/2.0
        self.max_x = math.ceil(env_max_x/self.resolution)*self.resolution + self.resolution/2.0
        self.min_y = math.floor(env_min_y/self.resolution)*self.resolution - self.resolution/2.0
        self.max_y = math.ceil(env_max_y/self.resolution)*self.resolution + self.resolution/2.0


def main(meta_path_generation_method='all_planning',
         path_segmentation_generation_type = 'motion_mode_and_traversability_segmentation',
         traversability_select_criterion='mean',
         traversability_threshold = 0.3,
         environment_path='environment',
         surface_source='mix_test_environment_2',
         log_file_name='exp_result.txt',
         start_env_id=0,
         end_env_id=9999,
         recording_data=False,
         use_env_transition_bias=False,
         load_planning_result=False,
         robot_name='hermes_full'):

    ### Initialize the ros node
    rave.raveLogInfo('Using %s method...'%(meta_path_generation_method))

    escher_planning_data_path = '../data/escher_motion_planning_data/'

    ### Initialize the environment handler
    rave.raveLogInfo('Load the Environment Handler.')
    env_handler = environment_handler(enable_viewer=True)
    env = env_handler.env
    structures = env_handler.structures
    env_map_grid_dim = map_grid_dim(0, 0, 0, 0, 0.135)

    ### Construct the hand transition model
    hand_transition_model = []
    # # hand_pitch = [-100.0,-90.0,-80.0,-70.0,-60.0,-50.0,-40.0,-30.0,-20.0,-10.0,0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0]
    # # hand_pitch = [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0]
    # hand_pitch = [10.0,20.0,30.0,40.0,50.0,60.0]
    # # hand_yaw = [0.0]
    # hand_yaw = [-20.0,0.0,20.0]
    # for pitch in hand_pitch:
    #     for yaw in hand_yaw:
    #         hand_transition_model.append((pitch,yaw))
    hand_transition_model.append((-99.0,-99.0))

    ### Construct the disturbance rejection hand transition model
    disturbance_rejection_hand_transition_model = []
    # hand_pitch = [-100.0,-90.0,-80.0,-70.0,-60.0,-50.0,-40.0,-30.0,-20.0,-10.0,0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0]
    # hand_pitch = [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0]
    # hand_pitch = [10.0,20.0,30.0,40.0,50.0,60.0]
    # hand_pitch = [-40.0,-35.0,-30.0,-25.0,-20.0,-15.0,-10.0,-5.0,0.0,0.5,10.0,15.0,20.0,25.0,30.0,35.0,40.0]
    hand_pitch = [0.0,-5.0,5.0,-10.0,10.0,-15.0,15.0,-20.0,20.0,-25.0,25.0,-30.0,30.0,-35.0,35.0,-40.0,40.0]
    # hand_yaw = [0.0]
    hand_yaw = [-20.0,-15.0,-10.0,-5.0,0.0,5.0,10.0,15.0,20.0]
    # hand_yaw = [0.0,-5.0,5.0,-10.0,10.0,-15.0,15.0,-20.0,20.0]
    for pitch in hand_pitch:
        for yaw in hand_yaw:
            disturbance_rejection_hand_transition_model.append((pitch,yaw))
    # disturbance_rejection_hand_transition_model.append((-99.0,-99.0))

    ### Load the step transition model
    try:
        print('Load step_transition_model...', end='')
        # f = open(escher_planning_data_path + 'step_transition_model_ik_verified.txt','r')
        # f = open(escher_planning_data_path + 'step_transition_model_wide_range.txt','r')
        # f = open(escher_planning_data_path + 'step_transition_model_mid_range.txt','r')
        f = open(escher_planning_data_path + 'step_transition_model_mid_range_2.txt','r')
        # f = open(escher_planning_data_path + 'step_transition_model_short_range_straight.txt','r')
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


    ### Load the capture step transition model
    try:
        print('Load step_transition_model...', end='')
        # f = open(escher_planning_data_path + 'step_transition_model_ik_verified.txt','r')
        # f = open(escher_planning_data_path + 'step_transition_model_wide_range.txt','r')
        # f = open(escher_planning_data_path + 'step_transition_model_mid_range.txt','r')
        f = open(escher_planning_data_path + 'step_transition_model_mid_range_2.txt','r')
        # f = open(escher_planning_data_path + 'step_transition_model_short_range_straight.txt','r')
        # f = open(escher_planning_data_path + 'step_transition_model_mid_range_straight.txt','r')
        # f = open(escher_planning_data_path + 'step_transition_model_straight_v3.txt','r')
        # f = open(escher_planning_data_path + 'step_transition_model_straight_dynopt_test.txt','r')
        line = ' '
        disturbance_rejection_step_transition_model = []

        while(True):
            line = f.readline()
            if(line == ''):
                break
            disturbance_rejection_step_transition_model.append((float(line[0:5]),float(line[6:11]),float(line[12:17])))

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

    env_id = start_env_id

    while (env_id <= end_env_id):

        rave.raveLogInfo('Initialize the Environment.')

        # Remove the obstacles to avoid duplicated adding obstacles in next run
        for struct in structures:
            if struct.kinbody is not None:
                escher.env.Remove(struct.kinbody)
                struct.kinbody = None

        env_handler.update_environment(escher, escher_planning_data_path + environment_path + '/env_' + str(env_id), surface_source)

        goal_x = env_handler.goal_x
        goal_y = env_handler.goal_y
        goal_theta = 0

        goal = [goal_x, goal_y, goal_theta]

        env_handler.DrawRegion(env_handler.env, xyzrpy_to_SE3([goal_x,goal_y,0.02,0,0,goal_theta]), 0.2)

        # set up the structures
        structures = env_handler.structures
        structures_dict = {}
        for struct in structures:
            structures_dict[struct.id] = struct
        env_map_grid_dim.update_grid_boundary(structures)

        rave.raveLogInfo('Initialize the Robot and the C++ Interface.')
        # iniitialize the robot
        escher.robot.SetDOFValues(escher.OriginalDOFValues)

        # Initialize Escher C++ interface
        escher_cpp = escher_openrave_cpp_wrapper(env)

        rave.raveLogInfo('Start running C++ interface.')
        # Construct the initial node

        initial_left_leg = [0.025,0.1,0.0,0,0,0]
        initial_right_leg = [0.025,-0.1,0.0,0,0,0]
        initial_left_arm = [-99.0,-99.0,-99.0,-99.0,-99.0,-99.0]
        initial_right_arm = [-99.0,-99.0,-99.0,-99.0,-99.0,-99.0]

        initial_node = node(initial_left_leg, initial_right_leg, initial_left_arm, initial_right_arm)

        # IPython.embed()

        disturbance_samples = []
        robot_mass = 63.47

        # disturbance_magnitude = 0.5 * robot_mass
        # disturbance_sample_num = 8
        # for i in range(disturbance_sample_num):
        #     disturbance_samples.append([disturbance_magnitude * math.cos(2*i*math.pi/disturbance_sample_num),
        #                                 disturbance_magnitude * math.sin(2*i*math.pi/disturbance_sample_num),
        #                                 0, 0, 0, 0, 1.0/disturbance_sample_num])

        # disturbance_magnitude = 0.5 * robot_mass
        # disturbance_samples.append([0, disturbance_magnitude, 0, 0, 0, 0, 0.5])
        # disturbance_samples.append([0, -disturbance_magnitude, 0, 0, 0, 0, 0.5])

        # disturbance_samples.append([0, 0.2 * robot_mass, 0, 0, 0, 0, 0.25])
        # disturbance_samples.append([0, 0.5 * robot_mass, 0, 0, 0, 0, 0.5])
        # disturbance_samples.append([0, 0.8 * robot_mass, 0, 0, 0, 0, 0.25])

        # disturbance_samples.append([0, 0.2 * robot_mass, 0, 0, 0, 0, 0.25])
        # disturbance_samples.append([0, 0.4 * robot_mass, 0, 0, 0, 0, 0.5])
        # disturbance_samples.append([0, 0.6 * robot_mass, 0, 0, 0, 0, 0.25])

        # disturbance_samples.append([0, 0.2 * robot_mass, 0, 0, 0, 0, 0.15])
        disturbance_samples.append([0, 0.5 * robot_mass, 0, 0, 0, 0, 0.8])
        disturbance_samples.append([0, 0.8 * robot_mass, 0, 0, 0, 0, 0.2])

        # disturbance_samples.append([0, 0.8 * robot_mass, 0, 0, 0, 0, 1.0])

        # disturbance_samples.append([0, 0.5 * robot_mass, 0, 0, 0, 0, 0.25])
        # disturbance_samples.append([0, 0.7 * robot_mass, 0, 0, 0, 0, 0.5])
        # disturbance_samples.append([0, 0.9 * robot_mass, 0, 0, 0, 0, 0.25])

        # disturbance_samples.append([0, 0.5 * robot_mass, 0, 0, 0, 0, 1.0])
        # disturbance_samples.append([0, 0.8 * robot_mass, 0, 0, 0, 0, 0.5])

        # disturbance_samples.append([0, 0.8 * robot_mass, 0, 0, 0, 0, 1.0])
        # disturbance_samples.append([0, 0.8 * robot_mass, 0, 0, 0, 0, 0.5])

        # disturbance_magnitude = 0.5 * robot_mass
        # disturbance_samples.append([disturbance_magnitude, 0, 0, 0, 0, 0, 0.5])
        # disturbance_samples.append([-disturbance_magnitude, 0, 0, 0, 0, 0, 0.5])

        # disturbance_samples.append([0, 0.2*robot_mass, 0, 0, 0, 0, 0.25])
        # disturbance_samples.append([0, 0.4*robot_mass, 0, 0, 0, 0, 0.25])
        # disturbance_samples.append([0, 0.6*robot_mass, 0, 0, 0, 0, 0.25])
        # disturbance_samples.append([0, 0.8*robot_mass, 0, 0, 0, 0, 0.25])

        # disturbance_sample_num = 1
        # for i in range(disturbance_sample_num):
        #     disturbance_samples.append([0, 0, 0, 0, 0, 0, 1.0])


        # # for collect data
        # escher_cpp.SendStartPlanningFromScratch(robot_name=robot_name,
        #                                         escher=escher,
        #                                         initial_state=initial_node,
        #                                         goal=goal,
        #                                         epsilon=0.3,
        #                                         foot_transition_model=step_transition_model,
        #                                         hand_transition_model=hand_transition_model,
        #                                         structures=structures,
        #                                         map_grid_dim=env_map_grid_dim,
        #                                         goal_radius=0.2,
        #                                         time_limit=300.0,
        #                                         planning_heuristics='euclidean',
        #                                         branching_method='contact_projection',
        #                                         output_first_solution=False,
        #                                         goal_as_exact_poses=False,
        #                                         use_dynamics_planning=True,
        #                                         use_learned_dynamics_model=False,
        #                                         enforce_stop_in_the_end=False,
        #                                         check_zero_step_capturability=False,
        #                                         check_one_step_capturability=False,
        #                                         check_contact_transition_feasibility=True,
        #                                         disturbance_samples=disturbance_samples,
        #                                         thread_num=1,
        #                                         # thread_num=multiprocessing.cpu_count(),
        #                                         planning_id=env_id,
        #                                         printing=False)

        escher.robot.SetTransform([[1,0,0,100],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

        # for planning test
        escher_cpp.SendStartPlanningFromScratch(robot_name=robot_name,
                                                escher=escher,
                                                initial_state=initial_node,
                                                goal=goal,
                                                epsilon=0.0,
                                                foot_transition_model=step_transition_model,
                                                hand_transition_model=hand_transition_model,
                                                disturbance_rejection_foot_transition_model=disturbance_rejection_step_transition_model,
                                                disturbance_rejection_hand_transition_model=disturbance_rejection_hand_transition_model,
                                                structures=structures,
                                                map_grid_dim=env_map_grid_dim,
                                                goal_radius=0.2,
                                                time_limit=10.0,
                                                planning_heuristics='euclidean',
                                                branching_method='contact_projection',
                                                output_first_solution=False,
                                                goal_as_exact_poses=False,
                                                use_dynamics_planning=True,
                                                use_learned_dynamics_model=True,
                                                enforce_stop_in_the_end=False,
                                                check_zero_step_capturability=True,
                                                check_one_step_capturability=True,
                                                disturbance_samples=disturbance_samples,
                                                thread_num=1,
                                                # thread_num=multiprocessing.cpu_count(),
                                                planning_id=env_id,
                                                printing=False)

        generate_Objects_cf("/home/yuchi/amd_workspace_video/workspace/src/catkin/humanoids/humanoid_control/motion_planning/momentumopt_sl/momentumopt_hermes_full/config/push_recovery/", structures)

        env_id += 1


if __name__ == "__main__":

    meta_path_generation_method = 'all_planning'
    path_segmentation_generation_type = 'motion_mode_and_traversability_segmentation'
    traversability_threshold = 0.3
    traversability_select_criterion = 'mean'
    surface_source = 'mix_test_environment_2'
    environment_path = 'environment'
    log_file_name = 'exp_result.txt'
    start_env_id = 0
    end_env_id = 9999
    recording_data = False
    use_env_transition_bias = False
    load_planning_result = False

    i = 1
    while i < len(sys.argv):

        command = sys.argv[i]

        i += 1

        if command == 'meta_path_generation_method':
            meta_path_generation_method = sys.argv[i]
        elif command == 'path_segmentation_generation_type':
            path_segmentation_generation_type = sys.argv[i]
        elif command == 'traversability_threshold':
            traversability_threshold = float(sys.argv[i])
        elif command == 'surface_source':
            surface_source = sys.argv[i]
        elif command == 'log_file_name':
            log_file_name = sys.argv[i]
        elif command == 'start_env_id':
            start_env_id = int(sys.argv[i])
        elif command == 'end_env_id':
            end_env_id = int(sys.argv[i])
        elif command == 'recording_data':
            if int(sys.argv[i]) == 0:
                recording_data = False
            else:
                recording_data = True
        elif command == 'use_env_transition_bias':
            if int(sys.argv[i]) == 0:
                use_env_transition_bias = False
            else:
                use_env_transition_bias = True
        elif command == 'load_planning_result':
            if int(sys.argv[i]) == 0:
                load_planning_result = False
            else:
                load_planning_result = True
        elif command == 'environment_path':
            environment_path = sys.argv[i]
        elif command == 'traversability_select_criterion':
            if sys.argv[i] == 'mean':
                traversability_select_criterion = 'mean'
            elif sys.argv[i] == 'max':
                traversability_select_criterion = 'max'
            else:
                print('Unknown traversability select criterion: %s. Abort.'%(sys.argv[i]))
                sys.exit()
        else:
            print('Invalid command: %s. Abort.'%(command))
            sys.exit()

        i += 1

    print('Escher Motion Planner Command:')
    print('meta_path_generation_method: %s'%(meta_path_generation_method))
    print('path_segmentation_generation_type: %s'%(path_segmentation_generation_type))
    print('traversability_select_criterion: %s'%(traversability_select_criterion))
    print('traversability_threshold: %5.2f'%(traversability_threshold))
    print('environment_path: %s'%(environment_path))
    print('surface_source: %s'%(surface_source))
    print('log_file_name: %s'%(log_file_name))
    print('start_env_id: %d'%(start_env_id))
    print('end_env_id: %d'%(end_env_id))
    print('recording_data: %r'%(recording_data))
    print('use_env_transition_bias: %r'%(use_env_transition_bias))
    print('load_planning_result: %r'%(load_planning_result))

    main(meta_path_generation_method = meta_path_generation_method,
         path_segmentation_generation_type = path_segmentation_generation_type,
         traversability_select_criterion = traversability_select_criterion,
         traversability_threshold = traversability_threshold,
         environment_path = environment_path,
         surface_source = surface_source,
         log_file_name = log_file_name,
         start_env_id = start_env_id,
         end_env_id = end_env_id,
         recording_data = recording_data,
         use_env_transition_bias = use_env_transition_bias,
         load_planning_result = load_planning_result)

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

from transformation_conversion import *
from environment_handler_2 import environment_handler
from structures_2 import *
from map_grid import map_grid_dim
from node import *

# foot projection
def foot_projection(robot_obj, node, structures, mapping_manip='all'):
    checking_left_foot = False
    checking_right_foot = False

    if(mapping_manip is 'all'):
        if(node.prev_move_manip is not None):
            if(node.prev_move_manip == LEFT_LEG): # move left leg
                checking_left_foot = True
            elif(node.prev_move_manip == RIGHT_LEG): # move right leg
                checking_right_foot = True
        else:
            checking_left_foot = True
            checking_right_foot = True

    elif(mapping_manip is 'l_leg'):
        checking_left_foot = True
    elif(mapping_manip is 'r_leg'):
        checking_right_foot = True

    if(checking_left_foot):
        left_foot_height = sys.float_info.min
        for struct in structures:
            left_foot_projection = struct.projection(robot_obj,np.array([[node.left_leg[0]],[node.left_leg[1]],[sys.float_info.max]]),np.array([[0],[0],[-1]],dtype=float),node.get_left_horizontal_yaw(),'foot')
            
            if(left_foot_projection is not None and left_foot_height < left_foot_projection[2]):
                left_foot_height = left_foot_projection[2]
                left_foot_projection_xyzrpy = SE3_to_xyzrpy(left_foot_projection)
                node.left_leg[0:3] = [round(i,3) for i in left_foot_projection_xyzrpy[0:3]]
                node.left_leg[3:6] = [round(i,1) for i in left_foot_projection_xyzrpy[3:6]]

        if left_foot_height == sys.float_info.min:
            return False # No projection

    if(checking_right_foot):
        right_foot_height = sys.float_info.min
        for struct in structures:
            right_foot_projection = struct.projection(robot_obj,np.array([[node.right_leg[0]],[node.right_leg[1]],[sys.float_info.max]]),np.array([[0],[0],[-1]],dtype=float),node.get_right_horizontal_yaw(),'foot')
            
            if(right_foot_projection is not None and right_foot_height < right_foot_projection[2]):
                right_foot_height = left_foot_projection[2]
                right_foot_projection_xyzrpy = SE3_to_xyzrpy(right_foot_projection)
                node.right_leg[0:3] = [round(i,3) for i in right_foot_projection_xyzrpy[0:3]]
                node.right_leg[3:6] = [round(i,1) for i in right_foot_projection_xyzrpy[3:6]]

        if right_foot_height == sys.float_info.min:
            return False # No projection

    return True


# hand projection
def hand_projection(robot_obj, manip, arm_orientation, node, structures):
    current_mean_leg_x = (node.left_leg[0] + node.right_leg[0])/2.0
    current_mean_leg_y = (node.left_leg[1] + node.right_leg[1])/2.0
    current_mean_leg_z = (node.left_leg[2] + node.right_leg[2])/2.0
    current_virtual_body_yaw = node.get_virtual_body_yaw()
    current_arm_orientation = [0,0]

    # assuming waist is fixed
    if(manip == LEFT_ARM):
        relative_shoulder_position = [0, robot_obj.shoulder_w/2.0, robot_obj.shoulder_z]
        current_arm_orientation[0] = current_virtual_body_yaw + 90.0 - arm_orientation[0]
    elif(manip == RIGHT_ARM):
        relative_shoulder_position = [0, -robot_obj.shoulder_w/2.0, robot_obj.shoulder_z]
        current_arm_orientation[0] = current_virtual_body_yaw - 90.0 + arm_orientation[0]

    current_arm_orientation[1] = arm_orientation[1]

    current_shoulder_x = current_mean_leg_x + math.cos(current_virtual_body_yaw*DEG2RAD) * relative_shoulder_position[0] - math.sin(current_virtual_body_yaw*DEG2RAD) * relative_shoulder_position[1]
    current_shoulder_y = current_mean_leg_y + math.sin(current_virtual_body_yaw*DEG2RAD) * relative_shoulder_position[0] + math.cos(current_virtual_body_yaw*DEG2RAD) * relative_shoulder_position[1]
    current_shoulder_z = current_mean_leg_z + relative_shoulder_position[2]

    current_shoulder_position = np.array([[current_shoulder_x],
                                          [current_shoulder_y],
                                          [current_shoulder_z],
                                          [1]])    

    arm_length = 9999.0
    valid_contact = False
    arm_pose = [None]*6

    for struct in structures:
        if(struct.geometry == 'trimesh' and struct.type == 'others'):
            arm_yaw = current_arm_orientation[0] * DEG2RAD
            arm_pitch = current_arm_orientation[1] * DEG2RAD

            arm_ray = np.array([[math.cos(arm_yaw) * math.cos(arm_pitch)],
                                [math.sin(arm_yaw) * math.cos(arm_pitch)],
                                [math.sin(arm_pitch)]])

            if(manip == LEFT_ARM):
                contact_transform = struct.projection(robot_obj,current_shoulder_position[0:3,0:1],arm_ray,0.0,'left_hand')
            elif(manip == RIGHT_ARM):
                contact_transform = struct.projection(robot_obj,current_shoulder_position[0:3,0:1],arm_ray,0.0,'right_hand')

            if(contact_transform is not None): # exist a valid contact on a surface
                temp_arm_length = np.linalg.norm(current_shoulder_position - contact_transform[0:4,3:4])

                if(temp_arm_length < arm_length):
                    arm_length = temp_arm_length

                    if(arm_length >= robot_obj.min_arm_length and arm_length <= robot_obj.max_arm_length):
                        valid_contact = True
                        arm_pose = SE3_to_xyzrpy(contact_transform)

            else:
                translation = struct.fast_projection_global_frame(current_shoulder_position[0:3,0:1],arm_ray)

                if(translation is not None and struct.inside_polygon(translation)): # exist projection of center point, but not enough space for a contact
                    temp_arm_length = np.linalg.norm(current_shoulder_position[0:3,0:1] - translation)

                    if(temp_arm_length < arm_length):
                        arm_length = temp_arm_length
                        valid_contact = False

    if(valid_contact):
        if(manip == LEFT_ARM):
            node.left_arm[0:3] = [round(i,3) for i in arm_pose[0:3]]
            node.left_arm[3:6] = [round(i,1) for i in arm_pose[3:6]]
        elif(manip == RIGHT_ARM):
            node.right_arm[0:3] = [round(i,3) for i in arm_pose[0:3]]
            node.right_arm[3:6] = [round(i,1) for i in arm_pose[3:6]]

    return valid_contact
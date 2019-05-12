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

    if mapping_manip is 'all':
        checking_left_foot = True
        checking_right_foot = True
    elif mapping_manip is 'l_leg':
        checking_left_foot = True
    elif mapping_manip is 'r_leg':
        checking_right_foot = True

    if checking_left_foot:
        left_foot_height = -sys.float_info.max

        for struct in structures:
            left_foot_projection = struct.projection(robot_obj,np.array([[node.left_leg[0]],[node.left_leg[1]],[sys.float_info.max]]),np.array([[0],[0],[-1]],dtype=float),node.get_left_horizontal_yaw(),'foot')

            if left_foot_projection is not None and left_foot_height < left_foot_projection[2,3]:
                left_foot_height = left_foot_projection[2,3]
                left_foot_projection_xyzrpy = SE3_to_xyzrpy(left_foot_projection)
                node.left_leg[0:3] = [round(i,3) for i in left_foot_projection_xyzrpy[0:3]]
                node.left_leg[3:6] = [round(i,1) for i in left_foot_projection_xyzrpy[3:6]]

        if left_foot_height == -sys.float_info.max:
            return False # No projection

    if checking_right_foot:
        right_foot_height = -sys.float_info.max
        for struct in structures:
            right_foot_projection = struct.projection(robot_obj,np.array([[node.right_leg[0]],[node.right_leg[1]],[sys.float_info.max]]),np.array([[0],[0],[-1]],dtype=float),node.get_right_horizontal_yaw(),'foot')
            
            if right_foot_projection is not None and right_foot_height < right_foot_projection[2,3]:
                right_foot_height = right_foot_projection[2,3]
                right_foot_projection_xyzrpy = SE3_to_xyzrpy(right_foot_projection)
                node.right_leg[0:3] = [round(i,3) for i in right_foot_projection_xyzrpy[0:3]]
                node.right_leg[3:6] = [round(i,1) for i in right_foot_projection_xyzrpy[3:6]]

        if right_foot_height == -sys.float_info.max:
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
                translation = struct.projection_global_frame(current_shoulder_position[0:3,0:1],arm_ray)

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

# branching contacts
def branching(current_node, foot_transition_model, hand_transition_model, structures, robot_obj):
    child_node_list = []

    current_left_leg = copy.copy(current_node.left_leg)
    current_right_leg = copy.copy(current_node.right_leg)
    current_left_arm = copy.copy(current_node.left_arm)
    current_right_arm = copy.copy(current_node.right_arm)

    l_leg_horizontal_yaw = current_node.get_left_horizontal_yaw()
    r_leg_horizontal_yaw = current_node.get_right_horizontal_yaw()

    # only branch left side of the manipulators because we specify that in the neural network
    # when query the network in planning, we will "mirror" the state if the contact transition is using right side of the manipulators
    move_manip = [LEFT_LEG, LEFT_ARM]

    for manip in move_manip:
        # foot/leg movement
        if manip == LEFT_LEG or manip == RIGHT_LEG:

            for step in foot_transition_model:
                l_leg_x, l_leg_y, l_leg_z, l_leg_roll, l_leg_pitch, l_leg_yaw = current_left_leg
                r_leg_x, r_leg_y, r_leg_z, r_leg_roll, r_leg_pitch, r_leg_yaw = current_right_leg

                if manip == LEFT_LEG:
                    l_leg_x = r_leg_x + math.cos(r_leg_horizontal_yaw*DEG2RAD) * step[0] - math.sin(r_leg_horizontal_yaw*DEG2RAD) * step[1]
                    l_leg_y = r_leg_y + math.sin(r_leg_horizontal_yaw*DEG2RAD) * step[0] + math.cos(r_leg_horizontal_yaw*DEG2RAD) * step[1]
                    l_leg_z = sys.float_info.max
                    l_leg_roll = 0
                    l_leg_pitch = 0
                    l_leg_yaw = r_leg_horizontal_yaw + step[2]

                    l_leg_x = round(l_leg_x,3)
                    l_leg_y = round(l_leg_y,3)
                    l_leg_yaw = round(l_leg_yaw,1)

                    new_left_leg = [l_leg_x, l_leg_y, l_leg_z, l_leg_roll, l_leg_pitch, l_leg_yaw]
                    child_node = node(new_left_leg, current_right_leg, current_left_arm, current_right_arm)
                elif manip == RIGHT_LEG:
                    r_leg_x = l_leg_x + math.cos(l_leg_horizontal_yaw*DEG2RAD) * step[0] - math.sin(l_leg_horizontal_yaw*DEG2RAD) * (-step[1])
                    r_leg_y = l_leg_y + math.sin(l_leg_horizontal_yaw*DEG2RAD) * step[0] + math.cos(l_leg_horizontal_yaw*DEG2RAD) * (-step[1])
                    r_leg_z = sys.float_info.max
                    r_leg_roll = 0
                    r_leg_pitch = 0
                    r_leg_yaw = l_leg_horizontal_yaw - step[2]

                    r_leg_x = round(r_leg_x,3)
                    r_leg_y = round(r_leg_y,3)
                    r_leg_yaw = round(r_leg_yaw,1)

                    new_right_leg = [r_leg_x, r_leg_y, r_leg_z, r_leg_roll, r_leg_pitch, r_leg_yaw]
                    child_node = node(current_left_leg, new_right_leg, current_left_arm, current_right_arm)

                if foot_projection(robot_obj, child_node, structures):
                    child_node.prev_move_manip = manip
                    if child_node.node_feasibile(robot_obj):
                        child_node_list.append(child_node)

        # hand/arm movement
        elif manip == LEFT_ARM or manip == RIGHT_ARM:

            for arm_orientation in hand_transition_model:
                contact_exist = True
                child_node = node(current_left_leg, current_right_leg, current_left_arm, current_right_arm)

                if(arm_orientation[0] == -99.0 and arm_orientation[1] == -99.0):
                    if(current_node.manip_in_contact('l_arm') and manip == LEFT_ARM):
                        child_node.left_arm = copy.copy(no_contact)
                    elif(current_node.manip_in_contact('r_arm') and manip == RIGHT_ARM):
                        child_node.right_arm = copy.copy(no_contact)
                    else:
                        continue
                else:
                    contact_exist = hand_projection(robot_obj, manip, arm_orientation, child_node, structures)

                if contact_exist:
                    child_node.prev_move_manip = manip
                    if child_node.node_feasibile(robot_obj):
                        child_node_list.append(child_node)

    return child_node_list
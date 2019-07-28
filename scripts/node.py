import math
import numpy as np
import openravepy as rave

from transformation_conversion import *

LEFT_LEG = 0; RIGHT_LEG = 1; LEFT_ARM = 2; RIGHT_ARM = 3
manip_dict = {LEFT_LEG:'l_leg', RIGHT_LEG:'r_leg', LEFT_ARM:'l_arm', RIGHT_ARM:'r_arm'}
manip_inv_dict = {'l_leg':LEFT_LEG, 'r_leg':RIGHT_LEG, 'l_arm':LEFT_ARM, 'r_arm':RIGHT_ARM}
no_contact = [-99.0, -99.0, -99.0, -99.0, -99.0, -99.0]

class node:
    def __init__(self,left_leg,right_leg,left_arm,right_arm):
        self.left_leg = [None] * len(left_leg)
        self.right_leg = [None] * len(right_leg)
        self.left_arm = [None] * len(left_arm)
        self.right_arm = [None] * len(right_arm)

        self.left_leg[0:3] = [round(i,3) for i in left_leg[0:3]]
        assert(self.left_leg[2] < 1.0)
        self.right_leg[0:3] = [round(i,3) for i in right_leg[0:3]]
        assert(self.right_leg[2] < 1.0)
        self.left_arm[0:3] = [round(i,3) for i in left_arm[0:3]]
        self.right_arm[0:3] = [round(i,3) for i in right_arm[0:3]]

        self.left_leg[3:6] = [round(i,1) for i in left_leg[3:6]]
        self.right_leg[3:6] = [round(i,1) for i in right_leg[3:6]]
        self.left_arm[3:6] = [round(i,1) for i in left_arm[3:6]]
        self.right_arm[3:6] = [round(i,1) for i in right_arm[3:6]]

        self.manip_pose_list = [self.left_leg, self.right_leg, self.left_arm, self.right_arm]

        self.prev_move_manip = None

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

    def get_manip_pose(self, manip):
        self.manip_pose_list = [self.left_leg, self.right_leg, self.left_arm, self.right_arm]
        if isinstance(manip, int):
            if manip < len(manip_dict):
                return self.manip_pose_list[manip]
            else:
                rave.raveLogError('Invalid manipulator index: %d'%(manip))
                raw_input()
        elif isinstance(manip, basestring):
            if manip in manip_inv_dict:
                return self.manip_pose_list[manip_inv_dict[manip]]
            else:
                rave.raveLogError('Invalid manipulator name: %s'%(manip))
                raw_input()
        else:
            rave.raveLogError('Unknown manipulator descriptor type.')
            raw_input()

    def get_mean_feet_xyzrpy(self):
        mean_yaw = self.get_virtual_body_yaw()
        mean_x = (self.left_leg[0] + self.right_leg[0])/2.0
        mean_y = (self.left_leg[1] + self.right_leg[1])/2.0
        mean_z = (self.left_leg[2] + self.right_leg[2])/2.0

        return [mean_x, mean_y, mean_z, 0, 0, mean_yaw] # why don't need roll and pitch

    def manip_in_contact(self, manip): # if manip in contact, return true
        self.manip_pose_list = [self.left_leg, self.right_leg, self.left_arm, self.right_arm]
        if isinstance(manip, int):
            if manip < len(manip_dict):
                return self.manip_pose_list[manip][0] != -99.0
            else:
                rave.raveLogError('Invalid manipulator index: %d'%(manip))
                raw_input()
        elif isinstance(manip, basestring): # basestring is the superclass for str and unicode
            if manip in manip_inv_dict:
                return self.manip_pose_list[manip_inv_dict[manip]][0] != -99.0
            else:
                rave.raveLogError('Invalid manipulator name: %s'%(manip))
                raw_input()
        else:
            rave.raveLogError('Unknown manipulator descriptor type.')
            raw_input()

    def get_virtual_body_pose(self): # map from a node (stance) to a pose in SE(2)
        mean_feet_pose = self.get_mean_feet_xyzrpy()
        mean_feet_position = np.array(mean_feet_pose[0:2])
        virtual_body_yaw = self.get_virtual_body_yaw() # orientation (same as mean_feet_pose[5]?)

        orientation_rotation_matrix = rpy_to_SO3([0, 0, virtual_body_yaw])
        orientation_unit_vec = np.array([math.cos(virtual_body_yaw*DEG2RAD), math.sin(virtual_body_yaw*DEG2RAD)])

        foot_contact_num = 2
        hand_contact_num = 0
        rotated_x = 0 # initialize to the sum of left_feet_x and right_feet_x, which is also 0 because the origin is the mean_feet_x

        if(self.manip_in_contact('l_arm')):
            rotated_x += np.dot(np.array(self.get_manip_pose('l_arm')[0:2])-mean_feet_position, orientation_unit_vec)
            hand_contact_num += 1

        if(self.manip_in_contact('r_arm')):
            rotated_x += np.dot(np.array(self.get_manip_pose('r_arm')[0:2])-mean_feet_position, orientation_unit_vec)
            hand_contact_num += 1

        if hand_contact_num != 0:
            rotated_x = float(rotated_x) / (hand_contact_num + foot_contact_num)
        
        virtual_body_position = np.round(np.atleast_2d(mean_feet_position).T + np.dot(orientation_rotation_matrix[0:2,0:2], np.array([[rotated_x],[0]])),3)

        return [virtual_body_position[0,0], virtual_body_position[1,0], 0, 0, 0, virtual_body_yaw]

    def node_feasibile(self, robot_obj):
        # check if feet are too far away (should not be possible, but check it anyway)
        if np.linalg.norm(np.array(self.left_leg[0:3]) - np.array(self.right_leg[0:3])) > 3.0:
            rave.raveLogError('Large distance between feet should not be possible.')
            print("left_leg: ", self.left_leg)
            print("right_leg: ", self.right_leg)
            raw_input()

        # check if hands are too far away from feet mean position
        mean_feet_pose = self.get_mean_feet_xyzrpy()
        virtual_body_yaw = self.get_virtual_body_yaw()
        virtual_body_yaw_rad = virtual_body_yaw * DEG2RAD

        if self.manip_in_contact('l_arm'):
            relative_shoulder_position = [0, robot_obj.shoulder_w/2.0, robot_obj.shoulder_z]
            shoulder_position = np.array(mean_feet_pose[0:3])
            shoulder_position[0] += math.cos(virtual_body_yaw_rad) * relative_shoulder_position[0] - math.sin(virtual_body_yaw_rad) * relative_shoulder_position[1]
            shoulder_position[1] += math.sin(virtual_body_yaw_rad) * relative_shoulder_position[0] + math.cos(virtual_body_yaw_rad) * relative_shoulder_position[1]
            shoulder_position[2] += relative_shoulder_position[2]

            left_hand_to_shoulder_dist = np.linalg.norm(np.array(self.left_arm[0:3]) - shoulder_position)
            if left_hand_to_shoulder_dist < robot_obj.min_arm_length or left_hand_to_shoulder_dist > robot_obj.max_arm_length:
                return False

        if self.manip_in_contact('r_arm'):
            relative_shoulder_position = [0, -robot_obj.shoulder_w/2.0, robot_obj.shoulder_z]
            shoulder_position = np.array(mean_feet_pose[0:3])
            shoulder_position[0] += math.cos(virtual_body_yaw_rad) * relative_shoulder_position[0] - math.sin(virtual_body_yaw_rad) * relative_shoulder_position[1]
            shoulder_position[1] += math.sin(virtual_body_yaw_rad) * relative_shoulder_position[0] + math.cos(virtual_body_yaw_rad) * relative_shoulder_position[1]
            shoulder_position[2] += relative_shoulder_position[2]

            right_hand_to_shoulder_dist = np.linalg.norm(np.array(self.right_arm[0:3]) - shoulder_position)
            if right_hand_to_shoulder_dist < robot_obj.min_arm_length or right_hand_to_shoulder_dist > robot_obj.max_arm_length:
                return False

        return True

    def get_contact_manip_num(self):
        contact_manip_num = 0
        for manip in manip_dict:
            contact_manip_num += int(self.manip_in_contact(manip))
        return contact_manip_num



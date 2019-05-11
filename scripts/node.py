import math
import numpy as np
import openrave as rave

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
        self.right_leg[0:3] = [round(i,3) for i in right_leg[0:3]]
        self.left_arm[0:3] = [round(i,3) for i in left_arm[0:3]]
        self.right_arm[0:3] = [round(i,3) for i in right_arm[0:3]]

        self.left_leg[3:6] = [round(i,1) for i in left_leg[3:6]]
        self.right_leg[3:6] = [round(i,1) for i in right_leg[3:6]]
        self.left_arm[3:6] = [round(i,1) for i in left_arm[3:6]]
        self.right_arm[3:6] = [round(i,1) for i in right_arm[3:6]]

        self.manip_pose_list = [self.left_leg, self.right_leg, self.left_arm, self.right_arm]

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

        return [mean_x, mean_y, mean_z, 0, 0, mean_yaw]

    def manip_in_contact(self, manip):
        if isinstance(manip, int):
            if manip < len(manip_dict):
                return self.manip_pose_list[manip][0] != -99.0
            else:
                rave.raveLogError('Invalid manipulator index: %d'%(manip))
                raw_input()
        elif isinstance(manip, basestring):
            if manip in manip_inv_dict:
                return self.manip_pose_list[manip_inv_dict[manip]][0] != -99.0
            else:
                rave.raveLogError('Invalid manipulator name: %s'%(manip))
                raw_input()
        else:
            rave.raveLogError('Unknown manipulator descriptor type.')
            raw_input()

    def get_virtual_body_pose(self):
        mean_feet_pose = self.get_mean_feet_xyzrpy()
        mean_feet_position = np.array(mean_feet_pose[0:2])
        mean_yaw = mean_feet_pose[5]

        orientation_rotation_matrix = rpy_to_SO3(mean_feet_pose[3:6])
        orientation_unit_vec = np.array([math.cos(mean_yaw*DEG2RAD), math.sin(mean_yaw*DEG2RAD)])

        foot_contact_num = 2
        hand_contact_num = 0
        rotated_x = 0

        if(self.manip_in_contact('l_arm')):
            rotated_x += np.dot(np.array(self.get_manip_pose('l_arm')[0:2])-mean_feet_position, orientation_unit_vec)
            hand_contact_num += 1

        if(self.manip_in_contact('r_arm')):
            rotated_x += np.dot(np.array(self.get_manip_pose('r_arm')[0:2])-mean_feet_position, orientation_unit_vec)
            hand_contact_num += 1

        if hand_contact_num != 0:
            rotated_x = float(rotated_x) / (hand_contact_num + foot_contact_num)
        
        virtual_body_position = np.atleast_2d(mean_feet_position).T + np.dot(orientation_rotation_matrix[0:2,0:2], np.array([[rotated_x],[0]]))

        return [virtual_body_position[0,0], virtual_body_position[1,0], 0, 0, 0, mean_yaw]


import numpy as np
import math
import openravepy as rave
import copy
import random
import IPython

from transformation_conversion import *

def DrawStance(current_node,robot_obj,handles):

    env = robot_obj.env

    foot_corners = np.array([[robot_obj.foot_h/2, robot_obj.foot_w/2, 0.01, 1],
                             [-robot_obj.foot_h/2, robot_obj.foot_w/2, 0.01, 1],
                             [-robot_obj.foot_h/2, -robot_obj.foot_w/2, 0.01, 1],
                             [robot_obj.foot_h/2, -robot_obj.foot_w/2, 0.01, 1]])
    foot_corners = np.transpose(foot_corners)

    hand_corners = np.array([[-0.01, robot_obj.hand_h/2, robot_obj.hand_w/2, 1],
                             [-0.01, -robot_obj.hand_h/2, robot_obj.hand_w/2, 1],
                             [-0.01, -robot_obj.hand_h/2, -robot_obj.hand_w/2, 1],
                             [-0.01, robot_obj.hand_h/2, -robot_obj.hand_w/2, 1]])
    hand_corners = np.transpose(hand_corners)

    c = current_node

    # draw left foot pose
    left_leg_transform = xyzrpy_to_SE3(c.left_leg)
    foot_corners_transformed = np.dot(left_leg_transform,foot_corners)
    foot_corners_transformed = np.delete(foot_corners_transformed, 3, 0)
    foot_corners_transformed = np.transpose(foot_corners_transformed)
    foot_corners_transformed = np.append(foot_corners_transformed, [foot_corners_transformed[0,:]], 0)
    handles.append(env.drawlinestrip(points = foot_corners_transformed,
    linewidth = 5.0,
    colors = np.array(((1,0,0),(1,0,0),(1,0,0),(1,0,0),(1,0,0)))))

    # draw right foot pose
    right_leg_transform = xyzrpy_to_SE3(c.right_leg)
    foot_corners_transformed = np.dot(right_leg_transform,foot_corners)
    foot_corners_transformed = np.delete(foot_corners_transformed, 3, 0)
    foot_corners_transformed = np.transpose(foot_corners_transformed)
    foot_corners_transformed = np.append(foot_corners_transformed, [foot_corners_transformed[0,:]], 0)
    handles.append(env.drawlinestrip(points = foot_corners_transformed,
    linewidth = 5.0,
    colors = np.array(((0,1,0),(0,1,0),(0,1,0),(0,1,0),(0,1,0)))))

    # draw left hand pose
    if(c.left_arm[0] != -99.0):
        left_arm_transform = xyzrpy_to_SE3(c.left_arm)
        hand_corners_transformed = np.dot(left_arm_transform,hand_corners)
        hand_corners_transformed = np.delete(hand_corners_transformed, 3, 0)
        hand_corners_transformed = np.transpose(hand_corners_transformed)
        hand_corners_transformed = np.append(hand_corners_transformed, [hand_corners_transformed[0,:]], 0)
        handles.append(env.drawlinestrip(points = hand_corners_transformed,
        linewidth = 5.0,
        colors = np.array(((0,0,1),(0,0,1),(0,0,1),(0,0,1),(0,0,1)))))

    # draw right hand pose
    if(c.right_arm[0] != -99.0):
        right_arm_transform = xyzrpy_to_SE3(c.right_arm)
        hand_corners_transformed = np.dot(right_arm_transform,hand_corners)
        hand_corners_transformed = np.delete(hand_corners_transformed, 3, 0)
        hand_corners_transformed = np.transpose(hand_corners_transformed)
        hand_corners_transformed = np.append(hand_corners_transformed, [hand_corners_transformed[0,:]], 0)
        handles.append(env.drawlinestrip(points = hand_corners_transformed,
        linewidth = 5.0,
        colors = np.array(((1,1,0),(1,1,0),(1,1,0),(1,1,0),(1,1,0)))))
from robot import HumanoidRobot
from transformation_conversion import *

import numpy as np
import openravepy as rave
import math
import IPython

#initialize link configuration
def hermes_full(env, active_dof_mode='whole_body', urdf_name=None):

    urdf = 'package://hermes_full_description/robot/hermes_full.urdf'
    srdf = 'package://hermes_full_description/robot/hermes_full.srdf'

    if(urdf_name is not None):
        urdf = 'package://hermes_full_description/robot/' + urdf_name + '.urdf'

    hermes_full = HumanoidRobot(env, urdf_path=urdf, srdf_path=srdf)

    # set up the manipulators
    hermes_full.manip.l_arm.SetLocalToolTransform(np.array([
        [ 0, 1,  0, 0.075],
        [ 1, 0,  0, 0],
        [ 0, 0, -1, 0],
        [ 0, 0,  0, 1]])
    )
    hermes_full.manip.l_arm.SetLocalToolDirection(np.array([1, 0, 0]))

    hermes_full.manip.r_arm.SetLocalToolTransform(np.array([
        [ 0, -1, 0, 0.075],
        [ 1,  0, 0, 0],
        [ 0,  0, 1, 0],
        [ 0,  0, 0, 1]])
    )
    hermes_full.manip.r_arm.SetLocalToolDirection(np.array([1, 0, 0]))

    hermes_full.manip.l_leg.SetLocalToolTransform(np.array([
        [  0,  0, 1, -0.0512],
        [  0,  1, 0, 0],
        [ -1,  0, 0, 0],
        [  0,  0, 0, 1]])
    )
    hermes_full.manip.l_leg.SetLocalToolDirection(np.array([0, 0, -1]))

    hermes_full.manip.r_leg.SetLocalToolTransform(np.array([
        [ 0, 0, 1, -0.0512],
        [ 0, -1, 0, 0],
        [ 1, 0, 0, 0],
        [ 0, 0, 0, 1]])
    )
    hermes_full.manip.r_leg.SetLocalToolDirection(np.array([0, 0, -1]))

    hermes_full.foot_h = 0.16
    hermes_full.foot_w = 0.06
    hermes_full.hand_h = 0.06
    hermes_full.hand_w = 0.06
    hermes_full.foot_radius = math.sqrt((hermes_full.foot_h/2.0)**2 + (hermes_full.foot_w/2.0)**2)
    hermes_full.hand_radius = math.sqrt((hermes_full.hand_h/2.0)**2 + (hermes_full.hand_w/2.0)**2)

    hermes_full.robot_z = 0.7
    hermes_full.top_z = 1.4
    hermes_full.shoulder_z = 1.2
    hermes_full.shoulder_w = 0.392

    hermes_full.max_arm_length = 0.4 # 0.5 arm length
    hermes_full.min_arm_length = 0.2

    hermes_full.max_stride = 0.4 # no test

    # specify the active DOFs
    hermes_full.additional_active_DOFs = ['x_prismatic_joint','y_prismatic_joint','z_prismatic_joint','roll_revolute_joint','pitch_revolute_joint','yaw_revolute_joint','B_TR','B_TAA']

    l_arm_indices = hermes_full.robot.GetManipulator('l_arm').GetArmIndices()
    r_arm_indices = hermes_full.robot.GetManipulator('r_arm').GetArmIndices()
    l_leg_indices = hermes_full.robot.GetManipulator('l_leg').GetArmIndices()
    r_leg_indices = hermes_full.robot.GetManipulator('r_leg').GetArmIndices()

    additional_active_DOF_indices = [hermes_full.robot.GetJoint(joint_index).GetDOFIndex() for joint_index in hermes_full.additional_active_DOFs]
    whole_body_indices = np.concatenate((l_arm_indices, r_arm_indices, l_leg_indices, r_leg_indices, additional_active_DOF_indices))
    legs_indices = np.concatenate((l_leg_indices, r_leg_indices, additional_active_DOF_indices))

    if active_dof_mode == 'whole_body':
        hermes_full.robot.SetActiveDOFs(whole_body_indices)
    elif active_dof_mode == 'legs_only':
        hermes_full.robot.SetActiveDOFs(legs_indices)


    # initialize robot config data (DOFNameDict, joint limits)
    hermes_full.initialize_config_data()

    # initialize robot collision box
    hermes_full.initialize_end_effector_collision_box()

    out_of_env_transform = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-99.0],[0,0,0,1]])
    hermes_full.body_collision_box = rave.RaveCreateKinBody(hermes_full.env,'')
    hermes_full.body_collision_box.SetName('body_collision_box')
    hermes_full.body_collision_box.InitFromBoxes(np.array([[0, 0, (0.5+hermes_full.top_z)/2.0, 0.3, hermes_full.shoulder_w/2.0, (hermes_full.top_z-0.5)/2.0]]), False)
    # hermes_full.body_collision_box.InitFromBoxes(np.array([[0, 0, 0, 0.18, 0.29, 0.56]]), True)
    # hermes_full.body_collision_box.GetLinks()[0].GetGeometries()[0].SetTransparency(0.2)
    hermes_full.env.AddKinBody(hermes_full.body_collision_box)
    hermes_full.body_collision_box.SetTransform(out_of_env_transform)

    # Construct the hermes_full robot transform
    # hermes_full.robot.SetTransform(np.array([[0,1,0,0],
    #                                     [-1,0,0,0],
    #                                     [0,0,1,1],
    #                                     [0,0,0,1]], dtype=float))

    hermes_full.body_collision_box_offset = np.array([[1,0,0,0],
                                                      [0,1,0,-0.05],
                                                      [0,0,1,0.33],
                                                      [0,0,0,1]], dtype=float)
    hermes_full.inverse_body_collision_box_offset = inverse_SE3(hermes_full.body_collision_box_offset)

    hermes_full.origin_body_transform = np.array([[0,1,0,0],
                                                  [-1,0,0,0],
                                                  [0,0,1,0],
                                                  [0,0,0,1]], dtype=float)

    # Construct the GazeboOriginalDOFValues
    hermes_full.GazeboOriginalDOFValues = np.zeros(hermes_full.robot.GetDOF())
    hermes_full.GazeboOriginalDOFValues[hermes_full.DOFNameIndexDict['z_prismatic_joint']] = 0.8
    hermes_full.GazeboOriginalDOFValues[hermes_full.DOFNameIndexDict['yaw_revolute_joint']] = -math.pi/2

    # Construct the OriginalDOFValues
    hermes_full.robot.SetDOFValues(hermes_full.GazeboOriginalDOFValues)

    DOFValues = hermes_full.robot.GetDOFValues()

    hermes_full.robot.SetDOFValues(DOFValues)

    ActiveDOFValues = hermes_full.robot.GetActiveDOFValues()
    for d in range(len(ActiveDOFValues)):
        ActiveDOFValues[d] = (hermes_full.lower_limits[d] + hermes_full.higher_limits[d])/2.0
    hermes_full.robot.SetActiveDOFValues(ActiveDOFValues)

    hermes_full.OriginalDOFValues = hermes_full.robot.GetDOFValues()
    hermes_full.OriginalDOFValues[hermes_full.DOFNameIndexDict['z_prismatic_joint']] += 0.8
    hermes_full.OriginalDOFValues[hermes_full.DOFNameIndexDict['yaw_revolute_joint']] += -math.pi/2

    hermes_full.GazeboOriginalDOFValues = np.copy(hermes_full.OriginalDOFValues)

    hermes_full.mass = 63.47

    # IPython.embed()

    return hermes_full

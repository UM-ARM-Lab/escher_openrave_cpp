from robot import HumanoidRobot
from transformation_conversion import *

import numpy as np
import openravepy as rave
import math
import IPython

#initialize link configuration
def athena(env, active_dof_mode='whole_body', urdf_name=None):

    # urdf = 'package://athena_description/robot/athena_step_interpolation.urdf'
    # urdf = 'package://athena_description/robot/athena_ik_fast_test.urdf'
    urdf = 'package://athena_description/robot/athena.urdf'
    srdf = 'package://athena_description/robot/athena.srdf'

    if(urdf_name is not None):
        urdf = 'package://athena_description/robot/' + urdf_name + '.urdf'

    athena = HumanoidRobot(env, urdf_path=urdf, srdf_path=srdf)

    # set up the manipulators
    athena.manip.l_arm.SetLocalToolDirection(np.array([1, 0, 0]))
    athena.manip.l_arm.SetLocalToolTransform(np.array([
        [ 0,  1, 0, 0.108],
        [ 0,  0, 1, 0.0],
        [ 1,  0, 0, 0.04],
        [ 0,  0, 0,   1]])
    )

    athena.manip.r_arm.SetLocalToolDirection(np.array([1, 0, 0]))
    athena.manip.r_arm.SetLocalToolTransform(np.array([
        [ 0, -1, 0, 0.108],
        [ 0,  0, 1, 0.0],
        [-1,  0, 0, -0.04],
        [ 0,  0, 0, 1]])
    )

    athena.manip.l_leg.SetLocalToolTransform(np.array([
        [ 0, 0,-1, 0.1],
        [ 0,-1, 0, 0.0],
        [-1, 0, 0,-0.045],
        [ 0, 0, 0, 1]])
    )
    athena.manip.l_leg.SetLocalToolDirection(np.array([0, 0, -1]))

    athena.manip.r_leg.SetLocalToolTransform(np.array([
        [ 0, 0,-1, 0.1],
        [ 0, 1, 0, 0.0],
        [ 1, 0, 0,0.045],
        [ 0, 0, 0, 1]])
    )
    athena.manip.r_leg.SetLocalToolDirection(np.array([0, 0, -1]))

    athena.foot_h = 0.22
    athena.foot_w = 0.05
    athena.hand_h = 0.09
    athena.hand_w = 0.09
    athena.foot_radius = math.sqrt((athena.foot_h/2.0)**2 + (athena.foot_w/2.0)**2)
    athena.hand_radius = math.sqrt((athena.hand_h/2.0)**2 + (athena.hand_w/2.0)**2)

    athena.robot_z = 1.0
    athena.top_z = 1.85
    athena.shoulder_z = 1.4
    athena.shoulder_w = 0.592

    # athena.max_arm_length = 0.75 # 0.688 arm length
    athena.max_arm_length = 0.5 # 0.688 arm length
    athena.min_arm_length = 0.2

    athena.max_stride = 0.4 # wrong

    # specify the active DOFs
    athena.additional_active_DOFs = ['x_prismatic_joint','y_prismatic_joint','z_prismatic_joint','roll_revolute_joint','pitch_revolute_joint','yaw_revolute_joint','B_TR','B_TAA']

    l_arm_indices = athena.robot.GetManipulator('l_arm').GetArmIndices()
    r_arm_indices = athena.robot.GetManipulator('r_arm').GetArmIndices()
    l_leg_indices = athena.robot.GetManipulator('l_leg').GetArmIndices()
    r_leg_indices = athena.robot.GetManipulator('r_leg').GetArmIndices()

    additional_active_DOF_indices = [athena.robot.GetJoint(joint_index).GetDOFIndex() for joint_index in athena.additional_active_DOFs]
    whole_body_indices = np.concatenate((l_arm_indices, r_arm_indices, l_leg_indices, r_leg_indices, additional_active_DOF_indices))
    legs_indices = np.concatenate((l_leg_indices, r_leg_indices, additional_active_DOF_indices))

    if active_dof_mode == 'whole_body':
        athena.robot.SetActiveDOFs(whole_body_indices)
    elif active_dof_mode == 'legs_only':
        athena.robot.SetActiveDOFs(legs_indices)


    # initialize robot config data (DOFNameDict, joint limits)
    athena.initialize_config_data()

    # initialize robot collision box
    athena.initialize_end_effector_collision_box()

    out_of_env_transform = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-99.0],[0,0,0,1]])
    athena.body_collision_box = rave.RaveCreateKinBody(athena.env,'')
    athena.body_collision_box.SetName('body_collision_box')
    athena.body_collision_box.InitFromBoxes(np.array([[0, 0, (0.5+athena.top_z)/2.0, 0.3, athena.shoulder_w/2.0, (athena.top_z-0.5)/2.0]]), False)
    # athena.body_collision_box.InitFromBoxes(np.array([[0, 0, 0, 0.18, 0.29, 0.56]]), True)
    # athena.body_collision_box.GetLinks()[0].GetGeometries()[0].SetTransparency(0.2)
    athena.env.AddKinBody(athena.body_collision_box)
    athena.body_collision_box.SetTransform(out_of_env_transform)

    # Construct the athena robot transform
    # athena.robot.SetTransform(np.array([[0,1,0,0],
    #                                     [-1,0,0,0],
    #                                     [0,0,1,1],
    #                                     [0,0,0,1]], dtype=float))

    athena.body_collision_box_offset = np.array([[1,0,0,0],
                                                 [0,1,0,-0.05],
                                                 [0,0,1,0.33],
                                                 [0,0,0,1]], dtype=float)
    athena.inverse_body_collision_box_offset = inverse_SE3(athena.body_collision_box_offset)

    athena.origin_body_transform = np.array([[0,1,0,0],
                                             [-1,0,0,0],
                                             [0,0,1,0],
                                             [0,0,0,1]], dtype=float)

    # Construct the GazeboOriginalDOFValues
    athena.GazeboOriginalDOFValues = np.zeros(athena.robot.GetDOF())
    athena.GazeboOriginalDOFValues[athena.DOFNameIndexDict['z_prismatic_joint']] = 1.0
    athena.GazeboOriginalDOFValues[athena.DOFNameIndexDict['yaw_revolute_joint']] = -math.pi/2

    # Construct the OriginalDOFValues
    athena.robot.SetDOFValues(athena.GazeboOriginalDOFValues)

    DOFValues = athena.robot.GetDOFValues()

    athena.robot.SetDOFValues(DOFValues)

    ActiveDOFValues = athena.robot.GetActiveDOFValues()
    for d in range(len(ActiveDOFValues)):
        ActiveDOFValues[d] = (athena.lower_limits[d] + athena.higher_limits[d])/2.0
    athena.robot.SetActiveDOFValues(ActiveDOFValues)

    athena.OriginalDOFValues = athena.robot.GetDOFValues()
    athena.OriginalDOFValues[athena.DOFNameIndexDict['z_prismatic_joint']] += 1.0
    athena.OriginalDOFValues[athena.DOFNameIndexDict['yaw_revolute_joint']] += -math.pi/2

    athena.GazeboOriginalDOFValues = np.copy(athena.OriginalDOFValues)

    athena.mass = 63.47

    # IPython.embed()

    return athena

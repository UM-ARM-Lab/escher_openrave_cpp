# Python interface for the C++ Escher motion planning module
import openravepy as rave

import time
import sys
import numpy as np
import scipy

class escher_openrave_cpp_wrapper(object):
    """
    Wrapper class for the OpenRave Escher plugin.

    :type env rave.Environment
    :type problem rave.Problem
    :type manip_indices dict[rave.Manipulator|str, int]
    """

    def __init__(self, env):
        self.module = rave.RaveCreateModule(env,'EscherMotionPlanning')
        env.AddModule(self.module,'')
        self.env = env

    def AppendStructuresCommand(self,cmd,structures):
        cmd.append('structures')
        cmd.append(len(structures))

        for struct in structures:
            cmd.append(struct.geometry)
            cmd.append(struct.kinbody.GetName())
            cmd.append(struct.id)

            if(struct.geometry == 'trimesh'):
                # plane parameters
                cmd.extend((struct.nx,struct.ny,struct.nz,struct.c))

                # vertices
                cmd.append(len(struct.vertices))

                for vertex in struct.vertices:
                    cmd.extend(vertex)

                # boundaries
                cmd.append(len(struct.boundaries))

                for edge in struct.boundaries:
                    cmd.extend(edge)

                # trimesh types
                cmd.append(struct.type)

            elif(struct.geometry == 'box'):
                print('Warning: Sent box structures, but is ignored.')
                continue

    def AppendMapGridCommand(self,cmd,dh_grid):
        cmd.append('map_grid')
        cmd.append(dh_grid.min_x)
        cmd.append(dh_grid.max_x)
        cmd.append(dh_grid.min_y)
        cmd.append(dh_grid.max_y)
        cmd.append(dh_grid.resolution)

        for i in range(dh_grid.dim_x):
            for j in range(dh_grid.dim_y):

                cmd.append(dh_grid.cell_2D_list[i][j].height)

                if(dh_grid.cell_2D_list[i][j].foot_ground_projection[0]):
                    cmd.append(1)
                else:
                    cmd.append(0)

                if(dh_grid.cell_2D_list[i][j].foot_ground_projection[1] is None):
                    cmd.append(-99)
                else:
                    cmd.append(dh_grid.cell_2D_list[i][j].foot_ground_projection[1])

                cmd.append(len(dh_grid.cell_2D_list[i][j].all_ground_structures))
                cmd.extend(dh_grid.cell_2D_list[i][j].all_ground_structures)

                for k in range(dh_grid.dim_theta):
                    if(dh_grid.cell_3D_list[i][j][k].parent is not None):
                        cmd.extend(dh_grid.cell_3D_list[i][j][k].parent.get_indices())
                    else:
                        cmd.extend((-99,-99,-99))
                    cmd.append(dh_grid.cell_3D_list[i][j][k].g)
                    cmd.append(dh_grid.cell_3D_list[i][j][k].h)

                    cmd.append(len(dh_grid.cell_3D_list[i][j][k].left_hand_checking_surface_index))
                    cmd.extend(dh_grid.cell_3D_list[i][j][k].left_hand_checking_surface_index)
                    cmd.append(len(dh_grid.cell_3D_list[i][j][k].right_hand_checking_surface_index))
                    cmd.extend(dh_grid.cell_3D_list[i][j][k].right_hand_checking_surface_index)

        # import IPython; IPython.embed()
        for key,window in dh_grid.left_foot_neighbor_window.iteritems():
            cmd.append(len(window))
            for cell in window:
                cmd.extend(cell)

        for key,window in dh_grid.right_foot_neighbor_window.iteritems():
            cmd.append(len(window))
            for cell in window:
                cmd.extend(cell)

        for key,window in dh_grid.chest_neighbor_window.iteritems():
            cmd.append(len(window))
            for cell in window:
                cmd.extend(cell)

    def AppendHandTransitionModelCommand(self,cmd,hand_transition_model):
        cmd.append('hand_transition_model')
        cmd.append(len(hand_transition_model))

        for hand_transition in hand_transition_model:
            cmd.extend(hand_transition)

    def AppendFootTransitionModelCommand(self,cmd,foot_transition_model):
        cmd.append('foot_transition_model')
        cmd.append(len(foot_transition_model))

        for foot_transition in foot_transition_model:
            cmd.extend(foot_transition)

    def AppendRobotPropertiesCommand(self,cmd,escher):
        cmd.append('robot_properties')
        for v in escher.OriginalDOFValues:
            cmd.append(v)
        for v in escher.GazeboOriginalDOFValues:
            cmd.append(v)

        cmd.append(escher.foot_h)
        cmd.append(escher.foot_w)
        cmd.append(escher.hand_h)
        cmd.append(escher.hand_w)
        cmd.append(escher.robot_z)
        cmd.append(escher.top_z)
        cmd.append(escher.shoulder_z)
        cmd.append(escher.shoulder_w)
        cmd.append(escher.max_arm_length)
        cmd.append(escher.min_arm_length)
        cmd.append(escher.max_stride)

    def SendStartCalculatingTraversabilityCommand(self,structures=None,footstep_windows_legs_only=None,footstep_windows=None,torso_transitions=None,footstep_window_grid_resolution=None,
                                                  dh_grid=None,hand_transition_model=None,parallelization=None,printing=False):
        start = time.time()

        cmd = ['StartCalculatingTraversability']

        if(printing):
            cmd.append('printing')

        if(structures is not None):
            self.AppendStructuresCommand(cmd, structures)

        if(footstep_windows_legs_only is not None):
            cmd.append('transition_footstep_window_cells_legs_only')
            cmd.append(len(footstep_windows_legs_only))

            for key,footstep_window in footstep_windows_legs_only.iteritems():
                cmd.append(key[0])
                cmd.append(key[1])
                cmd.append(int(key[2]))
                cmd.append(len(footstep_window))

                for cell_tuple in footstep_window:
                    cmd.extend(cell_tuple[0])
                    cmd.extend(cell_tuple[1])
                    cmd.extend(cell_tuple[2])

        if(footstep_windows is not None):
            cmd.append('transition_footstep_window_cells')
            cmd.append(len(footstep_windows))

            for key,footstep_window in footstep_windows.iteritems():
                cmd.append(key[0])
                cmd.append(key[1])
                cmd.append(int(key[2]))
                cmd.append(len(footstep_window))

                for cell_tuple in footstep_window:
                    cmd.extend(cell_tuple[0])
                    cmd.extend(cell_tuple[1])
                    cmd.extend(cell_tuple[2])

        if(torso_transitions is not None):
            cmd.append('torso_transitions')
            cmd.append(len(torso_transitions))

            for transition in torso_transitions:
                cmd.extend(transition)

        if(footstep_window_grid_resolution is not None):
            cmd.append('footstep_window_grid_resolution')
            cmd.append(footstep_window_grid_resolution)

        if(dh_grid is not None):
            self.AppendMapGridCommand(cmd, dh_grid)

        if(hand_transition_model is not None):
            self.AppendHandTransitionModelCommand(cmd, hand_transition_model)

        if(parallelization is not None):
            cmd.append('parallelization')

            if(parallelization):
                cmd.append(1)
            else:
                cmd.append(0)

        cmd_str = " ".join(str(item) for item in cmd)

        after_constructing_command = time.time()

        result_str = self.module.SendCommand(cmd_str)

        after_calculation = time.time()

        result = [float(x) for x in result_str.split()]

        footstep_transition_traversability_legs_only = {}
        footstep_transition_traversability = {}
        hand_transition_traversability = {}

        counter = 0

        footstep_transition_num = result[counter]
        counter += 1
        for i in range(int(footstep_transition_num)):
            footstep_transition_traversability_legs_only[tuple(int(x) for x in result[counter:counter+5])] = result[counter+5:counter+8]
            counter += 8

        footstep_transition_num = result[counter]
        counter += 1
        for i in range(int(footstep_transition_num)):
            footstep_transition_traversability[tuple(int(x) for x in result[counter:counter+5])] = result[counter+5:counter+8]
            counter += 8

        hand_transition_num = result[counter]
        counter += 1
        for i in range(int(hand_transition_num)):
            hand_transition_traversability[tuple(int(x) for x in result[counter:counter+3])] = result[counter+3:counter+15]
            counter += 15

        # print("Output message received in Python:")
        # print(result_str)

        after_parsing_output = time.time()

        print('Constructing Command Time: %d miliseconds.'%((after_constructing_command-start)*1000))
        print('Calculation Time: %d miliseconds.'%((after_calculation-after_constructing_command)*1000))
        print('Parsing Output Time: %d miliseconds.'%((after_parsing_output-after_calculation)*1000))

        return (footstep_transition_traversability_legs_only, footstep_transition_traversability, hand_transition_traversability)


    def SendStartConstructingContactRegions(self,structures=None,printing=False,structures_id=None):

        start = time.time()

        cmd = ['StartConstructingContactRegions']

        if(printing):
            cmd.append('printing')

        if(structures is not None):
            cmd.append('structures')
            cmd.append(len(structures))

            for struct in structures:
                cmd.append(struct.geometry)
                cmd.append(struct.kinbody.GetName())
                cmd.append(struct.id)

                if(struct.geometry == 'trimesh'):
                    # plane parameters
                    cmd.extend((struct.nx,struct.ny,struct.nz,struct.c))

                    # vertices
                    cmd.append(len(struct.vertices))

                    for vertex in struct.vertices:
                        cmd.extend(vertex)

                    # boundaries
                    cmd.append(len(struct.boundaries))

                    for edge in struct.boundaries:
                        cmd.extend(edge)

                    # trimesh types
                    cmd.append(struct.type)

                elif(struct.geometry == 'box'):
                    print('Warning: Sent box structures, but is ignored.')
                    continue

        if structures_id is not None: # specify the structures id of interest in extracting contact regions
            cmd.append('structures_id')
            cmd.append(len(structures_id))
            for s_id in structures_id:
                cmd.append(s_id)


        cmd_str = " ".join(str(item) for item in cmd)

        after_constructing_command = time.time()

        result_str = self.module.SendCommand(cmd_str)

        after_calculation = time.time()

        result = [float(x) for x in result_str.split()]

        counter = 0

        contact_points_num = int(result[counter])
        contact_points_values = [[0]*6 for i in range(contact_points_num)]
        counter += 1
        for i in range(contact_points_num):
            contact_points_values[i] = result[counter:counter+6]
            counter += 6

        contact_regions_num = int(result[counter])
        contact_regions_values = [[0]*7 for i in range(contact_regions_num)]
        counter += 1
        for i in range(contact_regions_num):
            contact_regions_values[i] = result[counter:counter+7]
            counter += 7


        # print("Output message received in Python:")
        # print(result_str)

        after_parsing_output = time.time()

        print('Constructing Command Time: %d miliseconds.'%((after_constructing_command-start)*1000))
        print('Contact Region and Point Calculation Time: %d miliseconds.'%((after_calculation-after_constructing_command)*1000))
        print('Parsing Output Time: %d miliseconds.'%((after_parsing_output-after_calculation)*1000))

        return (contact_points_values,contact_regions_values)


    def SendStartPlanningFromScratch(self,robot_name=None,escher=None,initial_state=None,goal=None,foot_transition_model=None,hand_transition_model=None,
                                     structures=None,goal_radius=None,time_limit=None,planning_heuristics='euclidean',output_first_solution=False,
                                     goal_as_exact_poses=False,thread_num=None,branching_method=None,printing=None):

        start = time.time()

        cmd = ['StartPlanningFromScratch']

        if printing:
            cmd.append('printing')

        if robot_name is not None:
            cmd.append('robot_name')
            cmd.append(robot_name)
        else:
            print('robot name is required for planning. Abort.')
            return

        if (initial_state is not None) and (escher is not None):
            cmd.append('initial_state')

            for i in range(4):
                pose = initial_state.get_manip_pose(i)
                for v in pose:
                    cmd.append(v)

            for i in range(4):
                pose = initial_state.get_manip_pose(i)

                if pose[0] == -99.0:
                    cmd.append(0)
                else:
                    cmd.append(1)

            mean_feet_xyzrpy = initial_state.get_mean_feet_xyzrpy()

            cmd.append(mean_feet_xyzrpy[0])
            cmd.append(mean_feet_xyzrpy[1])
            cmd.append(mean_feet_xyzrpy[2] + escher.robot_z)
            cmd.extend([0, 0, 0])

        else:
            print('initial state and robot properties(escher) is required for planning. Abort.')
            return

        if goal is not None:
            cmd.append('goal')
            for g in goal:
                cmd.append(g)
        else:
            print('goal is required for planning. Abort.')
            return

        if escher is not None:
            self.AppendRobotPropertiesCommand(cmd, escher)
        else:
            print('robot properties(escher) is required for planning. Abort.')
            return

        if structures is not None:
            self.AppendStructuresCommand(cmd, structures)
        else:
            print('structures are required for planning. Abort.')
            return

        if foot_transition_model is not None:
            self.AppendFootTransitionModelCommand(cmd, foot_transition_model)
        else:
            print('foot transition model is required for planning. Abort.')
            return

        if hand_transition_model is not None:
            self.AppendHandTransitionModelCommand(cmd, hand_transition_model)

        if (goal_radius is not None) and (time_limit is not None):
            cmd.append('planning_parameters')
            cmd.append(goal_radius)
            cmd.append(time_limit)

            if planning_heuristics == 'euclidean' or planning_heuristics == 'dijkstra':
                cmd.append(planning_heuristics)
            else:
                print('Unknown planning heuristics type %s. Abort.'%(planning_heuristics))
                return

            if output_first_solution:
                cmd.append(1)
            else:
                cmd.append(0)

            if goal_as_exact_poses:
                cmd.append(1)
            else:
                cmd.append(0)
        else:
            print('goal radius and time limit are required for planning. Abort.')
            return

        if thread_num is not None:
            cmd.append('thread_num')
            cmd.append(thread_num)

        if branching_method is not None:
            cmd.append('branching_method')
            cmd.append(branching_method)

        cmd_str = " ".join(str(item) for item in cmd)

        after_constructing_command = time.time()

        result_str = self.module.SendCommand(cmd_str)

        after_calculation = time.time()

        # result = [float(x) for x in result_str.split()]

        # parsing the outputs

        after_parsing_output = time.time()

        print('Constructing Command Time: %d miliseconds.'%((after_constructing_command-start)*1000))
        print('Contact Region and Point Calculation Time: %d miliseconds.'%((after_calculation-after_constructing_command)*1000))
        print('Parsing Output Time: %d miliseconds.'%((after_parsing_output-after_calculation)*1000))
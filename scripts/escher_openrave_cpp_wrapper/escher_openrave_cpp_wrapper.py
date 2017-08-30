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

        # rave.RaveInitialize()
        # rave.RaveLoadPlugin('')
        # EscherMotionPlanning = rave.RaveCreateModule(env,'EscherMotionPlanning')

        # # Build a mapping for manipulators to manipulator indices
        # self.manip_indices = {}
        # for index, manip in enumerate(env.GetRobot(robotname).GetManipulators()):
        #     self.manip_indices[manip] = index
        #     self.manip_indices[manip.GetName()] = index

    # def load_robot(env, urdf_path=None, srdf_path=None):
    #     if(not urdf_path):
    #         urdf_path = urdf

    #     if(not srdf_path):
    #         srdf_path = srdf


    #     rave.RaveLoadPlugin('../or_urdf/build/devel/lib/openrave-0.9/or_urdf_plugin')
    #     module = rave.RaveCreateModule(env, 'urdf')
    #     robot_name = module.SendCommand('load {} {}'.format(urdf_path, srdf_path))
    #     robot = env.GetRobot(robot_name)

    #     robot.GetManipulator('l_arm').SetLocalToolDirection(np.array([1, 0, 0]))
    #     robot.GetManipulator('l_arm').SetLocalToolTransform(np.array([
    #         [0,  1, 0, 0.18],
    #         [ -1, 0, 0, -0.025],
    #         [ 0,  0, 1, 0],
    #         [ 0,  0, 0, 1]])
    #     )

    #     robot.GetManipulator('r_arm').SetLocalToolDirection(np.array([1, 0, 0]))
    #     robot.GetManipulator('r_arm').SetLocalToolTransform(np.array([
    #         [ 0,  -1, 0, 0.18],
    #         [ 1,  0, 0, 0.025],
    #         [ 0,  0, 1, 0],
    #         [ 0,  0, 0, 1]])
    #     )

    #     robot.GetManipulator('l_leg').SetLocalToolDirection(np.array([0, 0, -1]))
    #     robot.GetManipulator('r_leg').SetLocalToolDirection(np.array([0, 0, -1]))

    #     return robot

    # OpenRave C++ plugin is called by sending string command. We can add parameters in this function to construct the command, and decode in C++ side.
    # For example, I can add an option whether to turn on the parallelization or not
    def SendStartPlanningCommand(self,robotname=None,goal=None,parallelization=None):
        cmd = ['StartPlanning']

        cmd.append('robotname')
        cmd.append(robotname)

        cmd.append('goal')

        for g in goal:
            cmd.append(g)

        if(parallelization is not None):
            cmd.append('parallelization')

            if(parallelization):
                cmd.append(1)
            else:
                cmd.append(0)

        cmd_str = " ".join(str(item) for item in cmd)

        result_str = self.module.SendCommand(cmd_str)

        print("Output message received in Python:")
        print(result_str)

        return

    def SendStartCalculatingTraversabilityCommand(self, structures=None, footstep_windows=None, torso_transitions=None, footstep_window_grid_dimension=None, 
                                                dh_grid=None, hand_transition_model=None, parallelization=None, printing=False):
        start = time.time()
        
        cmd = ['StartCalculatingTraversability']

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

        if(footstep_window_grid_dimension is not None):
            cmd.append('footstep_window_grid_dimension')
            cmd.extend(footstep_window_grid_dimension)

        if(dh_grid is not None):
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

        if(hand_transition_model is not None):
            cmd.append('hand_transition_model')
            cmd.append(len(hand_transition_model))

            for hand_transition in hand_transition_model:
                cmd.extend(hand_transition)

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

        footstep_transition_traversability = {}
        hand_transition_traversability = {}

        counter = 0

        footstep_transition_num = result[counter]
        counter += 1
        for i in range(int(footstep_transition_num)):
            footstep_transition_traversability[tuple(int(x) for x in result[counter:counter+5])] = result[counter+5]
            counter += 6

        hand_transition_num = result[counter]
        counter += 1
        for i in range(int(hand_transition_num)):
            hand_transition_traversability[tuple(int(x) for x in result[counter:counter+3])] = result[counter+3:counter+7]
            counter += 7        

        # print("Output message received in Python:")
        # print(result_str)

        after_parsing_output = time.time()

        print('Constructing Command Time: %d miliseconds.'%((after_constructing_command-start)*1000))
        print('Calculation Time: %d miliseconds.'%((after_calculation-after_constructing_command)*1000))
        print('Parsing Output Time: %d miliseconds.'%((after_parsing_output-after_calculation)*1000))

        return (footstep_transition_traversability,hand_transition_traversability)

# def main():
#     env = rave.Environment()
#     env.SetViewer('qtcoin')
#     env.Reset()

#     ## load the Escher robot
#     urdf = 'file://escher_model/escher_cpp.urdf'
#     srdf = 'file://escher_model/escher_cpp.srdf'

#     robot = load_robot(env, urdf_path=urdf, srdf_path=srdf)

#     ### INITIALIZE PLUGIN ###
#     rave.RaveInitialize()
#     rave.RaveLoadPlugin('build/escher_motion_planning')
#     EscherMotionPlanning = rave.RaveCreateModule(env,'EscherMotionPlanning')
#     ### END INITIALIZING PLUGIN ###

#     # print("python env pointer: " + RaveGetEnvironment())

#     SendStartPlanningCommand(EscherMotionPlanning,robotname=robot.GetName(),goal=[2.5,0.5,0.0],parallelization=True)

#     raw_input("Press enter to exit...")
#     # import IPython; IPython.embe        

        # print("Output message received in Python:")
        # print(result_str)

        after_parsing_output = time.time()

        print('Constructing Command Time: %d miliseconds.'%((after_constructing_command-start)*1000))
        print('Calculation Time: %d miliseconds.'%((after_calculation-after_constructing_command)*1000))
        print('Parsing Output Time: %d miliseconds.'%((after_parsing_output-after_calculation)*1000))

        return (footstep_transition_traversability,hand_transition_traversability)

# def main():
#     env = rave.Environment()d();

#     return



# if __name__ == "__main__":
#     main()
    

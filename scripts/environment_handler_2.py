import numpy as np
import openravepy as rave
import random
import math
import IPython
import os
import shutil
from stl import mesh

from structures_2 import *
from color_library import *

# random.seed(20190520)
# random.seed(20190731)
random.seed(20190805)
# random.seed(20190825)

class environment_handler:
    def __init__(self,env=None,structures=None):
        if(env is None):
            rave.misc.InitOpenRAVELogging()
            self.env = rave.Environment()  # create openrave environment
            self.env.SetViewer('qtcoin')  # attach viewer (optional)
            # self.env.GetViewer().SetCamera(np.array([[0,-1,0,1],
            #                                          [-1,0,0,0],
            #                                          [0,0,-1,7],
            #                                          [0,0,0,1]], dtype=float))

            fcl = rave.RaveCreateCollisionChecker(self.env, "fcl_")
            if fcl is not None:
                self.env.SetCollisionChecker(fcl)
            else:
                print("FCL Not installed, falling back to ode")
                self.env.SetCollisionChecker(rave.RaveCreateCollisionChecker(self.env, "ode"))

        else:
            self.env = env

        if structures is None:
            self.structures = []
        else:
            self.structures = structures

        self.draw_handles = []

        self.goal_x = 0.0
        self.goal_y = 0.0

        self.init_z = -9999.0
        self.goal_z = -9999.0

    def DrawSurface(self,surface,transparency=1.0,style='greyscale'):
        if style == 'random_color':
            r = random.random(); g = random.random(); b = random.random()

            total_rgb = math.sqrt(r**2+g**2+b**2)
            r = r/total_rgb; g = g/total_rgb; b = b/total_rgb

        elif style == 'greyscale':
            greyscale = random.uniform(0.2,0.8)

            r = greyscale; g = greyscale; b = greyscale

        elif style == 'random_green_yellow_color':
            color = random.choice(color_library)
            r = color[0] / 255.0; g = color[1] / 255.0; b = color[2] / 255.0

        for boundary in surface.boundaries:
            boundaries_point = np.zeros((2,3),dtype=float)
            boundaries_point[0,:] = np.array(surface.vertices[boundary[0]])
            boundaries_point[1,:] = np.array(surface.vertices[boundary[1]])

            self.draw_handles.append(self.env.drawlinestrip(points=boundaries_point, linewidth=5.0, colors=np.array((r,g,b))))

        self.draw_handles.append(self.env.drawtrimesh(surface.kinbody.GetLinks()[0].GetCollisionData().vertices, surface.kinbody.GetLinks()[0].GetCollisionData().indices, colors=np.array([r,g,b,transparency])))

    def DrawOrientation(self, transform, size=0.2):

        if (np.shape(transform) == (4,4) or np.shape(transform) == (3,3)):
            from_vec = []
            # if (location == None):
            from_vec = transform[0:3,3]
            # elif (type(location) == geometry_msgs.msg._Point.Point):
            #     from_vec = self.utils.PointToArray(location.position)

            to_vec_1 = from_vec + size*(transform[0:3,0])
            to_vec_2 = from_vec + size*(transform[0:3,1])
            to_vec_3 = from_vec + size*(transform[0:3,2])

            self.draw_handles.append(self.env.drawarrow(from_vec, to_vec_1, 0.005, [1, 0, 0]))
            self.draw_handles.append(self.env.drawarrow(from_vec, to_vec_2, 0.005, [0, 1, 0]))
            self.draw_handles.append(self.env.drawarrow(from_vec, to_vec_3, 0.005, [0, 0, 1]))

    def add_quadrilateral_surface(self,structures,projected_vertices,global_transform,surface_type='ground',surface_transparancy=1.0):
        # the projected surface must be in counter-clockwise order

        surface_vertices = [None] * 4
        surface_boundaries = [(0,1),(1,2),(2,3),(3,0)]
        surface_trimesh_indices = np.array([[0,1,2],[0,2,3]])

        if type(global_transform) == list:
            global_transform = xyzrpy_to_SE3(global_transform)

        for i in range(4):
            vertex = projected_vertices[i]

            global_vertex = np.dot(global_transform, np.array([[vertex[0]],[vertex[1]],[0],[1]], dtype=float))

            surface_vertices[i] = (global_vertex[0,0], global_vertex[1,0], global_vertex[2,0])

        surface_trimesh_vertices = np.zeros((4,3),dtype=float)
        for i in range(len(surface_trimesh_vertices)):
            for j in range(3):
                surface_trimesh_vertices[i,j] = surface_vertices[i][j]

        # surface_trimesh = rave.RaveCreateKinBody(self.env,'')
        # surface_trimesh.SetName('random_trimesh_' + str(len(structures)))
        # surface_trimesh.InitFromTrimesh(rave.TriMesh(surface_trimesh_vertices, surface_trimesh_indices),False)

        surface_trimesh = None

        # if(global_transform[2,2] >= 0):
        #     surface_plane_parameter = [global_transform[0,2],global_transform[1,2],global_transform[2,2],-(np.dot(global_transform[0:3,2],global_transform[0:3,3]))]
        # else:
        #     surface_plane_parameter = [-global_transform[0,2],-global_transform[1,2],-global_transform[2,2],np.dot(global_transform[0:3,2],global_transform[0:3,3])]

        surface_plane_parameter = [global_transform[0,2],global_transform[1,2],global_transform[2,2],-(np.dot(global_transform[0:3,2],global_transform[0:3,3]))]


        random_surface = trimesh_surface(len(structures),surface_plane_parameter,
                                                         surface_vertices,
                                                         surface_boundaries,
                                                         surface_trimesh,
                                                         surface_trimesh_vertices,
                                                         surface_trimesh_indices)
        random_surface.type = surface_type

        # self.env.AddKinBody(random_surface.kinbody)

        structures.append(random_surface)
        # self.DrawSurface(random_surface, transparency=surface_transparancy, style='random_green_yellow_color')
        # self.DrawOrientation(global_transform)

    def construct_tilted_rectangle_wall(self, structures, origin_pose, wall_spacing, max_tilted_angle, wall_length, wall_height=1.5, slope=0):

        surface_boundaries = [(0,1),(1,2),(2,3),(3,0)]
        surface_trimesh_indices = np.array([[0,1,2],[0,2,3]])

        surface_num = int(round(wall_length/wall_spacing))

        for ix in range(surface_num):
            x = (ix+0.5) * wall_spacing

            # surface_transform_bound = [(x,x), (0.0,0.0), (wall_height-0.1-x*slope, wall_height+0.1-x*slope), (-90-max_tilted_angle,-90+max_tilted_angle), (-max_tilted_angle,max_tilted_angle), (0,0)]
            surface_transform_bound = [(x,x), (0.0,0.0), (wall_height-x*slope, wall_height-x*slope), (-90-max_tilted_angle,-90+max_tilted_angle), (-max_tilted_angle,max_tilted_angle), (0,0)]
            surface_transform = [None] * 6

            for i in range(6):
                surface_transform[i] = random.uniform(surface_transform_bound[i][0], surface_transform_bound[i][1])

            surface_transform_matrix = np.dot(xyzrpy_to_SE3(origin_pose), xyzrpy_to_SE3(surface_transform))
            surface_transform = SE3_to_xyzrpy(surface_transform_matrix)

            surface_vertices = [(0.25,0.25),(-0.25,0.25),(-0.25,-0.25),(0.25,-0.25)]

            self.add_quadrilateral_surface(structures,surface_vertices,surface_transform,surface_type='others',surface_transparancy=0.5)

    def construct_vertical_rectangle_wall(self, structures, origin_pose, wall_spacing, max_tilted_angle, wall_length, wall_height=1.5, slope=0):

        surface_boundaries = [(0,1),(1,2),(2,3),(3,0)]
        surface_trimesh_indices = np.array([[0,1,2],[0,2,3]])

        surface_num = int(round(wall_length/wall_spacing))

        for ix in range(surface_num):
            x = (ix+0.5) * wall_spacing

            # surface_transform_bound = [(x,x), (0.0,0.0), (wall_height-0.1-x*slope, wall_height+0.1-x*slope), (-90-max_tilted_angle,-90+max_tilted_angle), (-max_tilted_angle,max_tilted_angle), (0,0)]
            surface_transform_bound = [(x,x), (0.0,0.0), (wall_height-x*slope, wall_height-x*slope), (-90-max_tilted_angle,-90+max_tilted_angle), (-max_tilted_angle,max_tilted_angle), (0,0)]
            surface_transform = [None] * 6

            for i in range(6):
                surface_transform[i] = (surface_transform_bound[i][0] + surface_transform_bound[i][1]) / 2

            surface_transform_matrix = np.dot(xyzrpy_to_SE3(origin_pose), xyzrpy_to_SE3(surface_transform))
            surface_transform = SE3_to_xyzrpy(surface_transform_matrix)

            surface_vertices = [(0.25,0.25),(-0.25,0.25),(-0.25,-0.25),(0.25,-0.25)]

            self.add_quadrilateral_surface(structures,surface_vertices,surface_transform,surface_type='others',surface_transparancy=0.5)

    def update_environment(self,escher,file_path=None,surface_source='dynopt_test_env_1',save_stl=False,save_stl_path='environment_stl/'):

        # remove existing surface kinbodies in openrave
        if(self.structures is not None):
            for struct in self.structures:
                if(struct.kinbody is not None):
                    self.env.Remove(struct.kinbody)

        for i in range(len(self.draw_handles)):
            if(self.draw_handles[i] is not None):
                self.draw_handles[i].SetShow(False)
                self.draw_handles[i].Close()
                self.draw_handles[i] = None
        del self.draw_handles[:]

        structures = []

        if(surface_source == 'flat_corridor_env'):

            self.goal_x = 4.0
            self.goal_y = 0.0

            # add the ground
            ground_max_x = 4.5
            ground_min_x = -1.0
            ground_max_y = 0.75
            ground_min_y = -0.75

            self.add_quadrilateral_surface(structures, [(ground_max_x,ground_min_y),
                                                        (ground_max_x,ground_max_y),
                                                        (ground_min_x,ground_max_y),
                                                        (ground_min_x,ground_min_y)], [0,0,0,0,0,0])

            # add the walls
            wall_max_x = ground_max_x
            wall_min_x = ground_min_x
            wall_max_y = 0.5
            wall_min_y = -0.5

            self.add_quadrilateral_surface(structures, [(wall_max_x,wall_min_y),
                                                        (wall_max_x,wall_max_y),
                                                        (wall_min_x,wall_max_y),
                                                        (wall_min_x,wall_min_y)],
                                                       [0,ground_max_y+0.1,1.2,90,0,0], surface_transparancy=0.5)

            self.add_quadrilateral_surface(structures, [(wall_max_x,wall_min_y),
                                                        (wall_max_x,wall_max_y),
                                                        (wall_min_x,wall_max_y),
                                                        (wall_min_x,wall_min_y)],
                                                       [0,ground_min_y-0.1,1.2,-90,0,0], surface_transparancy=0.5)

        elif(surface_source == 'flat_ground_env'):
            flat_ground_max_x = 4.0
            flat_ground_min_x = -1.0
            flat_ground_max_y = 1.0
            flat_ground_min_y = -1.0

            self.add_quadrilateral_surface(structures, [(flat_ground_max_x,flat_ground_min_y),
                                                        (flat_ground_max_x,flat_ground_max_y),
                                                        (flat_ground_min_x,flat_ground_max_y),
                                                        (flat_ground_min_x,flat_ground_min_y)], [0,0,0,0,0,0])

            self.goal_x = 3.0
            self.goal_y = 0.0

        elif(surface_source == 'large_flat_ground_env'):
            flat_ground_max_x = 4.0
            flat_ground_min_x = -4.0
            flat_ground_max_y = 4.0
            flat_ground_min_y = -4.0

            self.add_quadrilateral_surface(structures, [(flat_ground_max_x,flat_ground_min_y),
                                                        (flat_ground_max_x,flat_ground_max_y),
                                                        (flat_ground_min_x,flat_ground_max_y),
                                                        (flat_ground_min_x,flat_ground_min_y)], [0,0,0,0,0,0])

            self.goal_x = 3.0
            self.goal_y = 0.0

        elif(surface_source == 'stepping_stone_sequence'):
            stepping_stone_start_x = escher.foot_h/2.0+0.15
            stepping_stone_start_y = -0.3

            # initial_platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.5),(stepping_stone_start_x,0.5),(-stepping_stone_start_x,0.5),(-stepping_stone_start_x,-0.5)], [0,0,0,0,0,0])

            # stepping stone
            stepping_stone_size = (0.4,0.3)
            row_num = 3
            col_num = 2
            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                            (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                            (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                            (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row+0.5)*stepping_stone_size[0] + stepping_stone_start_x,
                                            (col+0.5)*stepping_stone_size[1] + stepping_stone_start_y,
                                            random.uniform(-0.05,0.05),
                                        #  random.random() * 20 * np.sign((col+0.5)*stepping_stone_size[1] + stepping_stone_start_y),
                                            random.uniform(-20,20),
                                            random.uniform(-20,20),
                                            0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)

            # final_platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.5),(stepping_stone_start_x,0.5),(-stepping_stone_start_x,0.5),(-stepping_stone_start_x,-0.5)],
                                            [2 * stepping_stone_start_x + row_num * stepping_stone_size[0],0,0,0,0,0])

            self.goal_x = 2 * stepping_stone_start_x + row_num * stepping_stone_size[0]
            self.goal_y = 0.0

        elif(surface_source == 'stepping_stone_sequence_and_stair'):
            stepping_stone_start_x = escher.foot_h/2.0+0.15
            stepping_stone_start_y = -0.3

            # initial_platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.5),(stepping_stone_start_x,0.5),(-stepping_stone_start_x,0.5),(-stepping_stone_start_x,-0.5)], [0,0,0,0,0,0])

            # stepping stone
            stepping_stone_size = (0.4,0.3)
            row_num = 4
            col_num = 2
            surface_projected_vertices = [(stepping_stone_size[0]/2.0, -stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0, stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0, stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0, -stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row+0.5) * stepping_stone_size[0] + stepping_stone_start_x,
                                         (col+0.5) * stepping_stone_size[1] + stepping_stone_start_y,
                                         random.uniform(-0.05, 0.05),
                                         random.uniform(-20, 20),
                                         0,
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)

            # flat platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.5),(stepping_stone_start_x,0.5),(-stepping_stone_start_x,0.5),(-stepping_stone_start_x,-0.5)],
                                            [2 * stepping_stone_start_x + row_num * stepping_stone_size[0],0,0,0,0,0])

            # stairs
            stair_num = 8
            stair_size = (escher.foot_h+0.05,1.0)
            for stair in range(stair_num):
                self.add_quadrilateral_surface(structures, [(stair_size[0]/2.0,-stair_size[1]/2.0),
                                                            (stair_size[0]/2.0,stair_size[1]/2.0),
                                                            (-stair_size[0]/2.0,stair_size[1]/2.0),
                                                            (-stair_size[0]/2.0,-stair_size[1]/2.0)],
                                                           [3 * stepping_stone_start_x + row_num * stepping_stone_size[0] + (stair+0.5) * stair_size[0],
                                                            0,
                                                            stair * 0.1,
                                                            0,0,0])



            self.goal_x = 3 * stepping_stone_start_x + row_num * stepping_stone_size[0] + (stair_num-0.5) * stair_size[0]
            self.goal_y = 0.0

        elif(surface_source == 'dynopt_one_step_test'):
            stepping_stone_start_x = escher.foot_h/2.0+0.03
            stepping_stone_start_y = escher.foot_w/2.0+0.03
            # self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.5),(stepping_stone_start_x,0.5),(-stepping_stone_start_x,0.5),(-stepping_stone_start_x,-0.5)], [0,0,0,0,0,0])

            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-stepping_stone_start_y),
                                                        (stepping_stone_start_x,stepping_stone_start_y),
                                                        (-stepping_stone_start_x,stepping_stone_start_y),
                                                        (-stepping_stone_start_x,-stepping_stone_start_y)],
                                                        [0.025,-0.1,0,-20,0,0])

            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-stepping_stone_start_y),
                                                        (stepping_stone_start_x,stepping_stone_start_y),
                                                        (-stepping_stone_start_x,stepping_stone_start_y),
                                                        (-stepping_stone_start_x,-stepping_stone_start_y)],
                                                        [0.025,0.1,0,-20,0,0])

            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-stepping_stone_start_y),
                                                        (stepping_stone_start_x,stepping_stone_start_y),
                                                        (-stepping_stone_start_x,stepping_stone_start_y),
                                                        (-stepping_stone_start_x,-stepping_stone_start_y)],
                                                        [0.3,-0.1,0,-20,0,0])

            self.goal_x = 0.25
            self.goal_y = 0.0

        elif(surface_source == 'dynopt_test_env_1'):
            stepping_stone_start_x = escher.foot_h/2.0+1.0
            stepping_stone_start_y = -0.8

            stepping_stone_size = (0.4,0.4)
            row_num = 3
            col_num = 4

            # initial_platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-(col_num/2.0)*stepping_stone_size[1]),
                                                        (stepping_stone_start_x,(col_num/2.0)*stepping_stone_size[1]),
                                                        (-0.5,(col_num/2.0)*stepping_stone_size[1]),
                                                        (-0.5,-(col_num/2.0)*stepping_stone_size[1])], [0,0,0,0,0,0])

            # stepping stone
            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]

            for row in range(row_num): # rows of stepping stones forward
                for col in range(0,col_num): # columns of stepping stones
                    surface_transform = [(row+0.5) * stepping_stone_size[0] + stepping_stone_start_x,
                                         (col+0.5) * stepping_stone_size[1] + stepping_stone_start_y,
                                         random.uniform(-0.05, 0.05),
                                         random.uniform(-20, 20),
                                         random.uniform(-20, 20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)

            # final_platform
            self.add_quadrilateral_surface(structures, [(0.5,-(col_num/2.0)*stepping_stone_size[1]),
                                                        (0.5,(col_num/2.0)*stepping_stone_size[1]),
                                                        (-stepping_stone_start_x,(col_num/2.0)*stepping_stone_size[1]),
                                                        (-stepping_stone_start_x,-(col_num/2.0)*stepping_stone_size[1])],
                                                        [2 * stepping_stone_start_x + row_num * stepping_stone_size[0],0,0,0,0,0])

            self.goal_x = 2 * stepping_stone_start_x + row_num * stepping_stone_size[0]
            self.goal_y = 0.0

        elif(surface_source == 'dynopt_test_env_2'):
            stepping_stone_start_x = escher.foot_h/2.0+1.0
            stepping_stone_start_y = -0.3

            # initial_platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-1.0),(stepping_stone_start_x,0.5),(-0.5,0.5),(-0.5,-1.0)], [0,0,0,0,0,0])

            gap = 0.5

            # side wall
            # self.add_quadrilateral_surface(structures, [(-0.5,-0.5),(-0.5,0.5),(0.5,0.5),(0.5,-0.5)],
            #                                [stepping_stone_start_x+gap/2.0, 0.65, 1.3, 90, 0, 0], surface_type='other')


            # final_platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-1.0),
                                                        (stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,-1.0)],
                                                       [2 * stepping_stone_start_x + gap,0,0,0,0,0])

            self.goal_x = 2 * stepping_stone_start_x + gap + 0.4
            self.goal_y = 0.0

        elif(surface_source == 'dynopt_test_env_3'):
            stepping_stone_start_x = escher.foot_h/2.0+1.0
            stepping_stone_start_y = -0.3

            # initial_platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-1.0),
                                                        (stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,-1.0)],
                                                       [0,0,0,0,0,0])

            # stepping stone
            stepping_stone_size = (0.4,0.3)
            row_num = 3
            col_num = 2
            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row+0.5)*stepping_stone_size[0] + stepping_stone_start_x,
                                         (col+0.5)*stepping_stone_size[1] + stepping_stone_start_y,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,0),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)

            # side wall
            wall_length = row_num*stepping_stone_size[0]
            self.add_quadrilateral_surface(structures, [(-wall_length/2.0,-0.5),
                                                        (-wall_length/2.0,0.5),
                                                        (wall_length/2.0,0.5),
                                                        (wall_length/2.0,-0.5)],
                                                       [stepping_stone_start_x+wall_length/2.0, 0.8, 1.3, 90, 0, 0], surface_type='other')


            # final_platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-1.0),
                                                        (stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,-1.0)],
                                                       [2 * stepping_stone_start_x + row_num * stepping_stone_size[0],0,0,0,0,0])

            self.goal_x = 2 * stepping_stone_start_x + row_num * stepping_stone_size[0]
            self.goal_y = 0.0

        elif(surface_source == 'dynopt_test_env_4'):
            stepping_stone_start_x = escher.foot_h/2.0 + 1.0
            stepping_stone_start_y = -0.3

            # initial_platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.3),
                                                        (stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,-0.3)],
                                                       [0,0,0,0,0,0])

            # tilted flat connector
            stepping_stone_size = (0.4,0.3)
            row_num = 3
            col_num = 2
            flat_connector_half_x = (row_num/2.0) * stepping_stone_size[0]
            flat_connector_half_y = (col_num/2.0) * stepping_stone_size[1]
            self.add_quadrilateral_surface(structures, [(flat_connector_half_x,-flat_connector_half_y),
                                                        (flat_connector_half_x,flat_connector_half_y),
                                                        (-flat_connector_half_x,flat_connector_half_y),
                                                        (-flat_connector_half_x,-flat_connector_half_y)],
                                                       [stepping_stone_start_x + flat_connector_half_x,0.1,0,-30,0,0])


            # side wall
            wall_length = row_num*stepping_stone_size[0]
            self.add_quadrilateral_surface(structures, [(-wall_length/2.0,-0.5),
                                                        (-wall_length/2.0,0.5),
                                                        (wall_length/2.0,0.5),
                                                        (wall_length/2.0,-0.5)],
                                                       [stepping_stone_start_x+wall_length/2.0, 0.8, 1.3, 90, 0, 0], surface_type='other')


            # final_platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.3),
                                                        (stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,-0.3)],
                                                       [2 * stepping_stone_start_x + row_num * stepping_stone_size[0],0,0,0,0,0])

            self.goal_x = 2 * stepping_stone_start_x + row_num * stepping_stone_size[0]
            self.goal_y = 0.0

        elif(surface_source == 'dynopt_test_env_5'):
            stepping_stone_start_x = escher.foot_h/2.0+1.0
            stepping_stone_start_y = -0.45

            # initial_platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.5),
                                                        (stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,-0.5)],
                                                       [0,0,0,0,0,0])

            # stepping stones
            stepping_stone_size = (0.4,0.3)
            row_num = 3
            col_num = 3
            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row+0.5)*stepping_stone_size[0] + stepping_stone_start_x,
                                         (col+0.5)*stepping_stone_size[1] + stepping_stone_start_y,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)


            # side wall
            wall_length = row_num*stepping_stone_size[0]
            self.add_quadrilateral_surface(structures, [(-wall_length/2.0,-0.5),
                                                        (-wall_length/2.0,0.5),
                                                        (wall_length/2.0,0.5),
                                                        (wall_length/2.0,-0.5)],
                                                       [stepping_stone_start_x+wall_length/2.0, 0.8, 1.3, 90, 0, 0], surface_type='other')

            self.add_quadrilateral_surface(structures, [(-wall_length/2.0,-0.5),
                                                        (-wall_length/2.0,0.5),
                                                        (wall_length/2.0,0.5),
                                                        (wall_length/2.0,-0.5)],
                                                       [stepping_stone_start_x+wall_length/2.0, -0.8, 1.3, -90, 0, 0], surface_type='other')


            # final_platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.5),
                                                        (stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,-0.5)],
                                                       [2 * stepping_stone_start_x + row_num * stepping_stone_size[0],0,0,0,0,0])

            self.goal_x = 2 * stepping_stone_start_x + row_num * stepping_stone_size[0]
            self.goal_y = 0.0

        elif(surface_source == 'dynopt_test_env_6'):
            stepping_stone_start_x = escher.foot_h/2.0+0.3
            stepping_stone_start_y = -0.45

            # initial_platform
            # self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.5),(stepping_stone_start_x,0.5),(-stepping_stone_start_x,0.5),(-stepping_stone_start_x,-0.5)], [0,0,0,0,0,0])
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.25),
                                                        (stepping_stone_start_x,0.25),
                                                        (-stepping_stone_start_x,0.25),
                                                        (-stepping_stone_start_x,-0.25)],
                                                       [0,0.25,0,random.uniform(-20,20),random.uniform(-20,20),0])

            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.25),
                                                        (stepping_stone_start_x,0.25),
                                                        (-stepping_stone_start_x,0.25),
                                                        (-stepping_stone_start_x,-0.25)],
                                                       [0,-0.25,0,random.uniform(-20,20),random.uniform(-20,20),0])

            # stepping stones
            stepping_stone_size = (0.4,0.3)
            row_num = 3
            col_num = 3
            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row+0.5)*stepping_stone_size[0] + stepping_stone_start_x,
                                         (col+0.5)*stepping_stone_size[1] + stepping_stone_start_y,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)


            # side wall
            wall_length = row_num*stepping_stone_size[0] + 1.3
            self.construct_tilted_rectangle_wall(structures, [stepping_stone_start_x + 0.5*row_num*stepping_stone_size[0]/2.0 - wall_length/2.0, -0.65, 0, 0, 0, 0], 0.5, 20, wall_length, slope=0)
            self.construct_tilted_rectangle_wall(structures, [stepping_stone_start_x + 0.5*row_num*stepping_stone_size[0]/2.0 + wall_length/2.0, 0.65, 0, 0, 0, 180], 0.5, 20, wall_length, slope=0)

            # self.add_quadrilateral_surface(structures, [(-wall_length/2.0,-0.5),(-wall_length/2.0,0.5),(wall_length/2.0,0.5),(wall_length/2.0,-0.5)],
            #                                [stepping_stone_start_x+wall_length/2.0, 0.8, 1.3, 90, 0, 0], surface_type='other')

            # self.add_quadrilateral_surface(structures, [(-wall_length/2.0,-0.5),(-wall_length/2.0,0.5),(wall_length/2.0,0.5),(wall_length/2.0,-0.5)],
            #                                [stepping_stone_start_x+wall_length/2.0, -0.8, 1.3, -90, 0, 0], surface_type='other')


            # final_platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.5),
                                                        (stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,-0.5)],
                                                       [2 * stepping_stone_start_x + row_num * stepping_stone_size[0],0,0,0,0,0])

            self.goal_x = 2 * stepping_stone_start_x + row_num * stepping_stone_size[0]
            self.goal_y = 0.0

        elif(surface_source == 'dynopt_test_env_7'):

            slope_angle = -30
            slope_angle_rad = slope_angle * DEG2RAD
            slope_start_x = 0.5
            slope_length = 3.0
            slope_width = 1.0
            final_platform_length = 3.0

            # initial_platform
            self.add_quadrilateral_surface(structures, [(slope_start_x,-slope_width/2.0),
                                                        (slope_start_x,slope_width/2.0),
                                                        (-0.5,slope_width/2.0),
                                                        (-0.5,-slope_width/2.0)],
                                                       [0,0,0,0,0,0])

            # slope
            self.add_quadrilateral_surface(structures, [(slope_length/2.0,-slope_width/2.0),
                                                        (slope_length/2.0,slope_width/2.0),
                                                        (-slope_length/2.0,slope_width/2.0),
                                                        (-slope_length/2.0,-slope_width/2.0)],
                                                        [slope_start_x + slope_length/2.0 * math.cos(slope_angle_rad),
                                                        0,
                                                        slope_length/2.0 * math.sin(slope_angle_rad),
                                                        0, -slope_angle, 0])

            # final_platform
            self.add_quadrilateral_surface(structures, [(final_platform_length/2.0,-slope_width/2.0),
                                                        (final_platform_length/2.0,slope_width/2.0),
                                                        (-final_platform_length/2.0,slope_width/2.0),
                                                        (-final_platform_length/2.0,-slope_width/2.0)],
                                                        [slope_start_x + slope_length * math.cos(slope_angle_rad) + final_platform_length/2.0,
                                                        0,
                                                        slope_length * math.sin(slope_angle_rad),
                                                        0, 0, 0])

            self.goal_x = slope_start_x + slope_length * math.cos(slope_angle_rad) + final_platform_length - 0.4
            self.goal_y = 0.0

        elif(surface_source == 'dynopt_test_env_8'):
            stepping_stone_start_x = escher.foot_h/2.0+0.3
            stepping_stone_start_y = -0.45

            # initial_platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.5),
                                                        (stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,-0.5)],
                                                       [0,0,0,0,0,0])
            # self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.25),(stepping_stone_start_x,0.25),(-stepping_stone_start_x,0.25),(-stepping_stone_start_x,-0.25)], [0,0.25,0,random.uniform(-20,20),random.uniform(-20,20),0])
            # self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.25),(stepping_stone_start_x,0.25),(-stepping_stone_start_x,0.25),(-stepping_stone_start_x,-0.25)], [0,-0.25,0,random.uniform(-20,20),random.uniform(-20,20),0])

            # stepping stones
            stepping_stone_size = (0.4,0.3)
            row_num = 5
            col_num = 3
            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row+0.5)*stepping_stone_size[0] + stepping_stone_start_x,
                                         (col+0.5)*stepping_stone_size[1] + stepping_stone_start_y,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)


            # side wall
            wall_length = row_num*stepping_stone_size[0]
            self.construct_tilted_rectangle_wall(structures, [stepping_stone_start_x + 0.5*row_num*stepping_stone_size[0] - wall_length/2.0, -0.7, 0, 0, 0, 0], 0.5, 20, wall_length, slope=0)
            self.construct_tilted_rectangle_wall(structures, [stepping_stone_start_x + 0.5*row_num*stepping_stone_size[0] + wall_length/2.0, 0.7, 0, 0, 0, 180], 0.5, 20, wall_length, slope=0)

            # self.add_quadrilateral_surface(structures, [(-wall_length/2.0,-0.5),(-wall_length/2.0,0.5),(wall_length/2.0,0.5),(wall_length/2.0,-0.5)],
            #                                [stepping_stone_start_x+wall_length/2.0, 0.8, 1.3, 90, 0, 0], surface_type='other')

            # self.add_quadrilateral_surface(structures, [(-wall_length/2.0,-0.5),(-wall_length/2.0,0.5),(wall_length/2.0,0.5),(wall_length/2.0,-0.5)],
            #                                [stepping_stone_start_x+wall_length/2.0, -0.8, 1.3, -90, 0, 0], surface_type='other')


            # final_platform
            self.add_quadrilateral_surface(structures, [(1.0,-0.5),(1.0,0.5),(-stepping_stone_start_x,0.5),(-stepping_stone_start_x,-0.5)],
                                                       [2 * stepping_stone_start_x + row_num * stepping_stone_size[0],0,0,0,0,0])

            self.goal_x = 2 * stepping_stone_start_x + row_num * stepping_stone_size[0] + 0.8
            self.goal_y = 0.0

        elif(surface_source == 'dynopt_test_env_9'): # new test environment for using shorter arm span
            stepping_stone_start_x = escher.foot_h/2.0+0.3
            stepping_stone_start_y = -0.45

            # initial_platform
            # self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.5),(stepping_stone_start_x,0.5),(-stepping_stone_start_x,0.5),(-stepping_stone_start_x,-0.5)], [0,0,0,0,0,0])
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.25),
                                                        (stepping_stone_start_x,0.25),
                                                        (-stepping_stone_start_x,0.25),
                                                        (-stepping_stone_start_x,-0.25)],
                                                       [0,0.25,0,random.uniform(-20,20),random.uniform(-20,20),0])

            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.25),
                                                        (stepping_stone_start_x,0.25),
                                                        (-stepping_stone_start_x,0.25),
                                                        (-stepping_stone_start_x,-0.25)],
                                                       [0,-0.25,0,random.uniform(-20,20),random.uniform(-20,20),0])

            # stepping stones
            stepping_stone_size = (0.4,0.3)
            row_num = 3
            col_num = 3
            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row+0.5)*stepping_stone_size[0] + stepping_stone_start_x,
                                         (col+0.5)*stepping_stone_size[1] + stepping_stone_start_y,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)


            # side wall
            wall_length = row_num*stepping_stone_size[0] + 1.3
            # self.construct_tilted_rectangle_wall(structures, [stepping_stone_start_x + 0.5*row_num*stepping_stone_size[0]/2.0 - wall_length/2.0, -0.55, 0, 0, 0, 0], 0.5, 20, wall_length, wall_height=1.3, slope=0)
            # self.construct_tilted_rectangle_wall(structures, [stepping_stone_start_x + 0.5*row_num*stepping_stone_size[0]/2.0 + wall_length/2.0, 0.65, 0, 0, 0, 180], 0.5, 20, wall_length, slope=0)

            # self.add_quadrilateral_surface(structures, [(-wall_length/2.0,-0.5),(-wall_length/2.0,0.5),(wall_length/2.0,0.5),(wall_length/2.0,-0.5)],
            #                                [stepping_stone_start_x+wall_length/2.0, 0.8, 1.3, 90, 0, 0], surface_type='other')

            # self.add_quadrilateral_surface(structures, [(-wall_length/2.0,-0.5),(-wall_length/2.0,0.5),(wall_length/2.0,0.5),(wall_length/2.0,-0.5)],
            #                                [stepping_stone_start_x+wall_length/2.0, -0.8, 1.3, -90, 0, 0], surface_type='other')


            # final_platform
            self.add_quadrilateral_surface(structures, [(stepping_stone_start_x,-0.5),
                                                        (stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,0.5),
                                                        (-stepping_stone_start_x,-0.5)],
                                                       [2 * stepping_stone_start_x + row_num * stepping_stone_size[0],0,0,0,0,0])

            self.goal_x = 2 * stepping_stone_start_x + row_num * stepping_stone_size[0]
            self.goal_y = 0.0

        elif(surface_source == 'capture_test_env_1'): # a room for the robot to go from one end to the other

            # stepping stones
            stepping_stone_size = (1.0,1.0)
            row_num = 3
            col_num = 3
            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones

                    # if row == 1 and col == 1:
                    #     continue

                    surface_transform = [row*stepping_stone_size[0],
                                         col*stepping_stone_size[1],
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)


            # side wall
            x_wall_length = row_num*stepping_stone_size[0]
            self.construct_tilted_rectangle_wall(structures, [0.5*row_num*stepping_stone_size[0] - x_wall_length/2.0 - 0.2, -0.5*stepping_stone_size[1] - 0.25, 0, 0, 0, 0], 0.5, 20, x_wall_length, wall_height=1.3, slope=0)
            self.construct_tilted_rectangle_wall(structures, [0.5*row_num*stepping_stone_size[0] + x_wall_length/2.0 - 0.2, (col_num-0.5)*stepping_stone_size[1] + 0.25, 0, 0, 0, 180], 0.5, 20, x_wall_length, wall_height=1.3, slope=0)

            y_wall_length = col_num*stepping_stone_size[1]
            self.construct_tilted_rectangle_wall(structures, [-0.5*stepping_stone_size[0] - 0.25, 0.5*col_num*stepping_stone_size[1] + y_wall_length/2.0 - 0.3, 0, 0, 0, 270], 0.5, 20, y_wall_length, wall_height=1.3, slope=0)
            self.construct_tilted_rectangle_wall(structures, [(row_num-0.5)*stepping_stone_size[0] + 0.25, 0.5*col_num*stepping_stone_size[1] - y_wall_length/2.0 - 0.3, 0, 0, 0, 90], 0.5, 20, y_wall_length, wall_height=1.3, slope=0)

            self.goal_x = (row_num-1) * stepping_stone_size[0]
            self.goal_y = (col_num-1) * stepping_stone_size[1]
            # self.goal_y = 0

        elif(surface_source == 'capture_test_env_2'): # a room for the robot to go from one end to the other

            corridor_length = 2.0
            corridor_start_x = 0.8
            narrow_corridor_width = 0.3
            wide_corridor_width = 0.8
            wide_corridor_y = -1.5

            # initial platform
            self.add_quadrilateral_surface(structures, [(-0.2,narrow_corridor_width/2.0),
                                                        (-0.2,wide_corridor_y-wide_corridor_width/2.0),
                                                        (corridor_start_x,wide_corridor_y-wide_corridor_width/2.0),
                                                        (corridor_start_x,narrow_corridor_width/2.0)],
                                                       [0,0,0,0,0,0])

            # narrow corridor
            self.add_quadrilateral_surface(structures, [(corridor_start_x,narrow_corridor_width/2.0),
                                                        (corridor_start_x,-narrow_corridor_width/2.0),
                                                        (corridor_start_x+corridor_length,-narrow_corridor_width/2.0),
                                                        (corridor_start_x+corridor_length,narrow_corridor_width/2.0)],
                                                       [0,0,0,0,0,0])

            # wide corridor
            self.add_quadrilateral_surface(structures, [(corridor_start_x,wide_corridor_y+wide_corridor_width/2.0),
                                                        (corridor_start_x,wide_corridor_y-wide_corridor_width/2.0),
                                                        (corridor_start_x+corridor_length,wide_corridor_y-wide_corridor_width/2.0),
                                                        (corridor_start_x+corridor_length,wide_corridor_y+wide_corridor_width/2.0)],
                                                       [0,0,0,0,0,0])

            # final platform
            self.add_quadrilateral_surface(structures, [(corridor_start_x+corridor_length,narrow_corridor_width/2.0),
                                                        (corridor_start_x+corridor_length,wide_corridor_y-wide_corridor_width/2.0),
                                                        (2*corridor_start_x+corridor_length+0.2,wide_corridor_y-wide_corridor_width/2.0),
                                                        (2*corridor_start_x+corridor_length+0.2,narrow_corridor_width/2.0)],
                                                       [0,0,0,0,0,0])

            self.goal_x = corridor_start_x * 2 + corridor_length
            self.goal_y = 0

        elif(surface_source == 'capture_test_env_3'): # a room for the robot to go from one end to the other

            corridor_length = 2.0
            corridor_width = 1.0

            # initial platform
            self.add_quadrilateral_surface(structures, [(-0.2,corridor_width/2.0),
                                                        (-0.2,-corridor_width/2.0),
                                                        (corridor_length,-corridor_width/2.0),
                                                        (corridor_length,corridor_width/2.0)],
                                                        [0,0,0,0,0,0])

            # left wall
            self.add_quadrilateral_surface(structures, [(-0.2,corridor_width/2.0),
                                                        (-0.2,-corridor_width/2.0),
                                                        (corridor_length,-corridor_width/2.0),
                                                        (corridor_length,corridor_width/2.0)],
                                                        [0,corridor_width/2.0+0.1,1.3,90,0,0],
                                                        surface_type='others')

            # right wall
            # self.add_quadrilateral_surface(structures, [(-0.2,corridor_width/2.0),
            #                                             (-0.2,-corridor_width/2.0),
            #                                             (corridor_length,-corridor_width/2.0),
            #                                             (corridor_length,corridor_width/2.0)],
            #                                             [0,-corridor_width/2.0-0.1,1.3,-90,0,0],
            #                                             surface_type='others')

            self.goal_x = corridor_length - 0.2
            self.goal_y = 0


        # overview of each env type
        # one_step_env_0: no wall, small side
        # one_step_env_1: no wall, medium side
        # one_step_env_2: no wall, large side
        # one_step_env_3: only left wall, small side
        # one_step_env_4: only left wall, medium side
        # one_step_env_5: only left wall, large side
        # one_step_env_6: only right wall, small side
        # one_step_env_7: only right wall, medium side
        # one_step_env_8: only right wall, large side
        # one_step_env_9: two wall, small side
        # one_step_env_10: two wall, medium side
        # one_step_env_11: two wall, large side

        elif surface_source == 'one_step_env_0':
            # see transitions of type 0
            # stepping stones
            stepping_stone_size = (0.4, 0.4)
            row_num = 7
            col_num = 7

            x_random = random.uniform(-0.2, 0.2)
            y_random = random.uniform(-0.2, 0.2)

            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row - row_num // 2)*stepping_stone_size[0] + x_random,
                                         (col - col_num // 2)*stepping_stone_size[1] + y_random,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)


        elif surface_source == 'one_step_env_3':
            # see transitions of type [0,1,2,4,5], mostly 2
            # stepping stones
            stepping_stone_size = (0.4, 0.4)
            row_num = 7
            col_num = 7

            x_random = random.uniform(-0.2, 0.2)
            y_random = random.uniform(-1.0, -0.4)

            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row - row_num // 2)*stepping_stone_size[0] + x_random,
                                         (col - col_num // 2)*stepping_stone_size[1] + y_random,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)
            # side wall
            x_wall_length = row_num*stepping_stone_size[0]
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] + x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (col_num-0.5-1)*stepping_stone_size[1] - stepping_stone_size[1] * 2 + y_random, 0, 0, 0, 180], 0.5, 20, x_wall_length, wall_height=1.15, slope=0)
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] + x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (col_num-0.5-1)*stepping_stone_size[1] - stepping_stone_size[1] * 2 + y_random, 0, 0, 0, 180], 0.5, 20, x_wall_length, wall_height=1.65, slope=0)


        elif surface_source == 'one_step_env_6':
            # see transitions of type [0,3], mostly 3
            # stepping stones
            stepping_stone_size = (0.4, 0.4)
            row_num = 7
            col_num = 7

            x_random = random.uniform(-0.2, 0.2)
            y_random = random.uniform(0.4, 1.0)

            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row - row_num // 2)*stepping_stone_size[0] + x_random,
                                         (col - col_num // 2)*stepping_stone_size[1] + y_random,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)
            # side wall
            x_wall_length = row_num*stepping_stone_size[0]
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] - x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (-0.5-1)*stepping_stone_size[1] - stepping_stone_size[1] * 2 + y_random, 0, 0, 0, 0], 0.5, 20, x_wall_length, wall_height=1.15, slope=0)
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] - x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (-0.5-1)*stepping_stone_size[1] - stepping_stone_size[1] * 2 + y_random, 0, 0, 0, 0], 0.5, 20, x_wall_length, wall_height=1.65, slope=0)


        elif surface_source == 'one_step_env_9':
            # see transitions of type 0-9, mostly 9
            # stepping stones
            stepping_stone_size = (0.4, 0.4)
            row_num = 7
            col_num = 5

            x_random = random.uniform(-0.2, 0.2)
            y_random = random.uniform(-0.2, 0.2)
            in_random = random.uniform(0.0, 0.6)

            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row - row_num // 2)*stepping_stone_size[0] + x_random,
                                         (col - col_num // 2)*stepping_stone_size[1] + y_random,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)
            # side wall
            x_wall_length = row_num*stepping_stone_size[0]
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] + x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (col_num-0.5-1)*stepping_stone_size[1] + y_random - stepping_stone_size[1] - in_random, 0, 0, 0, 180], 0.5, 20, x_wall_length, wall_height=1.15, slope=0)
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] - x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (-0.5-1)*stepping_stone_size[1] + y_random - stepping_stone_size[1] + in_random, 0, 0, 0, 0], 0.5, 20, x_wall_length, wall_height=1.15, slope=0)
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] + x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (col_num-0.5-1)*stepping_stone_size[1] + y_random - stepping_stone_size[1] - in_random, 0, 0, 0, 180], 0.5, 20, x_wall_length, wall_height=1.65, slope=0)
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] - x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (-0.5-1)*stepping_stone_size[1] + y_random - stepping_stone_size[1] + in_random, 0, 0, 0, 0], 0.5, 20, x_wall_length, wall_height=1.65, slope=0)


        elif surface_source == 'one_step_env_12':
            # stepping stones
            stepping_stone_size = (0.4, 0.4)
            row_num = 5
            col_num = 7

            x_random = random.uniform(-0.7, 1.0)
            y_random = random.uniform(-0.2, 0.2)

            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row - 5)*stepping_stone_size[0] + x_random,
                                         (col - col_num // 2)*stepping_stone_size[1] + y_random,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)
            
            vertical_surface_vertices = [(-0.2,1.4),(1.8,1.4),(1.8,-1.4),(-0.2,-1.4)]
            self.add_quadrilateral_surface(structures, vertical_surface_vertices, [x_random,y_random,0,0,0,0])


        elif surface_source == 'one_step_env_15':
            # stepping stones
            stepping_stone_size = (0.4, 0.4)
            row_num = 5
            col_num = 7

            x_random = random.uniform(-0.7, 1.0)
            y_random = random.uniform(-1.0, -0.4)

            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row - 5)*stepping_stone_size[0] + x_random,
                                         (col - col_num // 2)*stepping_stone_size[1] + y_random,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)
            
            vertical_surface_vertices = [(-0.2,1.4),(1.8,1.4),(1.8,-1.4),(-0.2,-1.4)]
            self.add_quadrilateral_surface(structures, vertical_surface_vertices, [x_random,y_random,0,0,0,0])

            # side wall
            x_wall_length = 11*stepping_stone_size[0]
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] + x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (col_num-0.5-1)*stepping_stone_size[1] - stepping_stone_size[1] * 2 + y_random, 0, 0, 0, 180], 0.5, 20, x_wall_length, wall_height=1.15, slope=0)
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] + x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (col_num-0.5-1)*stepping_stone_size[1] - stepping_stone_size[1] * 2 + y_random, 0, 0, 0, 180], 0.5, 20, x_wall_length, wall_height=1.65, slope=0)

            
        elif surface_source == 'one_step_env_18':
            # stepping stones
            stepping_stone_size = (0.4, 0.4)
            row_num = 5
            col_num = 7

            x_random = random.uniform(-0.7, 1.0)
            y_random = random.uniform(0.4, 1.0)

            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row - 5)*stepping_stone_size[0] + x_random,
                                         (col - col_num // 2)*stepping_stone_size[1] + y_random,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)
            
            vertical_surface_vertices = [(-0.2,1.4),(1.8,1.4),(1.8,-1.4),(-0.2,-1.4)]
            self.add_quadrilateral_surface(structures, vertical_surface_vertices, [x_random,y_random,0,0,0,0])

            # side wall
            x_wall_length = 11*stepping_stone_size[0]
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] - x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (-0.5-1)*stepping_stone_size[1] - stepping_stone_size[1] * 2 + y_random, 0, 0, 0, 0], 0.5, 20, x_wall_length, wall_height=1.15, slope=0)
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] - x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (-0.5-1)*stepping_stone_size[1] - stepping_stone_size[1] * 2 + y_random, 0, 0, 0, 0], 0.5, 20, x_wall_length, wall_height=1.65, slope=0)


        elif surface_source == 'one_step_env_21':
            # stepping stones
            stepping_stone_size = (0.4, 0.4)
            row_num = 5
            col_num = 5

            x_random = random.uniform(-0.7, 1.0)
            y_random = random.uniform(-0.2, 0.2)
            in_random = random.uniform(0.0, 0.6)

            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row - 5)*stepping_stone_size[0] + x_random,
                                         (col - 2)*stepping_stone_size[1] + y_random,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)
            
            vertical_surface_vertices = [(-0.2,1.0),(1.8,1.0),(1.8,-1.0),(-0.2,-1.0)]
            self.add_quadrilateral_surface(structures, vertical_surface_vertices, [x_random,y_random,0,0,0,0])

            # side wall
            x_wall_length = 11*stepping_stone_size[0]
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] + x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (col_num-0.5-1)*stepping_stone_size[1] + y_random - stepping_stone_size[1] - in_random, 0, 0, 0, 180], 0.5, 20, x_wall_length, wall_height=1.15, slope=0)
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] - x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (-0.5-1)*stepping_stone_size[1] + y_random - stepping_stone_size[1] + in_random, 0, 0, 0, 0], 0.5, 20, x_wall_length, wall_height=1.15, slope=0)
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] + x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (col_num-0.5-1)*stepping_stone_size[1] + y_random - stepping_stone_size[1] - in_random, 0, 0, 0, 180], 0.5, 20, x_wall_length, wall_height=1.65, slope=0)
            self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] - x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (-0.5-1)*stepping_stone_size[1] + y_random - stepping_stone_size[1] + in_random, 0, 0, 0, 0], 0.5, 20, x_wall_length, wall_height=1.65, slope=0)            


        elif surface_source == 'torso_path_planning_test_env_0':

            self.goal_x = 4.0
            self.goal_y = 0.0

            lateral_surface_vertices = [(0.2,-0.5),(0.2,0.2),(-0.2,0.2),(-0.2,-0.5)]
            vertical_surface_vertices = [(2.2,-0.2),(2.2,0.2),(-2.2,0.2),(-2.2,-0.2)]
            # vertical_surface_vertices = [(1.0,-0.2),(1.0,0.2),(-1.0,0.2),(-1.0,-0.2)]

            self.add_quadrilateral_surface(structures, lateral_surface_vertices, [0,0,0,0,0,0])
            self.add_quadrilateral_surface(structures, lateral_surface_vertices, [1.0,0,0,0,0,0])
            self.add_quadrilateral_surface(structures, lateral_surface_vertices, [2.0,0,0,0,0,0])
            self.add_quadrilateral_surface(structures, lateral_surface_vertices, [3.0,0,0,0,0,0])
            self.add_quadrilateral_surface(structures, lateral_surface_vertices, [4.0,0,0,0,0,0])

            self.add_quadrilateral_surface(structures, vertical_surface_vertices, [2.0,-0.7,0,0,0,0])
            self.add_quadrilateral_surface(structures, vertical_surface_vertices, [2.0,0.4,0,0,0,0])
            # self.add_quadrilateral_surface(structures, vertical_surface_vertices, [1.0,-0.7,0,0,0,0])
            # self.add_quadrilateral_surface(structures, vertical_surface_vertices, [3.0,0.4,0,0,0,0])


            # # side wall
            # x_wall_length = row_num*stepping_stone_size[0]
            self.construct_tilted_rectangle_wall(structures, [-0.2, -1.0, 0, 0, 0, 0], 0.5, 20, 4.4, wall_height=1.25, slope=0)
            self.construct_tilted_rectangle_wall(structures, [-0.2, -1.0, 0, 0, 0, 0], 0.5, 20, 4.4, wall_height=1.75, slope=0)
            # self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] + x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (col_num-0.5-1)*stepping_stone_size[1] - stepping_stone_size[1] * 2 + y_random, 0, 0, 0, 180], 0.5, 20, x_wall_length, wall_height=1.65, slope=0)


        elif surface_source == 'torso_path_planning_test_env_1':

            self.goal_x = 1.0
            self.goal_y = 0.0

            lateral_surface_vertices = [(0.2,-0.5),(0.2,0.2),(-0.2,0.2),(-0.2,-0.5)]
            vertical_surface_vertices = [(-0.8,-0.2),(-0.8,0.2),(-2.2,0.2),(-2.2,-0.2)]
            # vertical_surface_vertices = [(1.0,-0.2),(1.0,0.2),(-1.0,0.2),(-1.0,-0.2)]

            self.add_quadrilateral_surface(structures, lateral_surface_vertices, [0,0,0,0,0,0])
            self.add_quadrilateral_surface(structures, lateral_surface_vertices, [1.0,0,0,0,0,0])
            # self.add_quadrilateral_surface(structures, lateral_surface_vertices, [2.0,0,0,0,0,0])
            # self.add_quadrilateral_surface(structures, lateral_surface_vertices, [3.0,0,0,0,0,0])
            # self.add_quadrilateral_surface(structures, lateral_surface_vertices, [4.0,0,0,0,0,0])

            self.add_quadrilateral_surface(structures, vertical_surface_vertices, [2.0,-0.7,0,0,0,0])
            self.add_quadrilateral_surface(structures, vertical_surface_vertices, [2.0,0.4,0,0,0,0])
            # self.add_quadrilateral_surface(structures, vertical_surface_vertices, [1.0,-0.7,0,0,0,0])
            # self.add_quadrilateral_surface(structures, vertical_surface_vertices, [3.0,0.4,0,0,0,0])


            # # side wall
            # x_wall_length = row_num*stepping_stone_size[0]
            self.construct_tilted_rectangle_wall(structures, [-0.2, -1.0, 0, 0, 0, 0], 0.5, 20, 1.4, wall_height=1.15, slope=0)
            # self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] + x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (col_num-0.5-1)*stepping_stone_size[1] - stepping_stone_size[1] * 2 + y_random, 0, 0, 0, 180], 0.5, 20, x_wall_length, wall_height=1.65, slope=0)


        elif surface_source == 'torso_path_planning_test_env_2':

            self.goal_x = 1.0
            self.goal_y = 0.0

            # lateral_surface_vertices = [(0.2,-0.5),(0.2,0.2),(-0.2,0.2),(-0.2,-0.5)]
            vertical_surface_vertices = [(-0.8,-1.4),(-0.8,0.2),(-2.2,0.2),(-2.2,-1.4)]
            # vertical_surface_vertices = [(1.0,-0.2),(1.0,0.2),(-1.0,0.2),(-1.0,-0.2)]

            # self.add_quadrilateral_surface(structures, lateral_surface_vertices, [0,0,0,0,0,0])
            # self.add_quadrilateral_surface(structures, lateral_surface_vertices, [1.0,0,0,0,0,0])
            # self.add_quadrilateral_surface(structures, lateral_surface_vertices, [2.0,0,0,0,0,0])
            # self.add_quadrilateral_surface(structures, lateral_surface_vertices, [3.0,0,0,0,0,0])
            # self.add_quadrilateral_surface(structures, lateral_surface_vertices, [4.0,0,0,0,0,0])

            # self.add_quadrilateral_surface(structures, vertical_surface_vertices, [2.0,-0.7,0,0,0,0])
            self.add_quadrilateral_surface(structures, vertical_surface_vertices, [2.0,0.4,0,0,0,0])
            # self.add_quadrilateral_surface(structures, vertical_surface_vertices, [1.0,-0.7,0,0,0,0])
            # self.add_quadrilateral_surface(structures, vertical_surface_vertices, [3.0,0.4,0,0,0,0])


            # # side wall
            # x_wall_length = row_num*stepping_stone_size[0]
            self.construct_tilted_rectangle_wall(structures, [-0.2, -1.0, 0, 0, 0, 0], 0.5, 20, 1.4, wall_height=1.15, slope=0)
            # self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] + x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (col_num-0.5-1)*stepping_stone_size[1] - stepping_stone_size[1] * 2 + y_random, 0, 0, 0, 180], 0.5, 20, x_wall_length, wall_height=1.65, slope=0)


        elif surface_source == 'torso_path_planning_test_env_3':

            self.goal_x = 4.0
            self.goal_y = 0.0

            lateral_surface_vertices = [(0.2,-0.5),(0.2,0.2),(-0.2,0.2),(-0.2,-0.5)]
            vertical_surface_vertices = [(2.2,-0.2),(2.2,0.2),(-2.2,0.2),(-2.2,-0.2)]
            # vertical_surface_vertices = [(1.0,-0.2),(1.0,0.2),(-1.0,0.2),(-1.0,-0.2)]

            self.add_quadrilateral_surface(structures, lateral_surface_vertices, [0,0,0,0,0,0])
            self.add_quadrilateral_surface(structures, lateral_surface_vertices, [1.0,0,0,0,0,0])
            self.add_quadrilateral_surface(structures, lateral_surface_vertices, [2.0,0,0,0,0,0])
            self.add_quadrilateral_surface(structures, lateral_surface_vertices, [3.0,0,0,0,0,0])
            self.add_quadrilateral_surface(structures, lateral_surface_vertices, [4.0,0,0,0,0,0])

            self.add_quadrilateral_surface(structures, vertical_surface_vertices, [2.0,-0.7,0,0,0,0])
            self.add_quadrilateral_surface(structures, vertical_surface_vertices, [2.0,0.4,0,0,0,0])
            # self.add_quadrilateral_surface(structures, vertical_surface_vertices, [1.0,-0.7,0,0,0,0])
            # self.add_quadrilateral_surface(structures, vertical_surface_vertices, [3.0,0.4,0,0,0,0])


            # # side wall
            # x_wall_length = row_num*stepping_stone_size[0]
            self.construct_vertical_rectangle_wall(structures, [-0.2, -1.0, 0, 0, 0, 0], 0.5, 20, 4.4, wall_height=1.25, slope=0)
            self.construct_vertical_rectangle_wall(structures, [-0.2, -1.0, 0, 0, 0, 0], 0.5, 20, 4.4, wall_height=1.75, slope=0)
            # self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] + x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (col_num-0.5-1)*stepping_stone_size[1] - stepping_stone_size[1] * 2 + y_random, 0, 0, 0, 180], 0.5, 20, x_wall_length, wall_height=1.65, slope=0)


        elif surface_source == 'torso_path_planning_test_env_4':

            self.goal_x = 4.0
            self.goal_y = 0.0

            stepping_stone_size = (0.4, 0.4)
            row_num = 11
            col_num = 6

            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row - row_num // 2)*stepping_stone_size[0] + 2,
                                         (col - col_num // 2)*stepping_stone_size[1] + 0.4,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)

            # # side wall
            # x_wall_length = row_num*stepping_stone_size[0]
            self.construct_tilted_rectangle_wall(structures, [-0.2, -1.0, 0, 0, 0, 0], 0.5, 20, 4.4, wall_height=1.25, slope=0)
            self.construct_tilted_rectangle_wall(structures, [-0.2, -1.0, 0, 0, 0, 0], 0.5, 20, 4.4, wall_height=1.75, slope=0)
            # self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] + x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (col_num-0.5-1)*stepping_stone_size[1] - stepping_stone_size[1] * 2 + y_random, 0, 0, 0, 180], 0.5, 20, x_wall_length, wall_height=1.65, slope=0)


        elif surface_source == 'torso_path_planning_test_env_5':

            self.goal_x = 4.0
            self.goal_y = 0.0

            stepping_stone_size = (0.4, 0.4)
            row_num = 11
            col_num = 6

            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row - row_num // 2)*stepping_stone_size[0] + 2,
                                         (col - col_num // 2)*stepping_stone_size[1] - 0.5,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)

            # # side wall
            # x_wall_length = row_num*stepping_stone_size[0]
            self.construct_tilted_rectangle_wall(structures, [-0.2, -1.0-0.9, 0, 0, 0, 0], 0.5, 20, 4.4, wall_height=1.25, slope=0)
            self.construct_tilted_rectangle_wall(structures, [-0.2, -1.0-0.9, 0, 0, 0, 0], 0.5, 20, 4.4, wall_height=1.75, slope=0)
            # self.construct_tilted_rectangle_wall(structures, [(0.5*row_num-row_num // 2)*stepping_stone_size[0] + x_wall_length/2.0 - stepping_stone_size[0] / 2 + x_random, (col_num-0.5-1)*stepping_stone_size[1] - stepping_stone_size[1] * 2 + y_random, 0, 0, 0, 180], 0.5, 20, x_wall_length, wall_height=1.65, slope=0)


        else:
            raw_input('Unknown surface soruce: %s.'%(surface_source))


        # define a transform
        ground_wall_transform = xyzrpy_to_SE3([0,0,0,0,0,0])
        # ground_wall_transform = xyzrpy_to_SE3([0,0,0,0,0,random.randint(0, 15) * 22.5])

        for struct in structures:
            struct.transform_data(ground_wall_transform, None)

            surface_trimesh_indices = np.array([[0,1,2],[0,2,3]])

            surface_trimesh_vertices = np.zeros((4,3),dtype=float)
            for i in range(len(surface_trimesh_vertices)):
                for j in range(3):
                    surface_trimesh_vertices[i,j] = struct.vertices[i][j]

            surface_trimesh = rave.RaveCreateKinBody(self.env,'')
            surface_trimesh.SetName('random_trimesh_' + str(struct.id))
            surface_trimesh.InitFromTrimesh(rave.TriMesh(surface_trimesh_vertices, surface_trimesh_indices),False)
            struct.kinbody = surface_trimesh
            self.env.AddKinBody(struct.kinbody)

            self.DrawSurface(struct, transparency=0.5, style='random_green_yellow_color')
            # self.DrawOrientation(struct.transform)


        for struct in structures:
            if(struct.type == 'ground'):
                temp_init_p = struct.projection_global_frame(np.array([[0.0],[0.0],[99.0]]),np.array([[0.0],[0.0],[-1.0]]))
                temp_goal_p = struct.projection_global_frame(np.array([[self.goal_x],[self.goal_y],[99.0]]),np.array([[0.0],[0.0],[-1.0]]))

                if(temp_init_p is not None and struct.inside_polygon(temp_init_p) and temp_init_p[2,0] >= self.init_z):
                    self.init_z = temp_init_p[2,0]

                if(temp_goal_p is not None and struct.inside_polygon(temp_goal_p) and temp_goal_p[2,0] >= self.goal_z):
                    self.goal_z = temp_goal_p[2,0]

        # store the mesh to stl files (if save_stl is true)
        if save_stl:
            # first clean all the files in the path to avoid confusion
            if os.path.exists(save_stl_path):
                shutil.rmtree(save_stl_path)
            os.mkdir(save_stl_path)

            for struct in structures:
                new_surface = mesh.Mesh(np.zeros(struct.trimesh_indices.shape[0]*2, dtype=mesh.Mesh.dtype))
                for i, face in enumerate(struct.trimesh_indices):
                    for j in range(3):
                        new_surface.vectors[i][j] = struct.trimesh_vertices[face[j],:]

                for i, face in enumerate(struct.trimesh_indices):
                    for j in range(3):
                        new_surface.vectors[struct.trimesh_indices.shape[0]+i][j] = struct.trimesh_vertices[face[2-j],:]

                new_surface.save(save_stl_path + struct.kinbody.GetName() + '.stl')

        self.structures = structures

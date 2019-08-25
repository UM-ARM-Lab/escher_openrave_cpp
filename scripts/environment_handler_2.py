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

from matplotlib import cm

class environment_handler:
    def __init__(self,env=None,structures=None,enable_viewer=True):
        if(env is None):
            rave.misc.InitOpenRAVELogging()
            self.env = rave.Environment()  # create openrave environment

            if enable_viewer:
                self.env.SetViewer('qtcoin')  # attach viewer (optional)
                # self.env.GetViewer().SetCamera(np.array([[1, 0, 0, 2.0],
                #                                          [0,-1, 0, 0],
                #                                          [0, 0,-1, 5.0],
                #                                          [0, 0, 0, 1]], dtype=float))
                self.env.GetViewer().SetCamera(np.array([[0, -1, 0, 1.0],
                                                         [-1, 0, 0, 0.0],
                                                         [0, 0, -1, 5.0],
                                                         [0, 0, 0, 1]], dtype=float))

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

    def DrawGridLevelSet(self,grid,transparancy=1.0,draw_grid=True,color_map_type='viridis'):

        color_map = cm.get_cmap(color_map_type, 100)
        grid_array = np.array(grid)
        min_value = np.min(grid_array[:,:,2])
        max_value = np.max(grid_array[:,:,2])
        min_max_diff = max_value - min_value

        for ix in range(len(grid)-1):
            for iy in range(len(grid[0])-1):
                trimesh_vertices = np.array([grid[ix][iy], grid[ix+1][iy], grid[ix+1][iy+1], grid[ix][iy+1]])
                trimesh_indices = np.array([[0, 1, 2], [0, 2, 3]])
                mean_value = (grid[ix][iy][2] + grid[ix+1][iy][2] + grid[ix+1][iy+1][2] + grid[ix][iy+1][2]) / 4.0
                color = color_map((mean_value-min_value) / min_max_diff)
                self.draw_handles.append(self.env.drawtrimesh(trimesh_vertices, trimesh_indices, colors=np.array(list(color[0:3])+[transparancy])))

                if draw_grid:
                    for i in range(np.size(trimesh_vertices,0)):
                        boundaries_point = np.zeros((2,3),dtype=float)
                        boundaries_point[0,:] = trimesh_vertices[i]
                        boundaries_point[1,:] = trimesh_vertices[(i+1) % np.size(trimesh_vertices,0)]
                        self.draw_handles.append(self.env.drawlinestrip(points=boundaries_point, linewidth=0.5, colors=np.zeros(3)))


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

    def DrawRegion(self, env, transform, radius):
        center = transform[0:3,3:4]
        x_vector = transform[0:3,0:1]
        y_vector = transform[0:3,1:2]

        region_boundary_points = np.zeros((37,3),dtype=float)

        for i in range(37):
            region_boundary_point = center + x_vector * radius * math.cos(i*10*math.pi/180.0) + y_vector * radius * math.sin(i*10*math.pi/180.0)
            region_boundary_points[i:i+1,:] = region_boundary_point.T

        self.draw_handles.append(self.env.drawlinestrip(points = region_boundary_points,linewidth = 5.0,colors = np.array((0,0,0))))

        # arrow_points = np.zeros((5,3),dtype=float)

        # arrow_points[0:1,:] = (center - radius*2.0/3.0 * x_vector).T
        # arrow_points[1:2,:] = (center + radius*2.0/3.0 * x_vector).T
        # arrow_points[2:3,:] = (center + radius/2.0 * y_vector).T
        # arrow_points[3:4,:] = (center + radius*2.0/3.0 * x_vector).T
        # arrow_points[4:5,:] = (center - radius/2.0 * y_vector).T

        # self.draw_handles.append(self.env.drawlinestrip(points = arrow_points,linewidth = 5.0,colors = np.array((0,0,0))))

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

        surface_trimesh = rave.RaveCreateKinBody(self.env,'')
        surface_trimesh.SetName('random_trimesh_' + str(len(structures)))
        surface_trimesh.InitFromTrimesh(rave.TriMesh(surface_trimesh_vertices, surface_trimesh_indices),False)

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

        self.env.AddKinBody(random_surface.kinbody)

        structures.append(random_surface)
        self.DrawSurface(random_surface, transparency=surface_transparancy, style='random_green_yellow_color')
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

            surface_vertices = [(0.2,0.2),(-0.2,0.2),(-0.2,-0.2),(0.2,-0.2)]

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

            self.goal_x = 2.0
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
                                                       [0,ground_max_y+0.1,1.2,90,0,0], surface_transparancy=0.5, surface_type='others')

            self.add_quadrilateral_surface(structures, [(wall_max_x,wall_min_y),
                                                        (wall_max_x,wall_max_y),
                                                        (wall_min_x,wall_max_y),
                                                        (wall_min_x,wall_min_y)],
                                                       [0,ground_min_y-0.1,1.2,-90,0,0], surface_transparancy=0.5, surface_type='others')

        elif(surface_source == 'flat_ground_env'):
            flat_ground_max_x = 4.0
            flat_ground_min_x = -1.0
            flat_ground_max_y = 1.0
            flat_ground_min_y = -1.0

            self.add_quadrilateral_surface(structures, [(flat_ground_max_x,flat_ground_min_y),
                                                        (flat_ground_max_x,flat_ground_max_y),
                                                        (flat_ground_min_x,flat_ground_max_y),
                                                        (flat_ground_min_x,flat_ground_min_y)], [0,0,0,0,0,0])

            self.goal_x = 1.0
            self.goal_y = 0.0

        elif(surface_source == 'rubble_field_env'):
            stepping_stone_size = (0.3,0.3)
            row_num = 8
            col_num = 10
            surface_projected_vertices = [(stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0),
                                          (stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,stepping_stone_size[1]/2.0),
                                          (-stepping_stone_size[0]/2.0,-stepping_stone_size[1]/2.0)]
            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num): # columns of stepping stones
                    surface_transform = [(row+0.5)*stepping_stone_size[0] + 0.2,
                                         (col+0.5)*stepping_stone_size[1] - 1.5,
                                         random.uniform(-0.05,0.05),
                                         random.uniform(-20,20),
                                         random.uniform(-20,20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)

            self.add_quadrilateral_surface(structures, [(0.2,-0.2,0),(0.2,0.2,0),(-0.2,0.2,0),(-0.2,-0.2,0)], [0,0,0,0,0,0])

            self.goal_x = random.random() * 1.0 + 1.0
            self.goal_y = random.random() * 2.5 - 1.25

        elif(surface_source == 'flat_ground_and_wall_env'):
            self.goal_x = 0.8
            self.goal_y = 0.0

            # add the ground
            ground_max_x = 4.5
            ground_min_x = -1.0
            ground_max_y = 0.55
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
                                                       [0,ground_max_y+0.1,1.2,90,0,0], surface_transparancy=0.5, surface_type='others')

        elif(surface_source == 'thin_flat_ground_env'):
            flat_ground_max_x = 4.0
            flat_ground_min_x = -1.0
            flat_ground_max_y = 0.25
            flat_ground_min_y = -0.25
            # flat_ground_max_y = 0.15
            # flat_ground_min_y = -0.15

            self.add_quadrilateral_surface(structures, [(flat_ground_max_x,flat_ground_min_y),
                                                        (flat_ground_max_x,flat_ground_max_y),
                                                        (flat_ground_min_x,flat_ground_max_y),
                                                        (flat_ground_min_x,flat_ground_min_y)], [0,0,0,0,0,0])

            self.goal_x = 1.0
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

        elif(surface_source == 'dynopt_test_env_6_hermes_full'):
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
            self.construct_tilted_rectangle_wall(structures, [stepping_stone_start_x + 0.5*row_num*stepping_stone_size[0]/2.0 - wall_length/2.0, -0.5, 0, 0, 0, 0], 0.5, 20, wall_length, wall_height=1.3, slope=0)
            self.construct_tilted_rectangle_wall(structures, [stepping_stone_start_x + 0.5*row_num*stepping_stone_size[0]/2.0 + wall_length/2.0, 0.5, 0, 0, 0, 180], 0.5, 20, wall_length, wall_height=1.3, slope=0)
            # self.construct_tilted_rectangle_wall(structures, [stepping_stone_start_x + 0.5*row_num*stepping_stone_size[0]/2.0 - wall_length/2.0, -0.45, 0, 0, 0, 0], 0.5, 20, wall_length, wall_height=1.3, slope=0)
            # self.construct_tilted_rectangle_wall(structures, [stepping_stone_start_x + 0.5*row_num*stepping_stone_size[0]/2.0 + wall_length/2.0, 0.45, 0, 0, 0, 180], 0.5, 20, wall_length, wall_height=1.3, slope=0)

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

        elif(surface_source == 'capture_test_env_3'): # narrow flat ground

            corridor_length = 2.2
            corridor_width = 0.7

            # initial platform
            self.add_quadrilateral_surface(structures, [(-0.2,corridor_width/2.0),
                                                        (-0.2,-corridor_width/2.0),
                                                        (corridor_length,-corridor_width/2.0),
                                                        (corridor_length,corridor_width/2.0)],
                                                        [0,0,0,0,0,0])

            # left wall
            # self.add_quadrilateral_surface(structures, [(-0.2,corridor_width/2.0),
            #                                             (-0.2,-corridor_width/2.0),
            #                                             (corridor_length,-corridor_width/2.0),
            #                                             (corridor_length,corridor_width/2.0)],
            #                                             [0,corridor_width/2.0+0.1,1.3,90,0,0],
            #                                             surface_type='others')

            # right wall
            # self.add_quadrilateral_surface(structures, [(-0.2,corridor_width/2.0),
            #                                             (-0.2,-corridor_width/2.0),
            #                                             (corridor_length,-corridor_width/2.0),
            #                                             (corridor_length,corridor_width/2.0)],
            #                                             [0,-corridor_width/2.0-0.1,1.3,-90,0,0],
            #                                             surface_type='others')

            self.goal_x = corridor_length - 0.4
            self.goal_y = 0

        elif(surface_source == 'capture_test_env_4'): # one wall rubble

            corridor_length = 2.0
            # corridor_width = 0.6
            # corridor_width = 1.2
            corridor_width = 1.8

            # random.seed(532) # 9(14,20,25), 3
            # random.seed(2478) # 10(15,21,26), 4
            random.seed(8) # 11(16,22,27), 5
            # random.seed(78945) # 12(17,23,28), 6
            # random.seed(29854745) # 13(18,24,29), 7

            # initial platform
            self.add_quadrilateral_surface(structures, [(-0.2,corridor_width/2.0),
                                                        (-0.2,-0.3),
                                                        (0.2,-0.3),
                                                        (0.2,corridor_width/2.0)],
                                                        [0,0,0,0,0,0])

            # final platform
            self.add_quadrilateral_surface(structures, [(corridor_length-0.4,corridor_width/2.0),
                                                        (corridor_length-0.4,-0.3),
                                                        (corridor_length,-0.3),
                                                        (corridor_length,corridor_width/2.0)],
                                                        [0,0,0,0,0,0])

            stepping_stone_size = (0.35,0.3)
            row_num = 4
            col_num = 2
            stepping_stone_start_x = 0.2
            stepping_stone_start_y = -0.3

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

            for row in range(row_num): # rows of stepping stones forward
                for col in range(col_num,col_num+2): # columns of stepping stones
                    surface_transform = [(row+0.5) * stepping_stone_size[0] + stepping_stone_start_x,
                                         (col+0.5) * stepping_stone_size[1] + stepping_stone_start_y,
                                         random.uniform(-0.05, 0.05),
                                         random.uniform(-20, 20),
                                         random.uniform(-20, 20),
                                         0]

                    self.add_quadrilateral_surface(structures, surface_projected_vertices, surface_transform)

            # left wall
            # self.add_quadrilateral_surface(structures, [(-0.2,corridor_width/2.0),
            #                                             (-0.2,-corridor_width/2.0),
            #                                             (corridor_length,-corridor_width/2.0),
            #                                             (corridor_length,corridor_width/2.0)],
            #                                             [0,corridor_width/2.0+0.2,1.2,90,0,0],
            #                                             surface_type='others')

            # self.add_quadrilateral_surface(structures, [(-0.2,corridor_width/2.0),
            #                                             (-0.2,-corridor_width/2.0),
            #                                             (corridor_length,-corridor_width/2.0),
            #                                             (corridor_length,corridor_width/2.0)],
            #                                             [0,corridor_width/2.0+0.35,1.2,90,0,0],
            #                                             surface_type='others')
            self.add_quadrilateral_surface(structures, [(-0.2,corridor_width/2.0),
                                                        (-0.2,-corridor_width/2.0),
                                                        (corridor_length,-corridor_width/2.0),
                                                        (corridor_length,corridor_width/2.0)],
                                                        [0,corridor_width/2.0+0.2,1.2,90,0,0],
                                                        surface_type='others')

            # right wall
            # self.add_quadrilateral_surface(structures, [(-0.2,corridor_width/2.0),
            #                                             (-0.2,-corridor_width/2.0),
            #                                             (corridor_length,-corridor_width/2.0),
            #                                             (corridor_length,corridor_width/2.0)],
            #                                             [0,-corridor_width/2.0-0.1,1.2,-90,0,0],
            #                                             surface_type='others')

            self.goal_x = corridor_length - 0.2
            self.goal_y = 0

        elif(surface_source == 'capture_test_env_5'): # oil platform
            ground_max_x = 3.2
            ground_min_x = -0.2
            ground_max_y = 1.5
            ground_min_y = -2.2

            obstacle_max_x = 2.1
            obstacle_min_x = 0.4
            obstacle_max_y = -0.2
            obstacle_min_y = -1.0
            # ground platforms
            self.add_quadrilateral_surface(structures, [(ground_min_x,ground_max_y),
                                                        (ground_min_x,ground_min_y),
                                                        (obstacle_min_x,ground_min_y),
                                                        (obstacle_min_x,ground_max_y)],
                                                        [0,0,0,0,0,0])

            self.add_quadrilateral_surface(structures, [(-(ground_max_x-ground_min_x)/2.0,(obstacle_min_y-ground_min_y)/2.0),
                                                        (-(ground_max_x-ground_min_x)/2.0,-(obstacle_min_y-ground_min_y)/2.0),
                                                        ((ground_max_x-ground_min_x)/2.0,-(obstacle_min_y-ground_min_y)/2.0),
                                                        ((ground_max_x-ground_min_x)/2.0,(obstacle_min_y-ground_min_y)/2.0)],
                                                        [(ground_min_x+ground_max_x)/2.0,(ground_min_y+obstacle_min_y)/2.0,0,0,0,0])

            self.add_quadrilateral_surface(structures, [(-(ground_max_x-ground_min_x)/2.0,(ground_max_y-obstacle_max_y)/2.0),
                                                        (-(ground_max_x-ground_min_x)/2.0,-(ground_max_y-obstacle_max_y)/2.0),
                                                        ((ground_max_x-ground_min_x)/2.0,-(ground_max_y-obstacle_max_y)/2.0),
                                                        ((ground_max_x-ground_min_x)/2.0,(ground_max_y-obstacle_max_y)/2.0)],
                                                        [(ground_min_x+ground_max_x)/2.0,(ground_max_y+obstacle_max_y)/2.0,0,0,0,0])

            self.add_quadrilateral_surface(structures, [(-(ground_max_x-obstacle_max_x)/2.0,(ground_max_y-ground_min_y)/2.0),
                                                        (-(ground_max_x-obstacle_max_x)/2.0,-(ground_max_y-ground_min_y)/2.0),
                                                        ((ground_max_x-obstacle_max_x)/2.0,-(ground_max_y-ground_min_y)/2.0),
                                                        ((ground_max_x-obstacle_max_x)/2.0,(ground_max_y-ground_min_y)/2.0)],
                                                        [(obstacle_max_x+ground_max_x)/2.0,(ground_max_y+ground_min_y)/2.0,0,0,0,0])

            # Obstacle
            body = rave.RaveCreateKinBody(self.env,'')
            body.SetName('obstacle')
            body.InitFromBoxes(np.array([[(obstacle_max_x+obstacle_min_x)/2.0,(obstacle_max_y+obstacle_min_y)/2.0,0.75,(obstacle_max_x-obstacle_min_x)/2.0 - 0.1,(obstacle_max_y-obstacle_min_y)/2.0 - 0.1,0.75]]),True)
            self.env.AddKinBody(body)


            # self.add_quadrilateral_surface(structures, [(-0.2,corridor_width/2.0),
            #                                             (-0.2,-corridor_width/2.0),
            #                                             (corridor_length,-corridor_width/2.0),
            #                                             (corridor_length,corridor_width/2.0)],
            #                                             [0,corridor_width/2.0+0.2,1.2,90,0,0],
            #                                             surface_type='others')

            # right wall
            # self.add_quadrilateral_surface(structures, [(-0.2,corridor_width/2.0),
            #                                             (-0.2,-corridor_width/2.0),
            #                                             (corridor_length,-corridor_width/2.0),
            #                                             (corridor_length,corridor_width/2.0)],
            #                                             [0,-corridor_width/2.0-0.1,1.2,-90,0,0],
            #                                             surface_type='others')

            self.goal_x = ground_max_x - 0.6
            self.goal_y = ground_max_y - 0.4

        else:
            raw_input('Unknown surface soruce: %s.'%(surface_source))





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

import math
import sys

class map_grid_dim:
    def __init__(self,min_x,max_x,min_y,max_y,resolution):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_theta = -150
        self.max_theta = 180
        self.resolution = resolution
        self.angle_resolution = 30

    def update_grid_boundary(self,structures):
        env_min_x = sys.maxint
        env_max_x = -sys.maxint
        env_min_y = sys.maxint
        env_max_y = -sys.maxint

        for struct in structures:
            for vertex in struct.vertices:
                env_min_x = min(vertex[0], env_min_x)
                env_max_x = max(vertex[0], env_max_x)
                env_min_y = min(vertex[1], env_min_y)
                env_max_y = max(vertex[1], env_max_y)

        self.min_x = math.floor(env_min_x/self.resolution)*self.resolution - self.resolution/2.0
        self.max_x = math.ceil(env_max_x/self.resolution)*self.resolution + self.resolution/2.0
        self.min_y = math.floor(env_min_y/self.resolution)*self.resolution - self.resolution/2.0
        self.max_y = math.ceil(env_max_y/self.resolution)*self.resolution + self.resolution/2.0
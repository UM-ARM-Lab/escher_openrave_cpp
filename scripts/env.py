import pickle
import numpy as np


def patch_depth_map(env_depth_map, resolution, vertices):
    """
    input:
    "env_depth_map" should be a 2d numpy array. It will be modified in place.
    
    "resolution" should be a float which represents the distance between adjacent pixels on the depth map.

    """




def main():
    file = open('../data/environments', 'r')
    environments = pickle.load(file)

    for environment in environments:
        for i in range(len(environment) // 15):
            for j in range(4):
                print(environment[15 * i + 3 * j], environment[15 * i + 3 * j + 1], environment[15 * i + 3 * j + 2])

if __name__ == '__main__':
    main()
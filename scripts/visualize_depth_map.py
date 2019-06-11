import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    for i in range(5):
        file = open('../data/test_depth_map/ground_depth_maps/' + str(i), 'r')
        data = pickle.load(file)
        plt.imshow(data, cmap='gray')
        plt.show()

        file = open('../data/test_depth_map/wall_depth_maps/' + str(i), 'r')
        data = pickle.load(file)
        plt.imshow(data, cmap='gray')
        plt.show()



if __name__ == "__main__":
    main()
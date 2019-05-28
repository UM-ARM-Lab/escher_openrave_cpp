import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    for i in range(10):
        file = open('../data/minimal/ground_depth_maps/' + str(i), 'r')
        data = pickle.load(file)
        plt.imshow(data, cmap='gray')
        plt.show()

        file = open('../data/minimal/wall_depth_maps/' + str(i), 'r')
        data = pickle.load(file)
        plt.imshow(data, cmap='gray')
        plt.show()



if __name__ == "__main__":
    main()
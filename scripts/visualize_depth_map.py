import pickle, os
import numpy as np
import matplotlib.pyplot as plt

def main():
    files = os.listdir('../data/test/wall_depth_maps')
    for file in files:
        with open('../data/test/wall_depth_maps/' + file, 'r') as depth_map:
            data = pickle.load(depth_map)
            plt.imshow(data[0], cmap='gray')
            plt.title(file)
            plt.show()



if __name__ == "__main__":
    main()
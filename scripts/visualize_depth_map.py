import pickle, os
import numpy as np
import matplotlib.pyplot as plt

def main():
    files = os.listdir('../data/ground_truth_p1p2/wall_depth_maps')
    for file in files:
        with open('../data/ground_truth_p1p2/wall_depth_maps/' + file, 'r') as depth_map:
            data = pickle.load(depth_map)
            plt.imshow(data, cmap='gray')
            plt.title(file)
            plt.show()



if __name__ == "__main__":
    main()
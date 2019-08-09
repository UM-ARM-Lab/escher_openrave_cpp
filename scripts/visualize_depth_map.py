import pickle, os
import numpy as np
import matplotlib.pyplot as plt

def main():
    files = os.listdir('../data/test/ground_depth_maps')
    for index, file in enumerate(files[-10:]):
        with open('../data/test/ground_depth_maps/' + file, 'r') as depth_map:
            data = pickle.load(depth_map)
            plt.figure(index)
            plt.imshow(data[0], cmap='gray')
            plt.title(file)
            plt.savefig(file + '_ground')
            # plt.show()



if __name__ == "__main__":
    main()

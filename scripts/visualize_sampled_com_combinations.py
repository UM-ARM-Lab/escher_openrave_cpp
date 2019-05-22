import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def main():
    for i in range(10):
        file = open('../data/dynopt_result/dataset/dynopt_total_data_' + str(i), 'r')
        all_com_combinations = pickle.load(file)[:, -13:-7]

        file = open('../data/COM/com_combinations_' + str(i), 'r')
        sampled_com_combinations = pickle.load(file)

        fig = plt.figure()

        subfig1 = fig.add_subplot(211, projection='3d')
        # xs = all_com_combinations[:, 0]
        # ys = all_com_combinations[:, 1]
        # zs = all_com_combinations[:, 2]
        xs = all_com_combinations[:, 3]
        ys = all_com_combinations[:, 4]
        zs = all_com_combinations[:, 5]
        subfig1.scatter(xs, ys, zs, marker='o')
        subfig1.set_xlabel('X label')
        subfig1.set_ylabel('Y label')
        subfig1.set_zlabel('Z label')

        subfig2 = fig.add_subplot(212, projection='3d')
        # xs = sampled_com_combinations[:, 0]
        # ys = sampled_com_combinations[:, 1]
        # zs = sampled_com_combinations[:, 2]
        xs = sampled_com_combinations[:, 3]
        ys = sampled_com_combinations[:, 4]
        zs = sampled_com_combinations[:, 5]
        subfig2.scatter(xs, ys, zs, marker='o')
        subfig2.set_xlabel('X label')
        subfig2.set_ylabel('Y label')
        subfig2.set_zlabel('Z label')

        plt.show()




if __name__ == "__main__":
    main()



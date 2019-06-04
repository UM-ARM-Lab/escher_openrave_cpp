import pickle, IPython, math, sys, random
import numpy as np
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt

SAMPLE_SIZE = 10000

random.seed(20190604)

def main():
    for transition_type in range(10):
        file = open('../data/dynopt_result/dataset/dynopt_total_data_' + str(transition_type), 'r')
        original_data = pickle.load(file)

        original_X = original_data[:, 1:-7]

        mean = np.mean(original_X, axis=0)
        std = np.std(original_X, axis=0)
        normalized_original_X = (original_X - mean) / std

        tree = BallTree(normalized_original_X)

        indices = random.sample(range(normalized_original_X.shape[0]), SAMPLE_SIZE)
        dist, _ = tree.query(normalized_original_X[indices], k=10)
        dist = np.mean(dist, axis=1)

        clipped = np.clip(dist, 0, 5)
        plt.figure(2 * transition_type)
        plt.hist(clipped, bins=np.arange(-0.1, 5.1, 0.1))
        plt.savefig('../data/test/regression_{}.png'.format(transition_type))


        file = open('../data/dynopt_result/dataset/dynopt_infeasible_total_data_' + str(transition_type), 'r')
        infeasible_original_data = pickle.load(file)
        
        infeasible_original_X = infeasible_original_data[:, 1:-1]
        all_original_X = np.concatenate((original_X, infeasible_original_X), axis=0)

        mean = np.mean(all_original_X, axis=0)
        std = np.std(all_original_X, axis=0)
        normalized_all_original_X = (all_original_X - mean) / std

        tree = BallTree(normalized_all_original_X)

        indices = random.sample(range(normalized_all_original_X.shape[0]), SAMPLE_SIZE)
        dist, _ = tree.query(normalized_all_original_X[indices], k=10)
        dist = np.mean(dist, axis=1)

        clipped = np.clip(dist, 0, 5)
        plt.figure(2 * transition_type + 1)
        plt.hist(clipped, bins=np.arange(-0.1, 5.1, 0.1))
        plt.savefig('../data/test/classification_{}.png'.format(transition_type))

        


       

        
        








  

if __name__ == "__main__":
    main()


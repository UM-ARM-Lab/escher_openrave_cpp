import pickle, IPython, math, random
import mkl
mkl.get_max_threads()
import faiss
import numpy as np
from sklearn.neighbors import BallTree
import timeit

SAMPLE_SIZE = 1000

random.seed(20190605)

def main():
    for transition_type in range(10):
        file = open('../data/dynopt_result/dataset/dynopt_total_data_' + str(transition_type), 'r')
        original_data = pickle.load(file)
        original_X = original_data[:, 1:-7]
        mean = np.mean(original_X, axis=0)
        std = np.std(original_X, axis=0)
        normalized_original_X = (original_X - mean) / std
        indices = random.sample(range(normalized_original_X.shape[0]), SAMPLE_SIZE)

        start = timeit.default_timer()
        tree = BallTree(normalized_original_X)
        dist, idx = tree.query(normalized_original_X[indices], k=5)
        dist = np.mean(dist, axis=1)
        print('sklearn', timeit.default_timer() - start)
        print(dist[:10])

        start = timeit.default_timer()
        normalized_original_X = np.float32(normalized_original_X)
        index = faiss.IndexFlatL2(normalized_original_X.shape[1])
        index.add(normalized_original_X)
        dist, idx = index.search(normalized_original_X[indices], 5)
        dist = dist.clip(min=0)
        dist = np.sqrt(dist)
        dist = np.mean(dist, axis=1)
        print('faiss', timeit.default_timer() - start)
        print(dist[:10])


if __name__ == "__main__":
    main()

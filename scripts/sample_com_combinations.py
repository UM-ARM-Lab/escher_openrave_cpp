import pickle
import numpy as np
from sklearn.neighbors import KDTree

RESOLUTION = 10.0


def main():
    for i in range(10):
        print('sample CoM combinations for type ' + str(i))
        file_handle = open('../data/dynopt_result/dataset/dynopt_total_data_' + str(i), 'r')
        data = pickle.load(file_handle)
        com_combinations = data[:, -13:-7]
        print('number of all CoM combinations: {}'.format(com_combinations.shape[0]))
        
        # normalize
        mean = np.mean(com_combinations, axis=0)
        std = np.std(com_combinations, axis=0)
        normalized_com_combinations = (com_combinations - mean) / std

        tree = KDTree(normalized_com_combinations, leaf_size=30, metric='euclidean')

        variables_min = {}
        variables_diff = {}
        for j in range(6):
            variables_min[j] = np.min(normalized_com_combinations[:, j])
            variables_diff[j] = (np.max(normalized_com_combinations[:, j]) - np.min(normalized_com_combinations[:, j])) / RESOLUTION

        query_com_combinations = np.zeros((RESOLUTION ** 6, 6), dtype=float)

        total_idx = 0
        for idx0 in range(RESOLUTION):
            for idx1 in range(RESOLUTION):
                for idx2 in range(RESOLUTION):
                    for idx3 in range(RESOLUTION):
                        for idx4 in range(RESOLUTION):
                            for idx5 in range(RESOLUTION):
                                query_com_combinations[total_idx, 0] = variables_min[0] + idx0 * variables_diff[0]
                                query_com_combinations[total_idx, 1] = variables_min[1] + idx1 * variables_diff[1]
                                query_com_combinations[total_idx, 2] = variables_min[2] + idx2 * variables_diff[2]
                                query_com_combinations[total_idx, 3] = variables_min[3] + idx3 * variables_diff[3]
                                query_com_combinations[total_idx, 4] = variables_min[4] + idx4 * variables_diff[4]
                                query_com_combinations[total_idx, 5] = variables_min[5] + idx5 * variables_diff[5]
                                total_idx += 1
                                
        assert(total_idx == RESOLUTION ** 6)
        indices = tree.query(query_com_combinations, k=1, return_distance=False).reshape(-1,)
        unique_indices = np.unique(indices, axis=0)
        print('number of sampled CoM combinations: {}'.format(unique_indices.shape[0]))

        sampled_com_combinations = com_combinations[unique_indices]
                                
        file_handle = open('../data/CoM/com_combinations_' + str(i), 'w')
        pickle.dump(sampled_com_combinations, file_handle)

if __name__ == "__main__":
    main()


import pickle, random
import numpy as np

random.seed(20190528)
SAMPLE_SIZE = 3000

def main():
    for i in range(10):
        print('sample CoM combinations for type ' + str(i))
        file_handle = open('../data/dynopt_result/dataset/dynopt_total_data_' + str(i), 'r')
        data = pickle.load(file_handle)
        com_combinations = data[:, -13:-7]
        indices = random.sample(range(data.shape[0]), SAMPLE_SIZE)
        sampled_com_combinations = com_combinations[indices]
        file_handle = open('../data/random_CoM/com_combinations_' + str(i), 'w')
        pickle.dump(sampled_com_combinations, file_handle)

if __name__ == "__main__":
    main()


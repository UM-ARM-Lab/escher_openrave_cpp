import pickle
import random

random.seed(20190515)
SAMPLE_SIZE = 1000

def main():
    for i in range(10):
        file_handle = open('../data/dynopt_result/dataset/dynopt_total_data_' + str(i), 'r')
        data = pickle.load(file_handle)
        com_combinations = data[:, -13:-7]
        com_set = random.sample(com_combinations, SAMPLE_SIZE)
        file_handle = open('../data/COM/com_combinations_' + str(i), 'w')
        pickle.dump(com_set, file_handle)

if __name__ == "__main__":
    main()


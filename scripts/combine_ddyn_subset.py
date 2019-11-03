import pickle, os, IPython
import numpy as np
MODEL_INDEX = 3

def main():
    ground_truth_dict = {}
    for environment_type in range(0, 22, 3):
        with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/data_' + str(environment_type) + '_model_' + str(MODEL_INDEX), 'r') as file:
            data = pickle.load(file)
            ground_truth_dict.update(data)

    partition = {'training': [],
                 'validation': [],
                 'test': []}
    for environment_type in range(0, 22, 3):
        with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/partition_' + str(environment_type) + '_model_' + str(MODEL_INDEX), 'r') as file:
            data = pickle.load(file)
            partition['training'] += data['training']
            partition['validation'] += data['validation']
            partition['test'] += data['test']

    assert(len(ground_truth_dict.keys()) == len(partition['training']) + len(partition['validation']) + len(partition['test']))
    # IPython.embed()
    with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/data_model_' + str(MODEL_INDEX), 'w') as file:
        pickle.dump(ground_truth_dict, file)

    with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/partition_model_' + str(MODEL_INDEX), 'w') as file:
        pickle.dump(partition, file)


if __name__ == '__main__':
    main()
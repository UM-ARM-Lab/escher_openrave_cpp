import pickle, os, IPython
import numpy as np


def main():
    p2_ddyn = {}
    for environment_type in range(3):
        with open('../data/ground_truth_p1p2/p2_ddyn_' + str(environment_type), 'r') as file:
            data = pickle.load(file)
            p2_ddyn.update(data)

    partition = {'training': [],
                 'validation': [],
                 'test': []}
    for environment_type in range(3):
        with open('../data/ground_truth_p1p2/partition_' + str(environment_type), 'r') as file:
            data = pickle.load(file)
            partition['training'] += data['training']
            partition['validation'] += data['validation']
            partition['test'] += data['test']

    assert(len(p2_ddyn.keys()) == len(partition['training']) + len(partition['validation']) + len(partition['test']))
    # IPython.embed()
    with open('../data/ground_truth_p1p2/p2_ddyn_no_wall_subset', 'w') as file:
        pickle.dump(p2_ddyn, file)

    with open('../data/ground_truth_p1p2/partition_no_wall_subset', 'w') as file:
        pickle.dump(partition, file)


if __name__ == '__main__':
    main()
import pickle, os, IPython
import numpy as np


def main():
    p2_ddyns = [file for file in os.listdir('../data/ground_truth_p1p2') if file.startswith('p2_ddyn_')]
    print('Combine the following files into p2_ddyn')
    print(p2_ddyns)
    p2_ddyn = {}
    for filename in p2_ddyns:
        with open('../data/ground_truth_p1p2/' + filename, 'r') as file:
            data = pickle.load(file)
            p2_ddyn.update(data)

    partitions = [file for file in os.listdir('../data/ground_truth_p1p2') if file.startswith('partition_')]
    print('Combine the following files into partition')
    print(partitions)
    partition = {'training': [],
                 'validation': [],
                 'test': []}
    for filename in partitions:
        with open('../data/ground_truth_p1p2/' + filename, 'r') as file:
            data = pickle.load(file)
            partition['training'] += data['training']
            partition['validation'] += data['validation']
            partition['test'] += data['test']

    assert(len(p2_ddyn.keys()) == len(partition['training']) + len(partition['validation']) + len(partition['test']))
    # IPython.embed()
    with open('../data/ground_truth_p1p2/p2_ddyn', 'w') as file:
        pickle.dump(p2_ddyn, file)

    with open('../data/ground_truth_p1p2/partition', 'w') as file:
        pickle.dump(partition, file)


if __name__ == '__main__':
    main()
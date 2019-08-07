import pickle, torch, os, IPython, math
import torch.nn as nn
import numpy as np
import pprint
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from torch.utils import data

from model_B_0 import Model
from special_dataset import Dataset
from train import loss_across_epoch

model_version = 'model_B_0_Adam_0001_Weighted_Loss'

def restore_checkpoint(model, checkpoint_dir):
    files = [file for file in os.listdir(checkpoint_dir)
        if file.startswith('epoch=') and file.endswith('.checkpoint.pth.tar')]

    if not files:
        print('No saved model parameters found')
        exit(1)

    print('Which epoch to load from in range [0, {}]?'.format(len(files)-1))
    requested_epoch = int(input())
    if requested_epoch not in range(len(files)):
        print('Invalid epoch number')
        exit(1)

    filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(requested_epoch))
    checkpoint = torch.load(filename)

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully load the model from epoch {}'.format(requested_epoch))
    except:
        print('Fail to load the model from epoch {}'.format(requested_epoch))
        exit(1)

    return model


def main():
    with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/p2_ddyn', 'r') as file:
        p2_ddyn = pickle.load(file)

    with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/partition', 'r') as file:
        partition = pickle.load(file)
    print('number of test examples: {}'.format(len(partition['test'])))
    test_dataset = Dataset(p2_ddyn, partition['test'])
    test_generator = data.DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=4)

    all_ddyns = []
    all_predicted_ddyns = []

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print('device: {}'.format(device))
    model = Model()
    model = restore_checkpoint(model, model_version + '_checkpoint/').to(device)

    model.eval()
    with torch.set_grad_enabled(False):
        for index, (ground_depth_maps, wall_depth_maps, p2s, ddyns, example_ids) in enumerate(test_generator):
            print('batch ' + str(index))
            ground_depth_maps, wall_depth_maps, p2s, ddyns = ground_depth_maps.to(device), wall_depth_maps.to(device), p2s.to(device), ddyns.to(device)
            predicted_ddyns = model(ground_depth_maps, wall_depth_maps, p2s).squeeze()
            all_ddyns += ddyns.tolist()
            all_predicted_ddyns += predicted_ddyns.tolist()

    covariance_matrix = np.cov(all_ddyns, all_predicted_ddyns)
    coefficient = covariance_matrix[0][1] / math.sqrt(covariance_matrix[0][0]) / math.sqrt(covariance_matrix[1][1])
    print('coefficient of all: ' + str(coefficient))

    mask = np.array(all_ddyns) < 200
    small_ddyns = np.array(all_ddyns)[np.argwhere(mask == True).reshape(-1,)]
    small_predicted_ddyns = np.array(all_predicted_ddyns)[np.argwhere(mask == True).reshape(-1,)]
    small_covariance_matrix = np.cov(small_ddyns.tolist(), small_predicted_ddyns.tolist())
    small_coefficient = small_covariance_matrix[0][1] / math.sqrt(small_covariance_matrix[0][0]) / math.sqrt(small_covariance_matrix[1][1])
    print('coefficient of small: ' + str(small_coefficient))
    # covariance matrix:
    # [[119942.76214571  89306.90088912]
    #  [ 89306.90088912  93197.306531  ]]
    # correlation: 0.8446879822072368
    #
    # for small data (real value < 200)
    # covariance matrix:
    # [[1603.5883441   1418.07456518]
    #  [1418.07456518  8677.19421232]]
    # correlation: 0.3801569749444143


    IPython.embed()
            
            
    


if __name__ == '__main__':
    main()

    

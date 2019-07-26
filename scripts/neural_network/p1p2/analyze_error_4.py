import pickle, torch, os, IPython
import torch.nn as nn
import numpy as np
import pprint
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from torch.utils import data

from model_v3 import Model
from special_dataset import Dataset
from train import loss_across_epoch

model_version = 'model_v3_00005'

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
        model.load_state_dict(checkpoint['state_dict'])
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

    all_examples = []
    negative_loss_10_examples = []
    negative_loss_20_examples = []
    negative_loss_30_examples = []
    negative_loss_40_examples = []
    negative_loss_50_examples = []

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print('device: {}'.format(device))
    model = Model()
    model = restore_checkpoint(model, model_version + '_checkpoint/').to(device)

    model.eval()
    with torch.set_grad_enabled(False):
        for ground_depth_maps, wall_depth_maps, p2s, ddyns, example_ids in test_generator:
            print('1')
            ground_depth_maps, wall_depth_maps, p2s, ddyns = ground_depth_maps.to(device), wall_depth_maps.to(device), p2s.to(device), ddyns.to(device)
            predicted_ddyns = model(ground_depth_maps, wall_depth_maps, p2s).squeeze()
            loss = predicted_ddyns - ddyns
            all_examples += ddyns.tolist()
            negative_loss_10_indices = loss < -10
            negative_loss_10_examples += ddyns[np.argwhere(negative_loss_10_indices == True).reshape(-1,)].tolist()
            negative_loss_20_indices = loss < -20
            negative_loss_20_examples += ddyns[np.argwhere(negative_loss_20_indices == True).reshape(-1,)].tolist()
            negative_loss_30_indices = loss < -30
            negative_loss_30_examples += ddyns[np.argwhere(negative_loss_30_indices == True).reshape(-1,)].tolist()
            negative_loss_40_indices = loss < -40
            negative_loss_40_examples += ddyns[np.argwhere(negative_loss_40_indices == True).reshape(-1,)].tolist()
            negative_loss_50_indices = loss < -50
            negative_loss_50_examples += ddyns[np.argwhere(negative_loss_50_indices == True).reshape(-1,)].tolist()
            
    negative_10_clipped = np.clip(np.array(negative_loss_10_examples), 0, 2000)
    negative_10_hist, _ = np.histogram(negative_10_clipped, bins=np.arange(0, 2010, 10))
    negative_20_clipped = np.clip(np.array(negative_loss_20_examples), 0, 2000)
    negative_20_hist, _ = np.histogram(negative_20_clipped, bins=np.arange(0, 2010, 10))
    negative_30_clipped = np.clip(np.array(negative_loss_30_examples), 0, 2000)
    negative_30_hist, _ = np.histogram(negative_30_clipped, bins=np.arange(0, 2010, 10))
    negative_40_clipped = np.clip(np.array(negative_loss_40_examples), 0, 2000)
    negative_40_hist, _ = np.histogram(negative_40_clipped, bins=np.arange(0, 2010, 10))
    negative_50_clipped = np.clip(np.array(negative_loss_50_examples), 0, 2000)
    negative_50_hist, _ = np.histogram(negative_50_clipped, bins=np.arange(0, 2010, 10))

    all_examples_clipped = np.clip(np.array(all_examples), 0, 2000)
    all_examples_hist, _ = np.histogram(all_examples_clipped, bins=np.arange(0, 2010, 10))
    
    plt.figure()
    plt.plot(np.arange(0, 2000, 10), negative_10_hist, '-', label='negative 10', color='lightskyblue')
    plt.plot(np.arange(0, 2000, 10), negative_20_hist, '-', label='negative 20', color='cornflowerblue')
    plt.plot(np.arange(0, 2000, 10), negative_30_hist, '-', label='negative 30', color='royalblue')
    plt.plot(np.arange(0, 2000, 10), negative_40_hist, '-', label='negative 40', color='blue')
    plt.plot(np.arange(0, 2000, 10), negative_50_hist, '-', label='negative 50', color='darkblue')

    plt.plot(np.arange(0, 2000, 10), all_examples_hist, '-', label='all', color='green')
    
    plt.title('distribution of negative error')
    plt.xlabel('real value')
    plt.ylabel('number of examples')
    plt.legend()
    plt.savefig('negative.png')
    


if __name__ == '__main__':
    main()

    

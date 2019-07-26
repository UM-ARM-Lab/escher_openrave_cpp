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

model_version = 'model_v3_00001_Adam_Weighted_Loss'

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
    test_dataset = Dataset(p2_ddyn, partition['test'][:10])
    test_generator = data.DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=4)

    all_examples = []
    positive_loss_10_examples = []
    positive_loss_20_examples = []
    positive_loss_30_examples = []
    positive_loss_40_examples = []
    positive_loss_50_examples = []

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
            positive_loss_10_indices = loss > 10
            positive_loss_10_examples += ddyns[np.argwhere(positive_loss_10_indices == True).reshape(-1,)].tolist()
            positive_loss_20_indices = loss > 20
            positive_loss_20_examples += ddyns[np.argwhere(positive_loss_20_indices == True).reshape(-1,)].tolist()
            positive_loss_30_indices = loss > 30
            positive_loss_30_examples += ddyns[np.argwhere(positive_loss_30_indices == True).reshape(-1,)].tolist()
            positive_loss_40_indices = loss > 40
            positive_loss_40_examples += ddyns[np.argwhere(positive_loss_40_indices == True).reshape(-1,)].tolist()
            positive_loss_50_indices = loss > 50
            positive_loss_50_examples += ddyns[np.argwhere(positive_loss_50_indices == True).reshape(-1,)].tolist()
            
    positive_10_clipped = np.clip(np.array(positive_loss_10_examples), 0, 2000)
    positive_10_hist, _ = np.histogram(positive_10_clipped, bins=np.arange(0, 2010, 10))
    positive_20_clipped = np.clip(np.array(positive_loss_20_examples), 0, 2000)
    positive_20_hist, _ = np.histogram(positive_20_clipped, bins=np.arange(0, 2010, 10))
    positive_30_clipped = np.clip(np.array(positive_loss_30_examples), 0, 2000)
    positive_30_hist, _ = np.histogram(positive_30_clipped, bins=np.arange(0, 2010, 10))
    positive_40_clipped = np.clip(np.array(positive_loss_40_examples), 0, 2000)
    positive_40_hist, _ = np.histogram(positive_40_clipped, bins=np.arange(0, 2010, 10))
    positive_50_clipped = np.clip(np.array(positive_loss_50_examples), 0, 2000)
    positive_50_hist, _ = np.histogram(positive_50_clipped, bins=np.arange(0, 2010, 10))

    all_examples_clipped = np.clip(np.array(all_examples), 0, 2000)
    all_examples_hist, _ = np.histogram(all_examples_clipped, bins=np.arange(0, 2010, 10))
    
    plt.figure()
    plt.plot(np.arange(0, 2000, 10), positive_10_hist, '-', label='positive 10', color='pink')
    plt.plot(np.arange(0, 2000, 10), positive_20_hist, '-', label='positive 20', color='hotpink')
    plt.plot(np.arange(0, 2000, 10), positive_30_hist, '-', label='positive 30', color='deeppink')
    plt.plot(np.arange(0, 2000, 10), positive_40_hist, '-', label='positive 40', color='red')
    plt.plot(np.arange(0, 2000, 10), positive_50_hist, '-', label='positive 50', color='darkred')

    plt.plot(np.arange(0, 2000, 10), all_examples_hist, '-', label='all', color='green')
    
    plt.title('distribution of positive error')
    plt.xlabel('real value')
    plt.ylabel('number of examples')
    plt.legend()
    plt.savefig('positive_weighted_loss.png')
    


if __name__ == '__main__':
    main()

    

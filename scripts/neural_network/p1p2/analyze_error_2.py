import pickle, torch, os, IPython
import torch.nn as nn
import numpy as np
import pprint
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

    positive_loss_examples = []
    negative_loss_examples = []

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
            positive_loss_indices = loss > 50
            positive_loss_examples += ddyns[np.argwhere(positive_loss_indices == True).reshape(-1,)].tolist()
            negative_loss_indices = loss < -50
            negative_loss_examples += ddyns[np.argwhere(negative_loss_indices == True).reshape(-1,)].tolist()

    positive_clipped = np.clip(np.array(positive_loss_examples), 0, 2000)
    positive_hist, _ = np.histogram(positive_clipped, bins=np.arange(0, 2010, 10))
    negative_clipped = np.clip(np.array(negative_loss_examples), 0, 2000)
    negative_hist, _ = np.histogram(negative_clipped, bins=np.arange(0, 2010, 10))
    
    plt.figure()
    plt.plot(range(200), positive_hist, '-o', label='positive', color='red')
    plt.plot(range(200), negative_hist, '-o', label='negative', color='blue')
    plt.title('distribution of large error')
    plt.xlabel('dynamic cost')
    plt.ylabel('number of examples')
    plt.legend()
    plt.savefig('error.png')
    


if __name__ == '__main__':
    main()

    
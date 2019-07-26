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

    all_examples = []
    positive_loss_50_examples = []
    positive_loss_100_examples = []
    positive_loss_200_examples = []
    negative_loss_50_examples = []
    negative_loss_100_examples = []
    negative_loss_200_examples = []

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
            positive_loss_50_indices = loss > 50
            positive_loss_50_examples += ddyns[np.argwhere(positive_loss_50_indices == True).reshape(-1,)].tolist()
            positive_loss_100_indices = loss > 100
            positive_loss_100_examples += ddyns[np.argwhere(positive_loss_100_indices == True).reshape(-1,)].tolist()
            positive_loss_200_indices = loss > 200
            positive_loss_200_examples += ddyns[np.argwhere(positive_loss_200_indices == True).reshape(-1,)].tolist()
            negative_loss_50_indices = loss < -50
            negative_loss_50_examples += ddyns[np.argwhere(negative_loss_50_indices == True).reshape(-1,)].tolist()
            negative_loss_100_indices = loss < -100
            negative_loss_100_examples += ddyns[np.argwhere(negative_loss_100_indices == True).reshape(-1,)].tolist()
            negative_loss_200_indices = loss < -200
            negative_loss_200_examples += ddyns[np.argwhere(negative_loss_200_indices == True).reshape(-1,)].tolist()

    positive_50_clipped = np.clip(np.array(positive_loss_50_examples), 0, 2000)
    positive_50_hist, _ = np.histogram(positive_50_clipped, bins=np.arange(0, 2010, 10))
    positive_100_clipped = np.clip(np.array(positive_loss_100_examples), 0, 2000)
    positive_100_hist, _ = np.histogram(positive_100_clipped, bins=np.arange(0, 2010, 10))
    positive_200_clipped = np.clip(np.array(positive_loss_200_examples), 0, 2000)
    positive_200_hist, _ = np.histogram(positive_200_clipped, bins=np.arange(0, 2010, 10))
    
    negative_50_clipped = np.clip(np.array(negative_loss_50_examples), 0, 2000)
    negative_50_hist, _ = np.histogram(negative_50_clipped, bins=np.arange(0, 2010, 10))
    negative_100_clipped = np.clip(np.array(negative_loss_100_examples), 0, 2000)
    negative_100_hist, _ = np.histogram(negative_100_clipped, bins=np.arange(0, 2010, 10))
    negative_200_clipped = np.clip(np.array(negative_loss_200_examples), 0, 2000)
    negative_200_hist, _ = np.histogram(negative_200_clipped, bins=np.arange(0, 2010, 10))

    all_examples_clipped = np.clip(np.array(all_examples), 0, 2000)
    all_examples_hist, _ = np.histogram(all_examples_clipped, bins=np.arange(0, 2010, 10))
    
    plt.figure()
    plt.plot(range(200), positive_50_hist, '-', label='positive 50', color='pink')
    plt.plot(range(200), positive_100_hist, '-', label='positive 100', color='deeppink')
    plt.plot(range(200), positive_200_hist, '-', label='positive 200', color='red')
    
    plt.plot(range(200), negative_50_hist, '-', label='negative 50', color='lightskyblue')
    plt.plot(range(200), negative_100_hist, '-', label='negative 100', color='blue')
    plt.plot(range(200), negative_200_hist, '-', label='negative 200', color='darkblue')

    plt.plot(range(200), all_examples_hist, '-', label='all', color='green')
    
    plt.title('distribution of large error')
    plt.xlabel('dynamic cost')
    plt.ylabel('number of examples')
    plt.legend()
    plt.savefig('error.png')
    


if __name__ == '__main__':
    main()

    
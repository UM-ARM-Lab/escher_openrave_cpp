import pickle, torch, os, IPython
import torch.nn as nn
import numpy as np
import pprint

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
    filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(requested_epoch))

    if not os.path.exists(filename):
        print('Invalid epoch number')
        exit(1)

    checkpoint = torch.load(filename)

    try:
        model.load_state_dict(checkpoint['state_dict'])
        print('Successfully load the model from epoch {}'.format(requested_epoch))
    except:
        print('Fail to load the model from epoch {}'.format(requested_epoch))
        exit(1)

    return model


def main():
    with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/p2_ddyn_no_wall_subset', 'r') as file:
        p2_ddyn = pickle.load(file)

    with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/partition_no_wall_subset', 'r') as file:
        partition = pickle.load(file)
    print('number of test examples: {}'.format(len(partition['test'])))
    test_dataset = Dataset(p2_ddyn, partition['test'])
    test_generator = data.DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=4)

    large_loss_examples = []
    small_loss_examples = []

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print('device: {}'.format(device))
    model = Model()
    model = restore_checkpoint(model, model_version + '_checkpoint/').to(device)
    criterion = nn.L1Loss()

    criterion_non_reduced = nn.L1Loss(reduction='none')

    model.eval()
    with torch.set_grad_enabled(False):
        loss_list = []
        length_list = []
        for ground_depth_maps, wall_depth_maps, p2s, ddyns, example_ids in test_generator:
            print('1')
            ground_depth_maps, wall_depth_maps, p2s, ddyns = ground_depth_maps.to(device), wall_depth_maps.to(device), p2s.to(device), ddyns.to(device)
            predicted_ddyns = model(ground_depth_maps, wall_depth_maps, p2s).squeeze()
            loss = criterion(predicted_ddyns, ddyns)

            loss_non_reduced = criterion_non_reduced(predicted_ddyns, ddyns).cpu().data.numpy()
            large_loss_indices = loss_non_reduced > 200
            large_loss_examples += np.array(example_ids)[np.argwhere(large_loss_indices == True).reshape(-1,)].tolist()
            # print('ground truth')
            # print(ddyns[np.argwhere(large_loss_indices == True).reshape(-1,)])
            # print('predicted')
            # print(predicted_ddyns[np.argwhere(large_loss_indices == True).reshape(-1,)])
            small_loss_indices = loss_non_reduced < 50
            small_loss_examples += np.array(example_ids)[np.argwhere(small_loss_indices == True).reshape(-1,)].tolist()

            loss_list.append(loss)
            length_list.append(predicted_ddyns.shape[0])
        epoch_loss = loss_across_epoch(loss_list, length_list)
        print('test loss: {:4.2f}'.format(epoch_loss))

    total_count = {}
    for i in range(12):
        total_count[i] = 0
    for example in partition['test']:
        for i in range(12):
            if example.startswith(str(i) + '_'):
                total_count[i] += 1
                break
    print('total')
    pprint.pprint(total_count)

    large_error_examples_count = {}
    for i in range(12):
        large_error_examples_count[i] = 0

    for example in large_loss_examples:
        for i in range(12):
            if example.startswith(str(i) + '_'):
                large_error_examples_count[i] += 1
                break
    for i in range(3):
        large_error_examples_count[i] = large_error_examples_count[i] * 100.0 / total_count[i]
    print('large error')
    pprint.pprint(large_error_examples_count)

    small_error_examples_count = {}
    for i in range(12):
        small_error_examples_count[i] = 0

    for example in small_loss_examples:
        for i in range(12):
            if example.startswith(str(i) + '_'):
                small_error_examples_count[i] += 1
                break
    for i in range(3):
        small_error_examples_count[i] = small_error_examples_count[i] * 100.0 / total_count[i]
    print('small error')
    pprint.pprint(small_error_examples_count)
    


if __name__ == '__main__':
    main()

    
import pickle, torch, os
import torch.nn as nn

from torch.utils import data

from model_0 import Model
from dataset import Dataset
from train import loss_across_epoch


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
    with open('../../../data/ground_truth_p1p2/p2_ddyn', 'r') as file:
        p2_ddyn = pickle.load(file)

    with open('../../../data/ground_truth_p1p2/partition', 'r') as file:
        partition = pickle.load(file)
    print('number of test examples: {}'.format(len(partition['test'])))
    test_dataset = Dataset(p2_ddyn, partition['test'])
    test_generator = data.DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    model = Model()
    model = restore_checkpoint(model, 'checkpoint/').to(device)
    criterion = nn.L1Loss()
    model.eval()
    with torch.set_grad_enabled(False):
        loss_list = []
        length_list = []
        for ground_depth_maps, wall_depth_maps, p2s, ddyns in test_generator:
            ground_depth_maps, wall_depth_maps, p2s, ddyns = ground_depth_maps.to(device), wall_depth_maps.to(device), p2s.to(device), ddyns.to(device)
            predicted_ddyns = model(ground_depth_maps, wall_depth_maps, p2s)
            loss = criterion(predicted_ddyns, ddyns)
            loss_list.append(loss)
            length_list.append(predicted_ddyns.shape[0])
        epoch_loss = loss_across_epoch(loss_list, length_list)
        print('test loss: {:4.2f}'.format(epoch_loss))


if __name__ == '__main__':
    main()

    
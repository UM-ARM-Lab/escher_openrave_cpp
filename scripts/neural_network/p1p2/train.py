import os, torch, pickle

from torch import nn, optim
from torch.utils import data

from model_0 import Model
from dataset import Dataset


def save_checkpoint(model, epoch, checkpoint_dir):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }
    filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(epoch))
    torch.save(state, filename)


def main():
    with open('../../../data/ground_truth_p1p2/p2_ddyn', 'r') as file:
        p2_ddyn = pickle.load(file)

    torch.manual_seed(20190615)
    partition = {'training': ['9_0_00-5_-1-10', '9_0_00-5_-1-11', '9_0_00-5_-1-12', '9_0_00-5_-100', '9_0_00-5_-101',
                              '9_0_00-5_-102', '9_0_00-5_0-1-1', '9_0_00-5_00-2', '9_0_00-5_00-1', '9_0_00-5_000'],
                 'validation': ['9_0_00-1_010', '9_0_00-1_011', '9_0_000_-1-1-1', '9_0_000_-1-10', '9_0_000_-1-11',
                                '9_0_000_-100', '9_0_000_-11-1', '9_0_000_-110', '9_0_000_-111', '9_0_000_0-1-1']}
    training_dataset = Dataset(p2_ddyn, partition['training'])
    training_generator = data.DataLoader(training_dataset, batch_size=64, shuffle=True, num_workers=4)
    validation_dataset = Dataset(p2_ddyn, partition['validation'])
    validation_generator = data.DataLoader(validation_dataset, batch_size=len(partition['validation']), shuffle=True, num_workers=4)

    learning_rate = 0.001
    num_epoch = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    model = Model().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        print('epoch: {}'.format(epoch))
        # training
        model.train()
        for ground_depth_maps, wall_depth_maps, p2s, ddyns in training_generator:
            ground_depth_maps, wall_depth_maps, p2s, ddyns = ground_depth_maps.to(device), wall_depth_maps.to(device), p2s.to(device), ddyns.to(device)
            optimizer.zero_grad()
            predicted_ddyns = model(ground_depth_maps, wall_depth_maps, p2s)
            loss = criterion(predicted_ddyns, ddyns)
            print('training loss: {:4.2f}'.format(loss))
            loss.backward()
            optimizer.step()
            save_checkpoint(model, epoch, 'checkpoint/')

        # validation
        model.eval()
        with torch.set_grad_enabled(False):
            for ground_depth_maps, wall_depth_maps, p2s, ddyns in validation_generator:
                ground_depth_maps, wall_depth_maps, p2s, ddyns = ground_depth_maps.to(device), wall_depth_maps.to(device), p2s.to(device), ddyns.to(device)
                predicted_ddyns = model(ground_depth_maps, wall_depth_maps, p2s)
                loss = criterion(predicted_ddyns, ddyns)
                print('validation loss: {:4.2f}'.format(loss))





if __name__ == "__main__":
    main()

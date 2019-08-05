import os, torch, pickle, shutil, IPython
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_B_0 import Model
from dataset import Dataset

model_version = 'model_B_0_0001_Adam_L1Loss'

def save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_dir):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(epoch))
    torch.save(state, filename)


def loss_across_epoch(loss_list, length_list):
    total_loss = 0.0
    total_length = 0
    for i, loss in enumerate(loss_list):
        total_loss += loss * length_list[i]
        total_length += length_list[i]
    return total_loss / total_length


# class Weighted_Loss(torch.nn.Module):
#     def __init__(self):
#         super(Weighted_Loss, self).__init__()

#     def forward(self, Input, Target):
#         loss = Input - Target
#         return torch.mean((3 - (-1)*torch.sign(loss)) / 2 * (7 / (1 + 0.01 * Target)) * torch.abs(loss))        


def main():
    with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/p2_ddyn', 'r') as file:
        p2_ddyn = pickle.load(file)

    with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/partition', 'r') as file:
        partition = pickle.load(file)
    print('number of training examples: {}'.format(len(partition['training'])))
    print('number of validation examples: {}'.format(len(partition['validation'])))

    torch.manual_seed(20190717)
    training_dataset = Dataset(p2_ddyn, partition['training'])
    training_generator = data.DataLoader(training_dataset, batch_size=64, shuffle=True, num_workers=4)
    validation_dataset = Dataset(p2_ddyn, partition['validation'])
    validation_generator = data.DataLoader(validation_dataset, batch_size=256, shuffle=True, num_workers=4)

    num_epoch = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    model = Model().to(device)
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True)
    # criterion = Weighted_Loss()
    criterion = nn.L1Loss()
    
    start_from_saved_model = False
    if os.path.exists(model_version + '_checkpoint/'):
        ans = raw_input("start from saved model?   [y/n] ")
        if ans == 'y' or ans == 'Y':
            start_from_saved_model = True
        elif ans == 'n' or ans == 'N':
            shutil.rmtree(model_version + '_checkpoint/')
            os.makedirs(model_version + '_checkpoint/')
        else:
            exit(1)
    else:
        os.makedirs(model_version + '_checkpoint/')

    finished_epoch_last_time = -1

    if start_from_saved_model:
        files = [file for file in os.listdir(model_version + '_checkpoint/')
            if file.startswith('epoch=') and file.endswith('.checkpoint.pth.tar')]

        if files:
            filename = os.path.join(model_version + '_checkpoint/', 'epoch={}.checkpoint.pth.tar'.format(len(files)-1))
            checkpoint = torch.load(filename)

            try:
                finished_epoch_last_time = checkpoint['epoch']
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['optimizer_state_dict'])
                print('Successfully load the model and optimizer from epoch {}'.format(finished_epoch_last_time))
            except:
                print('Fail to load the model or optimizer from epoch {}'.format(finished_epoch_last_time))
                exit(1)

        else:
            start_from_saved_model = False

    if not start_from_saved_model:
        training_loss = []
        validation_loss = []
        print('before training')
        model.eval()
        with torch.set_grad_enabled(False):
            loss_list = []
            length_list = []
            for ground_depth_maps, wall_depth_maps, p2s, ddyns in training_generator:
                ground_depth_maps, wall_depth_maps, p2s, ddyns = ground_depth_maps.to(device), wall_depth_maps.to(device), p2s.to(device), ddyns.to(device)
                predicted_ddyns = model(ground_depth_maps, wall_depth_maps, p2s).squeeze()
                loss = criterion(predicted_ddyns, ddyns)
                loss_list.append(loss)
                length_list.append(predicted_ddyns.shape[0])
            epoch_loss = loss_across_epoch(loss_list, length_list)
            training_loss.append(epoch_loss.cpu().data.tolist())
            print('training loss: {:4.2f}'.format(epoch_loss))

            loss_list = []
            length_list = []
            for ground_depth_maps, wall_depth_maps, p2s, ddyns in validation_generator:
                ground_depth_maps, wall_depth_maps, p2s, ddyns = ground_depth_maps.to(device), wall_depth_maps.to(device), p2s.to(device), ddyns.to(device)
                predicted_ddyns = model(ground_depth_maps, wall_depth_maps, p2s).squeeze()
                loss = criterion(predicted_ddyns, ddyns)
                loss_list.append(loss)
                length_list.append(predicted_ddyns.shape[0])
            epoch_loss = loss_across_epoch(loss_list, length_list)
            validation_loss.append(epoch_loss.cpu().data.tolist())
            print('validation loss: {:4.2f}'.format(epoch_loss))

    else:
        if os.path.exists(model_version + '_loss_history'):
            with open(model_version + '_loss_history', 'r') as file:
                loss_dict = pickle.load(file)
            training_loss = loss_dict['training_loss']
            validation_loss = loss_dict['validation_loss']
        else:
            training_loss = []
            validation_loss = []

    for epoch in range(num_epoch):
        print('epoch: {}'.format(epoch + finished_epoch_last_time + 1))
        # train
        model.train()
        for ground_depth_maps, wall_depth_maps, p2s, ddyns in training_generator:
            ground_depth_maps, wall_depth_maps, p2s, ddyns = ground_depth_maps.to(device), wall_depth_maps.to(device), p2s.to(device), ddyns.to(device)
            optimizer.zero_grad()
            predicted_ddyns = model(ground_depth_maps, wall_depth_maps, p2s).squeeze()
            loss = criterion(predicted_ddyns, ddyns)
            loss.backward()
            optimizer.step()
        save_checkpoint(epoch + finished_epoch_last_time + 1, model, optimizer, scheduler, model_version + '_checkpoint/')

        # loss on training data and validation data
        model.eval()
        with torch.set_grad_enabled(False):
            loss_list = []
            length_list = []
            for ground_depth_maps, wall_depth_maps, p2s, ddyns in training_generator:
                ground_depth_maps, wall_depth_maps, p2s, ddyns = ground_depth_maps.to(device), wall_depth_maps.to(device), p2s.to(device), ddyns.to(device)
                predicted_ddyns = model(ground_depth_maps, wall_depth_maps, p2s).squeeze()
                loss = criterion(predicted_ddyns, ddyns)
                loss_list.append(loss)
                length_list.append(predicted_ddyns.shape[0])
            epoch_loss = loss_across_epoch(loss_list, length_list)
            training_loss.append(epoch_loss.cpu().data.tolist())
            print('training loss: {:4.2f}'.format(epoch_loss))

            loss_list = []
            length_list = []
            for ground_depth_maps, wall_depth_maps, p2s, ddyns in validation_generator:
                ground_depth_maps, wall_depth_maps, p2s, ddyns = ground_depth_maps.to(device), wall_depth_maps.to(device), p2s.to(device), ddyns.to(device)
                predicted_ddyns = model(ground_depth_maps, wall_depth_maps, p2s).squeeze()
                loss = criterion(predicted_ddyns, ddyns)
                loss_list.append(loss)
                length_list.append(predicted_ddyns.shape[0])
            epoch_loss = loss_across_epoch(loss_list, length_list)
            validation_loss.append(epoch_loss.cpu().data.tolist())
            print('validation loss: {:4.2f}'.format(epoch_loss))
        scheduler.step(training_loss[-1])


    plt.figure()
    plt.plot(range(len(training_loss)), training_loss, 'o-', label='Training')
    plt.plot(range(len(training_loss)), validation_loss, 'o-', label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(model_version + '_loss.png')

    loss_dict = {'training_loss': training_loss,
                 'validation_loss': validation_loss}

    with open(model_version + '_loss_history', 'w') as file:
        pickle.dump(loss_dict, file)

    print('current model: ' + model_version)



if __name__ == "__main__":
    main()

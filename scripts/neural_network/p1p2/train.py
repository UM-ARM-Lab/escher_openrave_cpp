import os, torch, pickle, shutil
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils import data

from model_v10 import Model
from dataset import Dataset

model_version = 'model_v10_00001'

def save_checkpoint(model, epoch, checkpoint_dir):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
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


def main():
    with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/p2_ddyn', 'r') as file:
        p2_ddyn = pickle.load(file)

    with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/partition', 'r') as file:
        partition = pickle.load(file)
    print('number of training examples: {}'.format(len(partition['training'])))
    print('number of validation examples: {}'.format(len(partition['validation'])))

    torch.manual_seed(20190711)
    training_dataset = Dataset(p2_ddyn, partition['training'])
    training_generator = data.DataLoader(training_dataset, batch_size=64, shuffle=True, num_workers=4)
    validation_dataset = Dataset(p2_ddyn, partition['validation'])
    validation_generator = data.DataLoader(validation_dataset, batch_size=256, shuffle=True, num_workers=4)

    learning_rate = 0.0001
    num_epoch = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    model = Model().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if os.path.exists(model_version + '_checkpoint/'):
        shutil.rmtree(model_version + '_checkpoint/')
    os.makedirs(model_version + '_checkpoint/')

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
        training_loss.append(epoch_loss)
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
        validation_loss.append(epoch_loss)
        print('validation loss: {:4.2f}'.format(epoch_loss))

    num_epoch_no_improvement = 0
    for epoch in range(num_epoch):
        print('epoch: {}'.format(epoch))
        # train
        model.train()
        for ground_depth_maps, wall_depth_maps, p2s, ddyns in training_generator:
            ground_depth_maps, wall_depth_maps, p2s, ddyns = ground_depth_maps.to(device), wall_depth_maps.to(device), p2s.to(device), ddyns.to(device)
            optimizer.zero_grad()
            predicted_ddyns = model(ground_depth_maps, wall_depth_maps, p2s).squeeze()
            loss = criterion(predicted_ddyns, ddyns)
            loss.backward()
            optimizer.step()
        save_checkpoint(model, epoch, model_version + '_checkpoint/')

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
            training_loss.append(epoch_loss)
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
            validation_loss.append(epoch_loss)
            print('validation loss: {:4.2f}'.format(epoch_loss))
        if validation_loss[-1] - validation_loss[-2] > 0:
            num_epoch_no_improvement += 1
        else:
            num_epoch_no_improvement = 0
        if num_epoch_no_improvement == 5:
            break

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



if __name__ == "__main__":
    main()

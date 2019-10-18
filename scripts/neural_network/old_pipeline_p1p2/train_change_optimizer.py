import os, torch, pickle, shutil, IPython
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils import data

from model_v14 import Model
from dataset import Dataset

model_version = 'model_v14_00001_SGD'

def save_checkpoint(epoch, model, optimizer, checkpoint_dir):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
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

    torch.manual_seed(20190717)
    training_dataset = Dataset(p2_ddyn, partition['training'])
    training_generator = data.DataLoader(training_dataset, batch_size=64, shuffle=True, num_workers=4)
    validation_dataset = Dataset(p2_ddyn, partition['validation'])
    validation_generator = data.DataLoader(validation_dataset, batch_size=256, shuffle=True, num_workers=4)

    num_epoch = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    model = Model().to(device)
    optimizer = optim.ASGD(model.parameters(), lr=0.0001)
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
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
        with open(model_version + '_loss_history', 'r') as file:
            loss_dict = pickle.load(file)
        training_loss = loss_dict['training_loss']
        validation_loss = loss_dict['validation_loss']

    num_epoch_no_improvement = 0
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
        save_checkpoint(epoch + finished_epoch_last_time + 1, model, optimizer, model_version + '_checkpoint/')

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
    
    print('current model: ' + model_version)



if __name__ == "__main__":
    main()

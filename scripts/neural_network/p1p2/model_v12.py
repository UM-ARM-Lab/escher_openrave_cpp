import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.ground_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            Flatten(),
        )

        self.wall_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            Flatten(),
        )

        self.p2_net_ground = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 8*8),
            nn.Softmax(dim=1),
        )

        self.p2_net_wall = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 5*31),
            nn.Softmax(dim=1),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1),
            nn.LeakyReLU(),
        )
    

    def forward(self, ground_depth_map, wall_depth_map, p2):
        """
        "ground_depth_map" is a 2d array of shape (1, 65, 65)
        "wall_depth_map" is a 2d array of shape (1, 41, 252)
        "p2" is a 1d array of shape (3,)
        """
        arr1 = self.ground_net(ground_depth_map)
        arr1 = arr1.view(-1, 64, 8*8)
        p2_ground = self.p2_net_ground(p2)
        arr1 = torch.bmm(arr1, p2_ground.unsqueeze(2)).squeeze()
        arr2 = self.wall_net(wall_depth_map)
        arr2 = arr2.view(-1, 64, 5*31)
        p2_wall = self.p2_net_wall(p2)
        arr2 = torch.bmm(arr2, p2_wall.unsqueeze(2)).squeeze()
        arr = torch.cat((arr1, arr2), 1)
        z = self.fc(arr)
        return z
    

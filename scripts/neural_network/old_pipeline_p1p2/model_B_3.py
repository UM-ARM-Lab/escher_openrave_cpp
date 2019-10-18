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
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            Flatten(),
        )

        self.wall_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            Flatten(),
        )

        self.p2_net_ground = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 7*7*5),
            nn.LeakyReLU(),
        )

        self.p2_net_wall = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 6*120*5),
            nn.LeakyReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(32*5 + 32*5, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(512, 64),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.LeakyReLU(),
        )
    

    def forward(self, ground_depth_map, wall_depth_map, p2):
        """
        "ground_depth_map" is a 2d array of shape (1, 65, 65)
        "wall_depth_map" is a 2d array of shape (1, 25, 252)
        "p2" is a 1d array of shape (3,)
        """
        arr1 = self.ground_net(ground_depth_map)
        arr1 = arr1.view(-1, 32, 7*7)
        p2_ground = self.p2_net_ground(p2).view(-1, 7*7, 5)
        arr1 = torch.bmm(arr1, p2_ground).view(-1, 32*5)
        arr2 = self.wall_net(wall_depth_map)
        arr2 = arr2.view(-1, 32, 6*120)
        p2_wall = self.p2_net_wall(p2).view(-1, 6*120, 5)
        arr2 = torch.bmm(arr2, p2_wall).view(-1, 32*5)
        arr = torch.cat((arr1, arr2), 1)
        z = self.fc(arr)
        return z
    

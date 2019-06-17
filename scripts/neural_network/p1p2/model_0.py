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
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            Flatten(),
        )

        self.wall_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            Flatten(),
        )

        self.p2_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64*13*13 + 64*7*59 + 128, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
             nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 32),
            nn.LeakyReLU(),
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
        arr2 = self.wall_net(wall_depth_map)
        arr3 = self.p2_net(p2)
        arr = torch.cat((arr1, arr2, arr3), 1)
        z = self.fc(arr)
        return z
    

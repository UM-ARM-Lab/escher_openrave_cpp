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
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0),
            nn.LeakyReLU(),

            Flatten(),
        )

        self.left_wall_net = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1),
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

            Flatten(),
        )

        self.right_wall_net = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
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

            Flatten(),
        )

        self.p2_net = nn.Sequential(
            nn.Linear(3, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64*5*5 + 64*6*63 + 1024, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.LeakyReLU(),
        )
    

    def forward(self, ground_map, left_wall_map, right_wall_map, p2):
        """
        "ground_map" is a 3d array of shape (2, 65, 65)
        "left_wall_map" is a 3d array of shape (2, 25, 84)
        "right_wall_map" is a 3d array of shape (2, 25, 84)
        "p2" is a 1d array of shape (3,)
        """
        arr1 = self.ground_net(ground_map)
        arr2 = self.left_wall_net(left_wall_map)
        arr3 = self.right_wall_net(right_wall_map)
        arr4 = self.p2_net(p2)
        arr = torch.cat((arr1, arr2, arr3, arr4), 1)
        z = self.fc(arr)
        return z
    

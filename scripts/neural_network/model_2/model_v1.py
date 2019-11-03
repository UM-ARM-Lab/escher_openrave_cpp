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
        self.ground_mask_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 65 * 65),
            nn.LeakyReLU(),
        )
        
        self.left_wall_mask_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 25 * 84),
            nn.LeakyReLU(),
        )
        
        self.right_wall_mask_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 25 * 84),
            nn.LeakyReLU(),
        )
        
        self.ground_net = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            #nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=0),
            #nn.LeakyReLU(),
   
            Flatten(),
        )

        self.left_wall_net = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            #nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=0),
            #nn.LeakyReLU(),
   
            Flatten(),
        )
        
        self.right_wall_net = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            #nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=0),
            #nn.LeakyReLU(),
   
            Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(4 * 24 * 24 + 4 * 4 * 34 * 2, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            
            nn.ReLU(),
        )
    

    def forward(self, ground_map, left_wall_map, right_wall_map, p2):
        """
        "ground_map" is a 3d array of shape (2, 65, 65)
        "left_wall_map" is a 3d array of shape (2, 25, 84)
        "right_wall_map" is a 3d array of shape (2, 25, 84)
        "p2" is a 1d array of shape (3,)
        """
        ground_mask = self.ground_mask_net(p2)
        ground_mask = ground_mask.view(-1, 1, 65, 65)
        masked_ground_map = ground_mask * ground_map
        ground = self.ground_net(masked_ground_map)
        
        left_wall_mask = self.left_wall_mask_net(p2)
        left_wall_mask = left_wall_mask.view(-1, 1, 25, 84)
        masked_left_wall_map = left_wall_mask * left_wall_map
        left_wall = self.left_wall_net(masked_left_wall_map)
        
        right_wall_mask = self.right_wall_mask_net(p2)
        right_wall_mask = right_wall_mask.view(-1, 1, 25, 84)
        masked_right_wall_map = right_wall_mask * right_wall_map
        right_wall = self.right_wall_net(masked_right_wall_map)
        
        arr = torch.cat((ground, left_wall, right_wall), 1)
        z = self.fc(arr)
        return z
    

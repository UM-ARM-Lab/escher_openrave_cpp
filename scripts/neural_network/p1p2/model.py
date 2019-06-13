import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    

    def forward(self, ground_depth_map, wall_depth_map, p2)
    """
    "ground_depth_map" is a 2d array of shape (61, 61)
    "wall_depth_map" is a 2d array of shape (41, 252)
    "p2" is a 1d array of shape (3,)
    """




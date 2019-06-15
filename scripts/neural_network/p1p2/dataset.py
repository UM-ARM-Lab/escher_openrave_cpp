import torch, numpy, pickle
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, p2_ddyn, example_ids):
        self.p2_ddyn = p2_ddyn
        self.example_ids = example_ids

    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, index):
        # example_id is a string
        example_id = self.example_ids[index]
        depth_map_id, p2, ddyn = self.p2_ddyn[example_id]
        with open('../../../data/ground_truth_p1p2/ground_depth_maps/' + depth_map_id, 'r') as file:
            ground_depth_map = pickle.load(file)
        with open('../../../data/ground_truth_p1p2/wall_depth_maps/' + depth_map_id, 'r') as file:
            wall_depth_map = pickle.load(file)
        return torch.unsqueeze(torch.from_numpy(ground_depth_map).float(), 0), torch.unsqueeze(torch.from_numpy(wall_depth_map).float(), 0), torch.from_numpy(p2).float(), ddyn
        
        
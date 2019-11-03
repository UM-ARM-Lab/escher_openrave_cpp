import torch, pickle
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, ground_truth_dict, example_ids):
        self.ground_truth_dict = ground_truth_dict
        self.example_ids = example_ids

    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, index):
        # example_id is a string
        example_id = self.example_ids[index]
        ground_map_id, wall_map_id, p2, _, dynamics_cost = self.ground_truth_dict[example_id]
        with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/ground_depth_and_boundary_maps/' + ground_map_id, 'r') as file:
            ground_map = pickle.load(file)
        with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/wall_depth_and_boundary_maps/' + wall_map_id + '_L', 'r') as file:
            left_wall_map = pickle.load(file)
        with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/wall_depth_and_boundary_maps/' + wall_map_id + '_R', 'r') as file:
            right_wall_map = pickle.load(file)
        return ground_map, left_wall_map, right_wall_map, p2, dynamics_cost
        
        
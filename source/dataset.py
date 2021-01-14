import os
import torch
from torch_geometric.data import InMemoryDataset
from utils import *

def load_dataset_scaffold(path, dataset='bace', seed=68, tasks=None): # dataset='hiv' seed='628

    save_path = path + 'processed/train_valid_test_{}_seed_{}.ckpt'.format(dataset, seed)
    if os.path.isfile(save_path):
        trn, val, test = torch.load(save_path)
        return trn, val, test

    """There are two weights: (+)+(-) over (-) and (+)+(-) over (+)"""

    torch.save([trn, val, test], save_path)
    return load_dataset_scaffold(path, dataset, seed, tasks)


class MultiDataset(InMemoryDataset):

    def __init__(self, root, dataset, tasks, transform=None, pre_transform=None, pre_filter=None):
        self.tasks = tasks
        self.dataset = dataset

        self.weights = 0
        super(MultiDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

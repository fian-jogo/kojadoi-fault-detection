"""
dataset.py
PyTorch Dataset wrapper untuk fault detection.
"""
import torch
from torch.utils.data import Dataset


class FaultWindowDataset(Dataset):
    """
    Dataset PyTorch untuk window fault detection.
    
    Setiap sample:
        X      : (T, N_NODES_TOTAL, N_FEATURES)
        y_det  : scalar (0/1)
        y_loc  : (N_NODES_TOTAL,) binary mask
    """
    def __init__(self, X, y_detect, y_loc):
        self.X = torch.from_numpy(X).float()
        self.y_detect = torch.from_numpy(y_detect).long()
        self.y_loc = torch.from_numpy(y_loc).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_detect[idx], self.y_loc[idx]
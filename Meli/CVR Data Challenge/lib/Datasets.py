import torch
from torch.utils.data import Dataset
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, continius_features:np.ndarray, categorical_features:np.ndarray, y:np.ndarray=None):    
        self.continius_features = torch.tensor(continius_features, dtype=torch.float)
        self.categorical_features = torch.tensor(categorical_features, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float) if y is not None else None

    def __len__(self):
        return len(self.continius_features)

    def __getitem__(self, idx):
        sample = self.continius_features[idx], self.categorical_features[idx], self.y[idx] if self.y is not None else None
        return sample
    
    def reverse_transform(self):
        return self.continius_features.numpy(), self.categorical_features.numpy(), self.y.numpy() if self.y is not None else None
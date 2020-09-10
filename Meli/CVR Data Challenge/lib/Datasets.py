import torch
from torch.utils.data import Dataset
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, numerical_features:np.ndarray, categorical_features:np.ndarray, y:np.ndarray=None):    
        self.numerical_features = torch.tensor(numerical_features, dtype=torch.float)
        self.categorical_features = torch.tensor(categorical_features, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float) if y is not None else None

    def __len__(self):
        return len(self.numerical_features)

    def __getitem__(self, idx):
        sample = self.numerical_features[idx], self.categorical_features[idx], self.y[idx] if self.y is not None else np.nan
        return sample
    
    def reverse_transform(self):
        return self.numerical_features.numpy(), self.categorical_features.numpy(), self.y.numpy() if self.y is not None else np.nan
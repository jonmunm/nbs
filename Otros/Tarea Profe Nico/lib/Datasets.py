import torch
from torch.utils.data import Dataset
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, X_numerical, X_categorical, Y=None, Y_dtype=None):    
        self.X_numerical = torch.tensor(X_numerical, dtype=torch.float)
        self.X_categorical = torch.tensor(X_categorical, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=Y_dtype) if Y is not None else None

    def __len__(self):
        return len(self.X_numerical)

    def __getitem__(self, idx):
        sample = self.X_numerical[idx], self.X_categorical[idx], self.Y[idx] if self.Y is not None else np.nan
        return sample
    
    def reverse_transform(self):
        return self.X_numerical.numpy(), self.X_categorical.numpy(), self.Y.numpy() if self.Y is not None else np.nan
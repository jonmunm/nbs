import torch
from torch.utils.data import Dataset
import numpy as np

class CustomTensorDataset(Dataset):
    def __init__(self, features:np.ndarray, y:np.ndarray, normalize=True):
        if normalize:
            features = (features - features.mean(axis=0))/features.std(axis=0)
        
        features, y = torch.tensor(features, dtype=torch.double), torch.tensor(y, dtype=torch.double)
        n = 1 if len(features.shape) == 1 else features.shape[1]
        
        self.features = torch.reshape(features, (-1, n))
        self.n = n
        self.y = torch.reshape(y, (-1, 1))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.features[idx], self.y[idx]
        return sample
    
    def reverse_transform(self):
        return self.features.numpy().squeeze(), self.y.numpy().squeeze()
    
    def get_dataset(self):
        return self.features, self.y
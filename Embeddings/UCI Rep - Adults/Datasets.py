import torch
from torch.utils.data import Dataset
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, continius_features:np.ndarray, categorical_features:np.ndarray, y:np.ndarray, normalize=True):
        if normalize:
            features = (features - features.mean(axis=0))/features.std(axis=0)
        
        self.continius_features = torch.tensor(continius_features, dtype=torch.float)
        self.categorical_features = torch.tensor(categorical_features, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.float)
        self.y = torch.reshape(y, (-1, 1))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.continius_features[idx], self.categorical_features[idx], self.y[idx]
        return sample
    
    def reverse_transform(self):
        return self.continius_features.numpy().squeeze(), self.categorical_features.numpy().squeeze(), self.y.numpy().squeeze()
    
#    def get_dataset(self):
#        return self.features, self.y
import torch
from torch.utils.data import Dataset
import numpy as np

class ProfitDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        data = np.loadtxt(txt_file, delimiter=',')
        features, y = torch.tensor(data[:,0], dtype=torch.double), torch.tensor(data[:,1], dtype=torch.double)
        
        self.transform = transform
        self.features = torch.reshape(features, (len(features), 1))
        self.y = torch.reshape(y, (len(y), 1))
        
        print(self.features.shape)
        print(self.y.shape)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        sample = self.features[idx], self.y[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def to_numpy(self):
        return self.features.numpy().squeeze(), self.y.numpy().squeeze()
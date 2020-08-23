import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
    
class MnistClassifier(nn.Module):
    def __init__(self, D_in, D_out):
        super(MnistClassifier, self).__init__()
        self.layer1=nn.Sequential(
            # Layer 1
            nn.Linear(D_in, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),            
        )
    
        self.layer2=nn.Sequential(
            # Layer 1
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),            
        )
        
        self.output=nn.Sequential(
            # Layer 1
            nn.Linear(64, D_out),
            nn.ReLU(),
            nn.BatchNorm1d(D_out),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.output(x)
    
    def fit(self, train_dl:DataLoader, epochs:int, opt:Optimizer, loss_fn:any) -> list:
        self.train()
        losses = []
        for i in range(epochs):
            for x, y in train_dl:
                y_pred = self.forward(x)
                loss = loss_fn(y_pred, y)
                losses.append(loss.item())

                loss.backward()
                opt.step()
                opt.zero_grad()
        return losses
    
    def predict(self, data_loader:DataLoader) -> torch.Tensor:
        self.eval()
        predictions = []
        with torch.no_grad():
            for x, _ in data_loader:
                preds = self.forward(x)
                predictions.append(preds)
        return torch.cat(predictions)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
    
class SimpleLogisticModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(SimpleLogisticModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.BatchNorm1d(H),
            nn.Linear(H, D_out),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        preds = self.layer(x)
        return preds
    
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

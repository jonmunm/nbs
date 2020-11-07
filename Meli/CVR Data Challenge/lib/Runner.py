import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch import tensor, Tensor

class Runner:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    def fit(self, data_loader, epochs, optimizer, scheduler, loss_fn):
        self.model.train()
        losses = []
        for i in range(epochs):
            for x_numerical, x_categorical, y in data_loader:
                y_pred = self.model.forward(x_numerical, x_categorical)
                loss = loss_fn(y_pred, y)
                losses.append(loss.item())

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                scheduler.step()
                
        return losses
    
    def predict(self, data_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for x_numerical, x_categorical, y in data_loader:
                preds = self.model.forward(x_numerical, x_categorical)
                predictions.append(preds)
        return torch.cat(predictions)
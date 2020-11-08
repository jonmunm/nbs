import math
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch import tensor, Tensor
from sklearn.metrics import roc_auc_score

class Runner:
    def __init__(self, model, optimizer, loss_fn, batch_scheduler=None, epoch_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batch_scheduler = batch_scheduler
        self.epoch_scheduler = epoch_scheduler
    
    def fit(self, train_data_loader, validation_data_loader, epochs, return_stats=True):
        self.model.train()
        losses = []
        stats = []
        
        for i in range(epochs):
            for X_numerical, X_categorical, Y in train_data_loader:
                predictions = self.model.forward(X_numerical, X_categorical)
                loss = self.loss_fn(predictions, Y)
                losses.append(loss.item())

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.batch_scheduler is not None:
                    self.batch_scheduler.step()
                    
            if self.epoch_scheduler is not None:
                self.epoch_scheduler.step()
                
            if return_stats:
                dl = DataLoader(train_data_loader.dataset, train_data_loader.batch_size, num_workers=train_data_loader.num_workers)
                train_predictions = self.predict(dl, return_numpy=False)
                train_loss = self.loss_fn(train_predictions, train_data_loader.dataset.Y).item()
                
                validation_predictions = self.predict(validation_data_loader, return_numpy=False)
                validation_loss = self.loss_fn(validation_predictions, validation_data_loader.dataset.Y).item()

                roc_auc = roc_auc_score(train_data_loader.dataset.Y.numpy(), train_predictions.numpy())

                stats.append({
                    'epoch' : i+1,
                    'train_loss' : train_loss,
                    'validation_loss' : validation_loss,
                    'roc_auc' : roc_auc
                })
         
        if return_stats:
            display(pd.DataFrame(stats))
                
        return losses
    
    def predict(self, data_loader, return_numpy=True):
        self.model.eval()
        predictions_list = []
        with torch.no_grad():
            for X_numerical, X_categorical, Y in data_loader:
                predictions = self.model.forward(X_numerical, X_categorical)
                predictions_list.append(predictions)
                
        return torch.cat(predictions_list).numpy() if return_numpy else torch.cat(predictions_list)
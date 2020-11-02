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
        
    def find_lr(self, data_loader, init_value=1e-8, final_value=10., beta=0.98):
        num = len(data_loader)-1
        mult = (final_value / init_value) ** (1/num)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        
        for x_numerical, x_categorical, y in data_loader:
            batch_num += 1
            
            #As before, get the loss for this mini-batch of inputs/outputs
            #inputs,labels = data
            #inputs, labels = Variable(inputs), Variable(labels)
            #optimizer.zero_grad()
            
            y_pred = self.model.forward(x_numerical, x_categorical)
            loss = self.loss_fn(y_pred, y)
            
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1-beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                return log_lrs, losses
            
            #Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
                
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            
            #Do the SGD step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            #Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr
        return log_lrs, losses
    
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
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import tensor, Tensor
    
class FullyConnectedNetworkRegressor(nn.Module):
    def __init__(self, q_numerical_features:int, q_categorical_features:int, hidden_layers_size:list, embedding_dims:list=None):
        super(FullyConnectedNetworkRegressor, self).__init__()
        
        self.hidden_layers_size = hidden_layers_size
        
        if embedding_dims is not None:
            embedding_sizes = sum([embedding_size for _, embedding_size in embedding_dims])
            self.embeddings_layer=nn.ModuleList(
                [nn.Embedding(vocabulary_size, embedding_size) for vocabulary_size, embedding_size in embedding_dims]
            )
            self.embedding_dropout = nn.Dropout(0.6)
        else:
            embedding_sizes = 0
            self.embeddings_layer = None
        
        self.layer_0 = nn.Sequential(
            nn.Linear(embedding_sizes + q_numerical_features, hidden_layers_size[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_layers_size[0]),            
        ) 
        
        for i, hidden_size in enumerate(hidden_layers_size[1:]):
            layer = nn.Sequential(
                nn.Linear(hidden_layers_size[i], hidden_layers_size[i+1]),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_layers_size[i+1]),            
            )
            setattr(self, f'layer_{i+1}', layer)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_layers_size[-1], 1),
            nn.Sigmoid()
        )
        
    def forward(self, numerical_features:Tensor, categorical_features:Tensor) -> Tensor:
        if self.embeddings_layer is not None:
            embeds = [emb_layer(categorical_features[:, i]) for i, emb_layer in enumerate(self.embeddings_layer)] 
            embeds = torch.cat(embeds, 1)
            x = self.embedding_dropout(embeds)
        else:
            embeds = tensor([])
        
        x = self.layer_0(torch.cat([embeds, numerical_features], 1))
        for i in range(1, self.hidden_layers_size):
            x = getattr(self, f'layer_{i}')(x)
        
        return self.output(x)
    
    def fit(self, train_dl:DataLoader, epochs:int, opt:Optimizer, loss_fn:any) -> list:
        self.train()
        losses = []
        for i in range(epochs):
            for x_numerical, x_categorical, y in train_dl:
                y_pred = self.forward(x_numerical, x_categorical)
                loss = loss_fn(y_pred, y)
                losses.append(loss.item())

                loss.backward()
                opt.step()
                opt.zero_grad()
        return losses
    
    def predict(self, data_loader:DataLoader) -> Tensor:
        self.eval()
        predictions = []
        with torch.no_grad():
            for x_numerical, x_categorical, y in data_loader:
                preds = self.forward(x_numerical, x_categorical)
                predictions.append(preds)
        return torch.cat(predictions)
    
    def dumps(self, filename:str):
        with open(filename, 'wb') as f:
            return pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loads(obj):
        return pickle.loads(obj)

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(D_in, D_out)
        
    def forward(self, x):
        preds = self.fc1(x)
        return preds
    
class UnivariateLinearModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(UnivariateLinearModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )
        
    def forward(self, x):
        preds = self.layer(x)
        return preds
    '''
    def __init__(self, D_in, H, D_out):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear1 = nn.Linear(H, D_out)
        
    def forward(self, x):
        out1 = self.linear1(x)
        return self.linear2(F.relu(out1))
    '''
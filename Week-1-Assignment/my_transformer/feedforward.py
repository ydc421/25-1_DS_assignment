import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super(FeedForwardLayer, self).__init__()
        #TODO two lines!
        self.first_layer = nn.Linear(d_model, d_ff)
        self.second_layer = nn.Linear(d_ff, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        #TODO one line!
        return self.second_layer(F.relu(self.first_layer(x)))

class DropoutLayer(nn.Module):
    def __init__(self, p: float) -> None:
        super(DropoutLayer, self).__init__()
        self.dropout = nn.Dropout(p)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x)

class ActivationLayer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        #TODO one line! -> Original: torch.where(x > 0, x, 0) -> Not optimized so hence, we will be using F.relu
        return F.relu(x)
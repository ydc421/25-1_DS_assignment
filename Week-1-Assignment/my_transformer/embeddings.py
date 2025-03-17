import torch
import torch.nn as nn
import math
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionEmbedding, self).__init__()
        #TODO
        pos_emb = torch.zeros(max_len, d_model) # X axis = the number of tokens, Y axis = length of each individual embedding vector

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1) # Turns this into a (d_model, 1)
        expon = (torch.arange(start= 0, end = d_model, step=2).float() / d_model)
        ratio = torch.pow(10000, expon)

        pos_emb[:,0::2] = torch.sin(position/ratio) # (d_model, d_model/2) where each vector is represented by a different dimension
        pos_emb[:,1::2] = torch.cos(position/ratio)
        # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
        # Code is based on this implementation: https://discuss.pytorch.org/t/what-does-register-buffer-do/121091
        self.register_buffer('pos_emb', pos_emb.unsqueeze(0))
    
    def forward(self, x: Tensor) -> Tensor:
        #TODO one line!
        return self.pos_emb[:, :x.size(1), :]
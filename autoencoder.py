import torch
from torch import nn, optim
from torch.nn import functional as F


class Coder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=True):
        super(Coder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.dl = nn.Dropout()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.dropout:
            return self.dl(F.sigmoid(self.fc1(x)))
        else:
            return F.sigmoid(self.fc1(x))

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=True):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = Coder(input_dim, hidden_dim, dropout=dropout)
        self.fc2 = Coder(hidden_dim, input_dim, dropout=dropout)

    def forward(self, x):
        encoded = self.fc1(x)
        decoded = self.fc2(encoded)
        return encoded, decoded
        

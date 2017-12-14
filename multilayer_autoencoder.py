import torch
from torch import nn, optim
from torch.nn import functional as F


class MultilayerCoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_dims, dropout=True):
        super(MultilayerCoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = len(hidden_layer_dims)
        self.dropout = dropout
        self.dl = nn.Dropout()
        fcs = []
        curr_dim = input_dim
        for dim in hidden_layer_dims:
            fcs.append(nn.Linear(curr_dim, dim))
            curr_dim = dim
        fcs.append(nn.Linear(curr_dim, output_dim))
        self.fcs = nn.ModuleList(fcs)

    def forward(self, x):
        out = x
        for fc in self.fcs:
            if (self.dropout):
                out = self.dl(F.sigmoid(fc(out)))
            else:
                out = F.sigmoid(fc(out))
        return out


class MultilayerAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layer_dims, dropout=True):
        super(MultilayerAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = len(hidden_layer_dims)
        self.c1 = MultilayerCoder(input_dim, hidden_dim, hidden_layer_dims, dropout=dropout)
        self.c2 = MultilayerCoder(
            hidden_dim, input_dim, hidden_layer_dims[::-1], dropout=dropout)

    def forward(self, x):
        encoded = self.c1(x)
        decoded = self.c2(encoded)
        return encoded, decoded

import torch
import torch.nn as nn
from model.parameters import DROPOUT


class MLP(nn.Module):
    def __init__(
        self, linear_layers, dropout=DROPOUT, skip_connection=False, skip_head=0
    ):
        super(MLP, self).__init__()
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.linears = nn.ModuleList(
            [
                nn.Linear(linear_layers[i], linear_layers[i + 1])
                for i in range(len(linear_layers) - 1)
            ]
        )
        self.bns = nn.ModuleList(
            [
                nn.BatchNorm1d(linear_layers[i + 1])
                for i in range(len(linear_layers) - 2)
            ]
        )

    def forward(self, x):
        out = x
        for i in range(len(self.linears)):
            out = self.linears[i](out)
            if i < len(self.bns):
                out = self.bns[i](out)
                out = self.activation(out)
                out = self.dropout(out)
        return out

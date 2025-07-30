import torch
import torch.nn as nn
from model.parameters import DROPOUT


class MLP(nn.Module):
    def __init__(
        self, linear_layers, dropout=DROPOUT, skip_connection=True, skip_head=0
    ):
        super(MLP, self).__init__()
        self.linear_layers = linear_layers
        self.skip_connection = skip_connection
        self.skip_head = skip_head
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.GELU()  # ✅ GELU activation

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
        identity = None
        for i in range(len(self.linears)):
            out = self.linears[i](out)
            if i < len(self.bns):  # Skip BN + GELU + Dropout for last layer
                out = self.bns[i](out)
                out = self.activation(out)  # ✅ GELU instead of ReLU
                out = self.dropout(out)

                if self.skip_connection:
                    if i == self.skip_head:
                        identity = out
                    elif identity is not None and (i - self.skip_head) % 2 == 0:
                        out = out + identity
                        identity = out
        return out

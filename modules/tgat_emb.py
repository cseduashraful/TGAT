# === TGAT MODULES ===
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class TimeEncode(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.freqs = nn.Parameter(torch.randn(dim // 2), requires_grad=False)

    def forward(self, t):
        t = t.unsqueeze(-1) * self.freqs * 2 * torch.pi
        return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)

class TGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__(aggr='add')
        self.time_enc = TimeEncode(time_dim)
        self.lin = nn.Linear(in_channels + time_dim, out_channels)
        self.attn = nn.Parameter(torch.Tensor(1, out_channels))
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, edge_index, edge_t):
        edge_attr = self.time_enc(edge_t)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr, index):
        msg = torch.cat([x_j, edge_attr], dim=-1)
        h = torch.tanh(self.lin(msg))
        alpha = softmax((h * self.attn).sum(dim=-1), index)
        return h * alpha.unsqueeze(-1)

class TGATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv = TGATConv(in_channels, out_channels, time_dim)

    def forward(self, x, edge_index, edge_t):
        return self.conv(x, edge_index, edge_t)

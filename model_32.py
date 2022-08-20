import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dropout_adj
import numpy as np


class GT32dim(nn.Module):
    def __init__(self,
                 num_classes,
                 regression,
                 adj_dropout=0,
                 # adj_dropout=0.2,
                 force_undirected=False,
                 latent_dim=None,
                 t_conv=TransformerConv
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.regression = regression
        self.adj_dropout = adj_dropout
        self.force_undirected = force_undirected

        if latent_dim is None:
            latent_dim = [(32, 32), (32, 32), (32, 32), (32, 32), (32, 32)]

        self.convs = torch.nn.ModuleList()
        for i in range(0, len(latent_dim)-1):
            self.convs.append(t_conv(latent_dim[i], latent_dim[i+1][0]))

        self.lin1 = Linear(sum(np.array(latent_dim)[1:, 0]),
                           sum(np.array(latent_dim)[1:, 0])//4)
        self.lin2 = Linear(sum(np.array(latent_dim)[1:, 0])//4, 128)
        if self.regression:
            self.lin3 = Linear(128, 1)
        else:
            self.lin3 = Linear(128, self.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        feature_0, feature_1 = torch.chunk(x, 2, dim=1)
        x = [feature_0, feature_1]
        if self.adj_dropout > 0:
            edge_index, _ = dropout_adj(
                edge_index, p=self.adj_dropout,
                # num_nodes 확인 ...
                force_undirected=self.force_undirected, num_nodes=len(x),
                training=self.training
            )

        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
            x = (x, x)
        concat_states = torch.cat(concat_states, 1)

        x = self.lin1(concat_states)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)  # regression  # 128 -> 1

        if self.regression:
            out = x.squeeze()
        else:
            out = F.log_softmax(x, dim=-1)
        return out

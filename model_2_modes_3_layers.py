import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn import global_mean_pool
import numpy as np


class GT32dim_2_modes_3_layers(nn.Module):
    def __init__(self,
                 num_classes,
                 regression,
                 adj_dropout=0,
                 # adj_dropout=0.2,
                 conv=TransformerConv,
                 num_heads=1,
                 dropout=False,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.regression = regression
        self.adj_dropout = adj_dropout
        self.dropout = dropout

        latent_dim = [(32, 32), (32, 32), (32, 32),
                      (32, 32), (32, 32),
                      (32, 32), (32, 32)]

        self.convs = torch.nn.ModuleList()
        for i in range(0, len(latent_dim) - 1):
            if i % 2 == 0:
                self.convs.append(conv(latent_dim[i], latent_dim[i + 1][0] // num_heads,
                                       heads=num_heads, edge_dim=latent_dim[i + 1][0]))
            else:
                self.convs.append(conv(latent_dim[i], latent_dim[i + 1][0] // num_heads,
                                       heads=num_heads))

        conv_out_dim = sum(np.array(latent_dim)[1:, 0])
        # num. of parameters : 100369, test rmse : 20.3
        # self.lin1 = Linear(conv_out_dim, conv_out_dim)
        # self.lin2 = Linear(conv_out_dim, conv_out_dim // 2)
        # self.lin3 = Linear(conv_out_dim // 2, conv_out_dim // 2)
        # self.lin4 = Linear(conv_out_dim // 2, conv_out_dim // 4)
        # self.lin5 = Linear(conv_out_dim // 4, conv_out_dim // 4)
        # if self.regression:
        #     self.lin6 = Linear(conv_out_dim // 4, 1)
        # else:
        #     self.lin6 = Linear(conv_out_dim // 4, self.num_classes)

        # num. of parameters : 55753, test rmse : 20.2
        # max node : 10000 이어도 결과 동일
        self.lin1 = Linear(conv_out_dim, conv_out_dim// 2)
        self.lin2 = Linear(conv_out_dim// 2, conv_out_dim // 4)
        self.lin3 = Linear(conv_out_dim // 4, conv_out_dim // 4)
        self.lin4 = Linear(conv_out_dim // 4, conv_out_dim // 8)
        self.lin5 = Linear(conv_out_dim // 8, conv_out_dim // 8)
        if self.regression:
            self.lin6 = Linear(conv_out_dim // 8, 1)
        else:
            self.lin6 = Linear(conv_out_dim // 8, self.num_classes)

        # num. of parameters : 33073, rest rmse : 42.14
        # self.lin1 = Linear(conv_out_dim, conv_out_dim// 8)
        # if self.regression:
        #     self.lin2 = Linear(conv_out_dim // 8, 1)
        # else:
        #     self.lin2 = Linear(conv_out_dim // 8, self.num_classes)

        # num. of parameters : 65665, rest rmse : 41.68
        # self.lin1 = Linear(conv_out_dim, conv_out_dim)
        # if self.regression:
        #     self.lin2 = Linear(conv_out_dim, 1)
        # else:
        #     self.lin2 = Linear(conv_out_dim, self.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()
        self.lin5.reset_parameters()
        self.lin6.reset_parameters()

    def forward(self, data):
        inter_emb, inter_index, edge_attr, \
        uv_target_index, uv_target_emb, target_uv_index, batch \
            = data.x, data.edge_index, data.edge_attr, \
              data.uv_target_index, data.uv_target_emb, data.target_uv_index, data.batch

        feature_0, feature_1 = torch.chunk(inter_emb, 2, dim=1)
        edge_attr = edge_attr.reshape(-1, 1).repeat(1, feature_0.shape[1])
        x = [feature_0, feature_1]

        concat_states = []
        for i in range(len(self.convs)):
            if i % 2 == 0:
                x = torch.tanh(self.convs[i](x, inter_index, edge_attr))
                concat_states.append(x)
            elif i % 2 == 1:
                x = [x, uv_target_emb]
                x = torch.tanh(self.convs[i](x, uv_target_index))
                concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        x =self.lin1(concat_states)
        x = global_mean_pool(x, batch)
        x =  F.relu(self.lin2(x))
        if self.dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        x = self.lin4(x)
        x = self.lin5(x)
        x = self.lin6(x)

        if self.regression:
            out = x.squeeze()
        else:
            out = F.log_softmax(x, dim=-1)
        return out

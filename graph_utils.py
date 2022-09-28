from __future__ import print_function

import multiprocessing as mp
import os
import random
import time
import warnings

import networkx as nx
import numpy as np
import scipy.sparse as ssp
import torch
import torch.multiprocessing
from torch_geometric.data import Data, Dataset, InMemoryDataset
from tqdm import tqdm

warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))

torch.multiprocessing.set_sharing_strategy('file_system')


class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return ssp.csr_matrix((data, indices, indptr), shape=shape)


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return ssp.csc_matrix((data, indices, indptr), shape=shape)


class DynamicFeatures(Dataset):
    def __init__(self, root, A, links, labels,
                 h, sample_ratio, max_nodes_per_hop,
                 regression,
                 u_features, v_features, u_dict, v_dict, class_values,
                 use_edge_feature=True,
                 max_num=None):
        super(DynamicFeatures, self).__init__(root)
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.regression = regression
        self.u_features = u_features
        self.v_features = v_features
        self.u_dict = u_dict
        self.v_dict = v_dict
        self.class_values = class_values
        self.use_edge_feature = use_edge_feature
        if max_num is not None:
            np.random.seed(123)
            num_links = len(links[0])
            perm = np.random.permutation(num_links)
            perm = perm[:max_num]
            self.links = (links[0][perm], links[1][perm])
            self.labels = labels[perm]

    def len(self):
        return len(self.links[0])

    def get(self, idx):
        i, j = self.links[0][idx], self.links[1][idx]
        g_label = self.labels[idx]

        tmp = subgraph_features(
            (i, j), self.Arow, self.Acol,
            self.regression,
            self.u_features, self.v_features, self.u_dict, self.v_dict,
            self.h, self.sample_ratio, self.max_nodes_per_hop,
            self.class_values, g_label
        )
        # return construct_pyg_features(*tmp)
        return construct_pyg_data(*tmp)

def subgraph_features(ind, Arow, Acol,
                       regression,
                       u_features, v_features, u_dict, v_dict,
                       h=1, sample_ratio=1.0, max_nodes_per_hop=None,
                       class_values=None, y=1):
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_visited, v_visited = {ind[0]}, {ind[1]}
    u_fringe, v_fringe = {ind[0]}, {ind[1]}
    for dist in range(1, h + 1):
        v_fringe, u_fringe = neighbors(u_fringe, Arow), neighbors(v_fringe, Acol)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)
        if sample_ratio < 1.0:
            u_fringe = random.sample(u_fringe, int(sample_ratio * len(u_fringe)))
            v_fringe = random.sample(v_fringe, int(sample_ratio * len(v_fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                u_fringe = random.sample(u_fringe, max_nodes_per_hop)
            if max_nodes_per_hop < len(v_fringe):
                v_fringe = random.sample(v_fringe, max_nodes_per_hop)
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)

    subgraph = Arow[u_nodes][:, v_nodes]
    subgraph[0, 0] = 0

    u, v, r = ssp.find(subgraph)
    r = r - 1
    y = class_values[y]

    # extract embedding vector using u_nodes index !
    u_node_indices = np.array(u_nodes)[u]
    u_emb_indices = torch.LongTensor([u_dict[i] for i in u_node_indices])
    sub_u_emb = u_features(u_emb_indices)
    target_u_index = np.array(u_nodes)[0]
    target_u_emb = u_features(torch.LongTensor([u_dict[target_u_index]]))

    v_node_indices = np.array(v_nodes)[v]
    v_emb_indices = torch.LongTensor([v_dict[j] for j in v_node_indices])
    sub_v_emb = v_features(v_emb_indices)
    target_v_index = np.array(v_nodes)[0]
    target_v_emb = v_features(torch.LongTensor([v_dict[target_v_index]]))

    uv_emb = np.vstack((sub_u_emb, sub_v_emb))
    vu_emb = np.vstack((sub_v_emb, sub_u_emb))

    u_target_emb = np.tile(target_u_emb, (len(sub_u_emb), 1))
    v_target_emb = np.tile(target_v_emb, (len(sub_v_emb), 1))
    uv_target_emb = np.vstack((u_target_emb, v_target_emb))
    return u, v, r, y, uv_emb, vu_emb, uv_target_emb, regression


def construct_pyg_features(u, v, r, y,
                           uv_emb, vu_emb, uv_target_emb, regression):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.FloatTensor(r)
    edge_attr = torch.cat([r, r])

    inter_index = torch.cat([torch.stack([u, v], dim=0), torch.stack([v, u], dim=0)], 1)
    inter_emb = torch.FloatTensor(np.concatenate((uv_emb, vu_emb), 1))

    u_target_index = torch.stack([u, torch.zeros_like(u)], dim=0)
    v_target_index = torch.stack([v, torch.zeros_like(v)], dim=0)
    uv_target_intra_index = torch.cat([u_target_index, v_target_index], 1)

    target_u_index = torch.stack([torch.zeros_like(u), u], dim=0)
    target_v_index = torch.stack([torch.zeros_like(v), v], dim=0)
    target_uv_intra_index = torch.cat([target_u_index, target_v_index], 1)

    uv_target_intra_emb = torch.FloatTensor(uv_target_emb)

    if regression:
        y = torch.FloatTensor([y])  # for regression
    else:
        y = torch.LongTensor([y])  # for classification
    # pytorch-geometric Data object 에서 x(=feature matrix) 를 지정해야 [num_nodes, num_node_features]
    # 로부터 num_nodes 를 추출할 수 있음
    data = Data(x=inter_emb, edge_index=inter_index, edge_attr=edge_attr, y=y,
                uv_target_index=uv_target_intra_index, uv_target_emb=uv_target_intra_emb,
                target_uv_index=target_uv_intra_index)
    # data.num_nodes = len(inter_index)
    return data


def construct_pyg_data(u, v, r, y, uv_emb, vu_emb, uv_target_emb, regression):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.FloatTensor(r)
    edge_attr = torch.cat([r, r], 0)

    # changed the edge index direction for j to i feature aggregation
    # however, this provision doesn't affect the performance...
    inter_index = torch.cat([torch.stack([v, u], dim=0), torch.stack([u, v], dim=0)], 1)
    inter_emb = torch.FloatTensor(np.concatenate((vu_emb, uv_emb), 1))

    u_target_index = torch.stack([u, torch.zeros_like(u)], dim=0)
    v_target_index = torch.stack([v, torch.zeros_like(v)], dim=0)
    uv_target_intra_index = torch.cat([u_target_index, v_target_index], 1)

    target_u_index = torch.stack([torch.zeros_like(u), u], dim=0)
    target_v_index = torch.stack([torch.zeros_like(v), v], dim=0)
    target_uv_intra_index = torch.cat([target_u_index, target_v_index], 1)

    uv_target_intra_emb = torch.FloatTensor(uv_target_emb)

    if regression:
        y = torch.FloatTensor([y])  # for regression
    else:
        y = torch.LongTensor([y])  # for classification

    data = Data(x=inter_emb, edge_index=inter_index, edge_attr=edge_attr, y=y,
                uv_target_index=uv_target_intra_index, uv_target_emb=uv_target_intra_emb,
                target_uv_index=target_uv_intra_index)
    return data


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    if not fringe:
        return set([])
    return set(A[list(fringe)].indices)

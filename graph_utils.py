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


class MyDynamicDataset(Dataset):
    def __init__(self, root, A, links, labels,
                 h, sample_ratio, max_nodes_per_hop,
                 regression, label_predict,
                 u_features, v_features, u_dict, v_dict, class_values, max_num=None):
        super(MyDynamicDataset, self).__init__(root)
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.regression = regression
        self.label_predict = label_predict
        self.u_features = u_features
        self.v_features = v_features
        self.u_dict = u_dict
        self.v_dict = v_dict
        self.class_values = class_values
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
        tmp = subgraph_embedding(
            (i, j), self.Arow, self.Acol,
            self.regression, self.label_predict,
            self.u_features, self.v_features,
            self.u_dict, self.v_dict,
            self.h, self.sample_ratio, self.max_nodes_per_hop, self.class_values, g_label
        )
        return construct_pyg_graph(*tmp)


def subgraph_embedding(ind, Arow, Acol,
                       regression, label_predict,
                       u_features, v_features, u_dict, v_dict,
                       h=1, sample_ratio=1.0, max_nodes_per_hop=None,
                       class_values=None, y=1):
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
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
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)

    subgraph = Arow[u_nodes][:, v_nodes]
    subgraph[0, 0] = 0

    # y 가 prediction target !
    u, v, r = ssp.find(subgraph)
    if label_predict:
        y = class_values[y]

    u_node_indices = np.array(u_nodes)[u]
    u_emb_indices = torch.LongTensor([u_dict[i] for i in u_node_indices])
    sub_u_emb = u_features(u_emb_indices)

    v_node_indices = np.array(v_nodes)[v]
    v_emb_indices = torch.LongTensor([v_dict[j] for j in v_node_indices])
    sub_v_emb = v_features(v_emb_indices)

    node_feature_0 = np.vstack((sub_u_emb, sub_v_emb))
    node_feature_1 = np.vstack((sub_v_emb, sub_u_emb))
    node_feature = np.concatenate((node_feature_0, node_feature_1), 1)
    return u, v, y, node_feature, regression


def construct_pyg_graph(u, v, y, node_features, regression):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    edge_index = torch.cat([torch.stack([u, v], dim=0), torch.stack([v, u], dim=0)], 1)
    x = torch.FloatTensor(node_features)
    if regression:
        y = torch.FloatTensor([y])  # for regression
    else:
        y = torch.LongTensor([y])  # for classification
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


class Dynamic_relative_Feature(Dataset):
    def __init__(self, root, A, links, labels,
                 h, sample_ratio, max_nodes_per_hop,
                 regression, label_predict,
                 u_features, v_features, u_dict, v_dict, class_values, max_num=None):
        super(Dynamic_relative_Feature, self).__init__(root)
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.regression = regression
        self.label_predict = label_predict
        self.u_features = u_features
        self.v_features = v_features
        self.u_dict = u_dict
        self.v_dict = v_dict
        self.class_values = class_values
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
        tmp = subgraph_rel_attention(
            (i, j), self.Arow, self.Acol,
            self.regression, self.label_predict,
            self.u_features, self.v_features,
            self.u_dict, self.v_dict,
            self.h, self.sample_ratio, self.max_nodes_per_hop, self.class_values, g_label
        )
        return relative_pyg_graph(*tmp)


def subgraph_rel_attention(ind, Arow, Acol,
                       regression, label_predict,
                       u_features, v_features, u_dict, v_dict,
                       h=1, sample_ratio=1.0, max_nodes_per_hop=None,
                       class_values=None, y=1):
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
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
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)

    subgraph = Arow[u_nodes][:, v_nodes]
    subgraph[0, 0] = 0

    u, v, r = ssp.find(subgraph)
    # v += len(u_nodes)
    # print('u : ', u)
    # print('v : ', v)
    # u :  [1 2 3 4 5 6 7]
    # v :  [0 0 0 0 0 0 0]
    if label_predict:
        y = class_values[y]

    u_node_indices = np.array(u_nodes)[u]
    u_emb_indices = torch.LongTensor([u_dict[i] for i in u_node_indices])
    sub_u_emb = u_features(u_emb_indices)

    v_node_indices = np.array(v_nodes)[v]
    v_emb_indices = torch.LongTensor([v_dict[j] for j in v_node_indices])
    sub_v_emb = v_features(v_emb_indices)

    node_feature_0 = np.vstack((sub_u_emb, sub_v_emb))
    # print('node_feature_0 : ', node_feature_0.shape)  # (_, 32)
    node_feature_1 = np.vstack((sub_u_emb, sub_v_emb))
    # print('node_feature_1 : ', node_feature_1.shape)  # (_, 32)
    node_feature = np.concatenate((node_feature_0, node_feature_1), 1)
    # print('node_feature shape : ', node_feature.shape)  # (_, 64)
    return u, v, y, node_feature, regression


def relative_pyg_graph(u, v, y, node_features, regression):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    edge_index = torch.cat([torch.stack([u, v]), torch.stack([v, u])], 1)
    x = torch.FloatTensor(node_features)
    if regression:
        y = torch.FloatTensor([y])  # for regression
    else:
        y = torch.LongTensor([y])  # for classification
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


class Dynamic_self_Feature(Dataset):
    def __init__(self, root, A, links, labels,
                 h, sample_ratio, max_nodes_per_hop,
                 regression, label_predict,
                 u_features, v_features, u_dict, v_dict, class_values, max_num=None):
        super(Dynamic_self_Feature, self).__init__(root)
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.regression = regression
        self.label_predict = label_predict
        self.u_features = u_features
        self.v_features = v_features
        self.u_dict = u_dict
        self.v_dict = v_dict
        self.class_values = class_values
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
        tmp = subgraph_self_attention(
            (i, j), self.Arow, self.Acol,
            self.regression, self.label_predict,
            self.u_features, self.v_features,
            self.u_dict, self.v_dict,
            self.h, self.sample_ratio, self.max_nodes_per_hop, self.class_values, g_label
        )
        return self_pyg_graph(*tmp)


def subgraph_self_attention(ind, Arow, Acol,
                       regression, label_predict,
                       u_features, v_features, u_dict, v_dict,
                       h=1, sample_ratio=1.0, max_nodes_per_hop=None,
                       class_values=None, y=1):
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
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
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)

    subgraph = Arow[u_nodes][:, v_nodes]
    subgraph[0, 0] = 0

    u, v, r = ssp.find(subgraph)
    # print('r : ', r)
    # print('y : ', y)
    # r :  [62. 62. 62. 62. 62. 62. 27. 27. 27. 27. 71. 62. 62. 62. 62. 62. 42. 27.
    #  14. 27. 42. 27. 62. 42. 62. 27. 14. 62. 62. 14. 62. 27. 42. 14. 62. 62.
    #  62. 62. 27. 62. 62. 42. 42. 42. 42. 27. 62. 62. 62. 62. 62. 14. 42. 62.
    #  71. 62. 62.]
    # y :  61
    # v += len(u_nodes)
    # print('u_nodes : ', u_nodes)
    # print('v_nodes : ', v_nodes)
    # u_nodes :  [297, 1733, 41, 105, 2742, 1977]
    # v_nodes :  [2506, 1092]
    # print('u : ', u)
    # print('v : ', v)
    # u :  [1 2 3 4 5 0]
    # v :  [0 0 0 0 0 1]

    # user list without target user ID(=0)
    u_array = np.array(u)
    u = np.delete(u_array, np.where(u_array == 0))
    # item list without target item ID(=0)
    v_array = np.array(v)
    v = np.delete(v_array, np.where(v_array == 0))

    if label_predict:
        y = class_values[y]

    u_node_indices = np.array(u_nodes)[u]
    u_emb_indices = torch.LongTensor([u_dict[i] for i in u_node_indices])
    sub_u_emb = u_features(u_emb_indices)
    # print('sub_u_emb : ', sub_u_emb.shape)
    target_u = u_features(torch.LongTensor([u_dict[u_nodes[0]]]))
    target_u_emb = target_u.expand_as(sub_u_emb)
    # print('target_u_emb : ', target_u_emb)
    # target_u_emb :  tensor([], size=(0, 32))
    #  neighbor 가 없는 node ...

    v_node_indices = np.array(v_nodes)[v]
    v_emb_indices = torch.LongTensor([v_dict[j] for j in v_node_indices])
    sub_v_emb = v_features(v_emb_indices)
    target_v_emb = v_features(torch.LongTensor([v_dict[v_nodes[0]]])).expand_as(sub_v_emb)
    # print('target_v_emb : ', target_v_emb)

    node_feature_0 = np.hstack((sub_u_emb, target_u_emb))
    # print('node_feature_0: ', node_feature_0.shape)  # [, 64]
    node_feature_1 = np.hstack((sub_v_emb, target_v_emb))
    node_feature = np.vstack((node_feature_0, node_feature_1))
    # print('node_feature : ', node_feature.shape)  # [, 64]
    return u, v, y, node_feature, regression


def self_pyg_graph(u, v, y, node_features, regression):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    edge_index = torch.cat(
        [torch.stack([u, torch.zeros_like(u)]),
         torch.stack([v, torch.zeros_like(v)])], 1)
    x = torch.FloatTensor(node_features)
    if regression:
        y = torch.FloatTensor([y])  # for regression
    else:
        y = torch.LongTensor([y])  # for classification
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    if not fringe:
        return set([])
    return set(A[list(fringe)].indices)


def PyGGraph_to_nx(data):
    edges = list(zip(data.edge_index[0, :].tolist(), data.edge_index[1, :].tolist()))
    g = nx.from_edgelist(edges)
    g.add_nodes_from(range(len(data.x)))  # in case some nodes are isolated
    # transform r back to rating label
    edge_types = {(u, v): data.edge_type[i].item() for i, (u, v) in enumerate(edges)}
    nx.set_edge_attributes(g, name='type', values=edge_types)
    node_types = dict(zip(range(data.num_nodes), torch.argmax(data.x, 1).tolist()))
    nx.set_node_attributes(g, name='type', values=node_types)
    g.graph['rating'] = data.y.item()
    return g

import os
import argparse
import networkx as nx
import scipy.sparse as sp
import numpy as np
import pandas as pd
import random
from pecanpy import pecanpy as node2vec


def map_data(data):
    uniq = list(set(data))
    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array([id_dict[x] for x in data])
    n = len(uniq)
    return data, id_dict, n


def make_edgelist(M):
    row, col = np.nonzero(M)
    print('unique row # : ', len(np.unique(row)))  # 6040
    print('unique col # : ', len(np.unique(col)))  # 3689
    col = col + 10000
    B = nx.Graph()
    B.add_nodes_from(row, bipartite=0)
    B.add_nodes_from(col, bipartite=1)
    bi_tuples = []
    for i in range(len(row)):
        bi_tuples.append((row[i], col[i]))
    B.add_edges_from(bi_tuples)
    top_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
    top_proj = nx.bipartite.projected_graph(B, top_nodes)
    bottom_nodes = B.nodes - top_nodes
    bottom_proj = nx.bipartite.projected_graph(B, bottom_nodes)
    return top_proj, bottom_proj


def make_weighted_edgelist(M):
    row, col = M.nonzero()
    print('unique row # : ', len(np.unique(row)))  # 1357
    print('unique col # : ', len(np.unique(col)))  # 1363
    col = col + 10000
    B = nx.Graph()
    B.add_nodes_from(row, bipartite=0)
    B.add_nodes_from(col, bipartite=1)
    bi_tuples = []
    for i in range(len(row)):
        bi_tuples.append((row[i], col[i]))
    B.add_edges_from(bi_tuples)
    top_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
    bottom_nodes = B.nodes - top_nodes
    top_proj = nx.bipartite.weighted_projected_graph(B, top_nodes)
    bottom_proj = nx.bipartite.weighted_projected_graph(B, bottom_nodes)
    return top_proj, bottom_proj


def make_ml_1m_train_mx(ml_dir):
    seed = 1234
    sep = r'\:\:'
    filename = os.path.join(ml_dir, 'ratings.dat')
    dtypes = {
        'u_nodes': np.int64, 'v_nodes': np.int64,
        'ratings': np.float32, 'timestamp': np.float64}
    data = pd.read_csv(filename, sep=sep, header=None,
                       names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'],
                       converters=dtypes, engine='python')
    data_array = data.values.tolist()
    random.seed(seed)
    random.shuffle(data_array)
    data_array = np.array(data_array)
    u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])
    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)
    u_nodes, v_nodes = u_nodes_ratings.astype(np.int64), \
                                       v_nodes_ratings.astype(np.int64)
    ratings = ratings.astype(np.float32)
    rating_dict = {r: i for i, r in enumerate(sorted(list(set(ratings))))}

    # number of test and validation edges
    num_test = int(np.ceil(ratings.shape[0] * 0.1))
    num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))
    num_train = ratings.shape[0] - num_val - num_test

    pairs_nonzero = np.vstack([u_nodes, v_nodes]).transpose()
    train_pairs_idx = pairs_nonzero[0:num_train]
    u_train_idx, v_train_idx = train_pairs_idx.transpose()
    # create labels
    all_labels = np.array([rating_dict[r] for r in ratings], dtype=np.int32)
    train_labels = all_labels[0:num_train]
    # make training adjacency matrix
    data = train_labels + 1.
    data = data.astype(np.float32)
    rating_mx_train = sp.csr_matrix((data, [u_train_idx, v_train_idx]),
                                    shape=[num_users, num_items], dtype=np.float32)
    return rating_mx_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PycanPy node embedding')
    # parser.add_argument('--dimensions', type=int, default=128,
    # parser.add_argument('--dimensions', type=int, default=64,
    parser.add_argument('--dimensions', type=int, default=32,
    # parser.add_argument('--dimensions', type=int, default=16,
                        help='number of dimensions. default is 128')
    # parser.add_argument('--walk-length', type=int, default=180,
    # parser.add_argument('--walk-length', type=int, default=160,
    # parser.add_argument('--walk-length', type=int, default=140,
    # parser.add_argument('--walk-length', type=int, default=120,
    # parser.add_argument('--walk-length', type=int, default=100,
    parser.add_argument('--walk-length', type=int, default=80,
    # parser.add_argument('--walk-length', type=int, default=60,
    # parser.add_argument('--walk-length', type=int, default=50,
    # parser.add_argument('--walk-length', type=int, default=40,
    # parser.add_argument('--walk-length', type=int, default=30,
    # parser.add_argument('--walk-length', type=int, default=25,
    # parser.add_argument('--walk-length', type=int, default=20,
    # parser.add_argument('--walk-length', type=int, default=15,
    # parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=10,
    # parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 10.')
    parser.add_argument('--window-size', type=int, default=10,
    # parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 10.')
    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)
    args = parser.parse_args()

    print('walk length : ', args.walk_length)
    print('num walks : ', args.num_walks)
    print('window size : ', args.window_size)
    print('embed dim. : ', args.dimensions)

    home = os.path.expanduser('~')
    data_dir = os.path.join(home, 'ALP')
    raw_data_dir = os.path.join(data_dir, 'raw_data')
    ml_1m_dir = os.path.join(raw_data_dir, 'ml_1m')
    M = make_ml_1m_train_mx(ml_1m_dir)

    ml_1m_edge_dir = os.path.join(ml_1m_dir, 'ml_1m_edge')
    if not os.path.exists(ml_1m_edge_dir):
        os.mkdir(ml_1m_edge_dir)
    # top_proj, bottom_proj = make_edgelist(M)
    # nx.write_edgelist(top_proj, os.path.join(ml_dir, 'top.edgelist'), data=False)
    # nx.write_edgelist(bottom_proj, os.path.join(ml_dir, 'btm.edgelist'), data=False)
    top_proj, bottom_proj = make_weighted_edgelist(M)
    nx.write_edgelist(top_proj, os.path.join(ml_1m_edge_dir, 'top_weighted.edgelist'), data=["weight"])
    nx.write_edgelist(bottom_proj, os.path.join(ml_1m_edge_dir, 'btm_weighted.edgelist'), data=["weight"])

    ml_1m_emb_dir = os.path.join(ml_1m_dir, 'ml_1m_emb')
    if not os.path.exists(ml_1m_emb_dir):
        os.makedirs(ml_1m_emb_dir)
    top_dir = os.path.join(ml_1m_edge_dir, 'top_weighted.edgelist')
    btm_dir = os.path.join(ml_1m_edge_dir, 'btm_weighted.edgelist')
    top_emb_dir = os.path.join(ml_1m_emb_dir, 'weighted_top_' +
                               'dim_' + str(args.dimensions) +
                               '_walklen_' + str(args.walk_length) +
                               '_num_walks_' + str(args.num_walks) +
                               '_window_' + str(args.window_size) +
                               '.wv')
    btm_emb_dir = os.path.join(ml_1m_emb_dir, 'weighted_btm_' +
                               'dim_' + str(args.dimensions) +
                               '_walklen_' + str(args.walk_length) +
                               '_num_walks_' + str(args.num_walks) +
                               '_window_' + str(args.window_size) +
                               '.wv')

    ml_g = node2vec.DenseOTF(p=args.p, q=args.q, workers=args.workers, verbose=True)
    # ml_g.read_edg(top_dir, weighted=False, directed=False, delimiter=" ")
    ml_g.read_edg(top_dir, weighted=True, directed=False, delimiter=" ")
    ml_g.preprocess_transition_probs()
    top_walks = ml_g.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length)
    top_wv = ml_g.embed(dim=args.dimensions,
                        num_walks=args.num_walks,
                        walk_length=args.walk_length,
                        window_size=args.window_size)  # changed the return value of 'embed()'
    top_wv.save(top_emb_dir)

    ml_g.read_edg(btm_dir, weighted=False, directed=False, delimiter=" ")
    ml_g.preprocess_transition_probs()
    btm_walks = ml_g.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length)
    btm_wv = ml_g.embed(dim=args.dimensions,
                        num_walks=args.num_walks,
                        walk_length=args.walk_length,
                        window_size=args.window_size)
    btm_wv.save(btm_emb_dir)
    print('embdding vector saved!')

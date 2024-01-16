import os
import argparse
import scipy.sparse as sp
import networkx as nx
import numpy as np
import pandas as pd
from pecanpy import pecanpy as node2vec


def map_data(data):
    uniq = list(set(data))
    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array([id_dict[x] for x in data])
    n = len(uniq)
    return data, id_dict, n


def make_edgelist(M):
    row, col = np.nonzero(M)
    print('unique row # : ', len(np.unique(row)))  # 943
    print('unique col # : ', len(np.unique(col)))  # 1614
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
    top_proj = nx.bipartite.projected_graph(B, top_nodes, multigraph=True)
    bottom_proj = nx.bipartite.projected_graph(B, bottom_nodes, multigraph=True)
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


def make_ml_100k_train_mx(ml_dir):
    sep = '\t'
    dtypes = {
        'u_nodes': np.int32, 'v_nodes': np.int32,
        'ratings': np.float32, 'timestamp': np.float64}
    filename_train = os.path.join(ml_dir, 'u1.base')
    filename_test = os.path.join(ml_dir, 'u1.test')

    data_train = pd.read_csv(
        filename_train, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)
    data_test = pd.read_csv(
        filename_test, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)
    data_array_train = data_train.values.tolist()
    data_array_train = np.array(data_array_train)
    data_array_test = data_test.values.tolist()
    data_array_test = np.array(data_array_test)

    data_array = np.concatenate([data_array_train, data_array_test], axis=0)

    u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])

    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)
    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)
    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1
    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(sorted(list(set(ratings))))}
    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(u_nodes)):
        assert (labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])
    labels = labels.reshape([-1])

    # number of test and validation edges, see cf-nade code
    num_train = data_array_train.shape[0]
    num_test = data_array_test.shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])
    for i in range(len(ratings)):
        assert (labels[idx_nonzero[i]] == rating_dict[ratings[i]])

    idx_nonzero_train = idx_nonzero[0:num_train + num_val]
    idx_nonzero_test = idx_nonzero[num_train + num_val:]
    # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)

    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]
    assert (len(test_idx) == num_test)

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))
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
    ml_100k_dir = os.path.join(raw_data_dir, 'ml_100k')
    M = make_ml_100k_train_mx(ml_100k_dir)

    ml_100k_edge_dir = os.path.join(ml_100k_dir, 'ml_100k_edge')
    if not os.path.exists(ml_100k_edge_dir):
        os.makedirs(ml_100k_edge_dir)
    # top_proj, bottom_proj = make_edgelist(M)
    # nx.write_edgelist(top_proj, os.path.join(ml_dir, 'top.edgelist'), data=False)
    # nx.write_edgelist(bottom_proj, os.path.join(ml_dir, 'btm.edgelist'), data=False)
    top_proj, bottom_proj = make_weighted_edgelist(M)
    nx.write_edgelist(top_proj, os.path.join(ml_100k_edge_dir, 'top_weighted.edgelist'), data=["weight"])
    nx.write_edgelist(bottom_proj, os.path.join(ml_100k_edge_dir, 'btm_weighted.edgelist'), data=["weight"])

    ml_100k_emb_dir = os.path.join(ml_100k_dir, 'ml_100k_emb')
    if not os.path.exists(ml_100k_emb_dir):
        os.makedirs(ml_100k_emb_dir)
    top_dir = os.path.join(ml_100k_edge_dir, 'top_weighted.edgelist')
    btm_dir = os.path.join(ml_100k_edge_dir, 'btm_weighted.edgelist')
    top_emb_dir = os.path.join(ml_100k_emb_dir, 'weighted_top_' +
                               'dim_' + str(args.dimensions) +
                               '_walklen_' + str(args.walk_length) +
                               '_num_walks_' + str(args.num_walks) +
                               '_window_' + str(args.window_size) +
                               '.wv')
    btm_emb_dir = os.path.join(ml_100k_emb_dir, 'weighted_btm_' +
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
    top_wv.save(top_emb_dir)  # 943

    ml_g.read_edg(btm_dir, weighted=False, directed=False, delimiter=" ")
    ml_g.preprocess_transition_probs()
    btm_walks = ml_g.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length)
    btm_wv = ml_g.embed(dim=args.dimensions,
                        num_walks=args.num_walks,
                        walk_length=args.walk_length,
                        window_size=args.window_size)
    btm_wv.save(btm_emb_dir)  # 1650
    print('embdding vector saved!')

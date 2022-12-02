import os
import argparse
import h5py
import scipy.sparse as sp
import networkx as nx
import numpy as np
from pecanpy import pecanpy as node2vec


def load_matlab_file(path_file, name_field):
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['raw_data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        out = np.asarray(ds).astype(np.float32).T
    db.close()
    return out


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


def weighted_sum(M, index):
    total_indices = []
    total_rating_weights = []
    zero_indices = []
    nonzero_indices = []
    for i in range(M.shape[index]):
        if M[i].sum() == 0:
            zero_indices.append(i)
            continue
        else:
            nonzero_indices.append(i)
            per_indices = []
            per_rating_weights = []
            for j in M[i].nonzero()[0]:
                per_indices.append(j)
                per_rating_weights.append(M[i][j])
            row_weights = np.array(per_rating_weights) / M[i].sum()
            total_indices.append(per_indices)
            total_rating_weights.append(row_weights)
    return total_indices, total_rating_weights, zero_indices, nonzero_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PycanPy node embedding')
    # parser.add_argument('--dimensions', type=int, default=128,
    # parser.add_argument('--dimensions', type=int, default=64,
    # parser.add_argument('--dimensions', type=int, default=32,
    parser.add_argument('--dimensions', type=int, default=16,
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
    data_dir = os.path.join(home, 'link_prediction/ALP/raw_data')

    yahoo_dir = os.path.join(data_dir, 'yahoo_music/training_test_dataset.mat')
    yahoo_M = load_matlab_file(yahoo_dir, 'M')
    top_proj, bottom_proj = make_weighted_edgelist(yahoo_M)
    nx.write_edgelist(top_proj, os.path.join(data_dir, 'yahoo_music/top_weighted.edgelist'), data=["weight"])
    nx.write_edgelist(bottom_proj, os.path.join(data_dir, 'yahoo_music/btm_weighted.edgelist'), data=["weight"])

    yahoo_emb_dir = os.path.join(data_dir, 'yahoo_music/emb')
    if not os.path.exists(yahoo_emb_dir):
        os.makedirs(yahoo_emb_dir)
    top_dir = os.path.join(data_dir, 'yahoo_music/top_weighted.edgelist')
    btm_dir = os.path.join(data_dir, 'yahoo_music/btm_weighted.edgelist')
    top_emb_dir = os.path.join(yahoo_emb_dir, 'weighted_top_' +
                               'dim_' + str(args.dimensions) +
                               '_walklen_' + str(args.walk_length) +
                               '_num_walks_' + str(args.num_walks) +
                               '_window_' + str(args.window_size) +
                               '.wv')
    btm_emb_dir = os.path.join(yahoo_emb_dir, 'weighted_btm_' +
                               'dim_' + str(args.dimensions) +
                               '_walklen_' + str(args.walk_length) +
                               '_num_walks_' + str(args.num_walks) +
                               '_window_' + str(args.window_size) +
                               '.wv')

    yahoo_g = node2vec.PreComp(p=args.p, q=args.q, workers=args.workers, verbose=True)
    yahoo_g.read_edg(top_dir, weighted=True, directed=False, delimiter=" ")
    yahoo_g.preprocess_transition_probs()
    top_walks = yahoo_g.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length)
    # embed() returns w2v.wv
    top_wv = yahoo_g.embed(dim=args.dimensions,
                           num_walks=args.num_walks,
                           walk_length=args.walk_length,
                           window_size=args.window_size)
    top_wv.save(top_emb_dir)

    yahoo_g.read_edg(btm_dir, weighted=False, directed=False, delimiter=" ")
    yahoo_g.preprocess_transition_probs()
    btm_walks = yahoo_g.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length)
    btm_wv = yahoo_g.embed(dim=args.dimensions,
                           num_walks=args.num_walks,
                           walk_length=args.walk_length,
                           window_size=args.window_size)
    btm_wv.save(btm_emb_dir)
    print('embdding vector saved!')

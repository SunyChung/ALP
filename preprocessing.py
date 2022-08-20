import numpy as np
import scipy.sparse as sp
import pickle as pkl
import os
import h5py
import random
import pandas as pd
from data_utils import map_data, download_dataset


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
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T
    db.close()
    return out


def load_ml_1m(dataset,
               u_wv_keys, v_wv_keys,
               testing=False, rating_map=None, post_rating_map=None, ratio=1.0):
    sep = r'\:\:'
    filename = 'raw_data/' + dataset + '/ratings.dat'
    dtypes = {
        'u_nodes': np.int64, 'v_nodes': np.int64,
        'ratings': np.float32, 'timestamp': np.float64}
    data = pd.read_csv(filename, sep=sep, header=None,
                       names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'],
                       converters=dtypes, engine='python')
    data_array = data.values.tolist()
    random.seed(1234)
    random.shuffle(data_array)
    data_array = np.array(data_array)
    u_nodes = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])
    u_nodes, u_dict, num_users = map_data(u_nodes)
    v_nodes, v_dict, num_items = map_data(v_nodes)
    u_nodes, v_nodes = u_nodes.astype(np.int64), v_nodes.astype(np.int64)
    ratings = ratings.astype(np.float32)

    if rating_map is not None:
        for i, x in enumerate(ratings):
            ratings[i] = rating_map[x]
    rating_dict = {r: i for i, r in enumerate(sorted(list(set(ratings))))}

    # number of test and validation edges
    num_test = int(np.ceil(ratings.shape[0] * 0.1))
    num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))
    num_train = ratings.shape[0] - num_val - num_test

    pairs_nonzero = np.vstack([u_nodes, v_nodes]).transpose()
    train_pairs_idx = pairs_nonzero[0:int(num_train * ratio)]
    val_pairs_idx = pairs_nonzero[num_train:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    train_pairs_re_idx = []
    for u, v in train_pairs_idx:
        if u in u_wv_keys and v in v_wv_keys:
            train_pairs_re_idx.append([u, v])
    train_pairs_idx = np.array(train_pairs_re_idx)
    idx_nonzero_train = np.array(
        [u * num_items + v for u, v in train_pairs_re_idx])

    val_pairs_re_idx = []
    for u, v in val_pairs_idx:
        if u in u_wv_keys and v in v_wv_keys:
            val_pairs_re_idx.append([u, v])
    val_pairs_idx = np.array(val_pairs_re_idx)
    idx_nonzero_val = np.array(
        [u * num_items + v for u, v in val_pairs_re_idx])

    test_pairs_re_idx = []
    for u, v in test_pairs_idx:
        if u in u_wv_keys and v in v_wv_keys:
            test_pairs_re_idx.append([u, v])
    test_pairs_idx = np.array(test_pairs_re_idx)
    idx_nonzero_test = np.array(
        [u * num_items + v for u, v in test_pairs_re_idx])

    u_train_idx, v_train_idx = train_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_test_idx, v_test_idx = test_pairs_idx.transpose()

    # create labels
    labels = np.full((num_users, num_items), -1, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])
    labels = labels.reshape([-1])
    train_labels = labels[idx_nonzero_train]
    val_labels = labels[idx_nonzero_val]
    test_labels = labels[idx_nonzero_test]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])

    class_values = np.sort(np.unique(ratings))
    # make training adjacency matrix
    if post_rating_map is None:
        data = train_labels + 1.
    else:
        data = np.array([post_rating_map[r] for r in class_values[train_labels]]) + 1.
    data = data.astype(np.float32)
    rating_mx_train = sp.csr_matrix((data, [u_train_idx, v_train_idx]),
                                    shape=[num_users, num_items], dtype=np.float32)
    return rating_mx_train, train_labels, u_train_idx, v_train_idx, \
           val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values


def load_ml_100k(dataset,
                 u_wv_keys, v_wv_keys,
                 testing=False, rating_map=None, post_rating_map=None, ratio=1.0):
    sep = '\t'
    # Check if files exist and download otherwise
    files = ['/u1.base', '/u1.test', '/u.item', '/u.user']
    fname = dataset
    data_dir = 'raw_data/' + fname
    download_dataset(fname, files, data_dir)
    dtypes = {
        'u_nodes': np.int32, 'v_nodes': np.int32,
        'ratings': np.float32, 'timestamp': np.float64}
    filename_train = 'raw_data/' + dataset + '/u1.base'
    filename_test = 'raw_data/' + dataset + '/u1.test'

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

    if ratio < 1.0:
        data_array_train \
            = data_array_train[data_array_train[:, -1].argsort()[:int(ratio * len(data_array_train))]]
    data_array = np.concatenate([data_array_train, data_array_test], axis=0)

    u_nodes = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])
    if rating_map is not None:
        for i, x in enumerate(ratings):
            ratings[i] = rating_map[x]
    u_nodes, u_dict, num_users = map_data(u_nodes)
    v_nodes, v_dict, num_items = map_data(v_nodes)
    u_nodes, v_nodes = u_nodes.astype(np.int64), v_nodes.astype(np.int32)
    ratings = ratings.astype(np.float64)

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(sorted(list(set(ratings))))}
    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1
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

    # idx_nonzero_train = idx_nonzero[0:num_train + num_val]
    # idx_nonzero_test = idx_nonzero[num_train + num_val:]
    pairs_nonzero_train = pairs_nonzero[0:num_train + num_val]
    pairs_nonzero_test = pairs_nonzero[num_train + num_val:]

    re_pair_nonzero_train = []
    for u, v in pairs_nonzero_train:
        if u in u_wv_keys and v in v_wv_keys:
            re_pair_nonzero_train.append([u, v])
    pairs_nonzero_train = np.array(re_pair_nonzero_train)
    idx_nonzero_train = np.array(
        [u * num_items + v for u, v in re_pair_nonzero_train])

    re_pair_nonzero_test = []
    for u, v in pairs_nonzero_test:
        if u in u_wv_keys and v in v_wv_keys:
            re_pair_nonzero_test.append([u, v])
    pairs_nonzero_test = np.array(re_pair_nonzero_test)
    idx_nonzero_test = np.array(
        [u * num_items + v for u, v in re_pair_nonzero_test])

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]
    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]
    # assert (len(test_idx) == num_test)
    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]
    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]
    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] \
            = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    return rating_mx_train, train_labels, u_train_idx, v_train_idx, \
           val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values


def load_data(dataset,
              u_wv_keys, v_wv_keys,
              testing=False, rating_map=None, post_rating_map=None,
              use_label=True):
    path_dataset = 'raw_data/' + dataset + '/training_test_dataset.mat'
    M = load_matlab_file(path_dataset, 'M')
    if rating_map is not None:
        M[np.where(M)] = [rating_map[x] for x in M[np.where(M)]]
    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest')
    num_users = M.shape[0]
    num_items = M.shape[1]
    print('Otraining num_train : ', np.where(Otraining)[0].shape[0])
    print('Otest num_test : ', np.where(Otest)[0].shape[0])
    u_nodes = np.where(M)[0].astype(np.int64)
    v_nodes = np.where(M)[1].astype(np.int64)
    ratings = M[np.where(M)].astype(np.float64)
    print('number of users = ', len(set(u_nodes)))
    print('number of item = ', len(set(v_nodes)))

    rating_dict = {r: i for i, r in enumerate(sorted(list(set(ratings))))}
    neutral_rating = -1  # 0 maybe?
    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    if use_label:
        labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])
    else:
        labels[u_nodes, v_nodes] = np.array([r for r in ratings])
    labels = labels.reshape([-1])

    pairs_nonzero_train = np.array(
        [[u, v] for u, v in zip(np.where(Otraining)[0], np.where(Otraining)[1])])
    re_pair_nonzero_train = []
    for u, v in pairs_nonzero_train:
        if u in u_wv_keys and v in v_wv_keys:
            re_pair_nonzero_train.append([u, v])
    pairs_nonzero_train = np.array(re_pair_nonzero_train)
    pairs_nonzero_test = np.array(
        [[u, v] for u, v in zip(np.where(Otest)[0], np.where(Otest)[1])])
    re_pair_nonzero_test = []
    for u, v in pairs_nonzero_test:
        if u in u_wv_keys and v in v_wv_keys:
            re_pair_nonzero_test.append([u, v])
    pairs_nonzero_test = np.array(re_pair_nonzero_test)

    idx_nonzero_train = np.array(
        [u * num_items + v for u, v in re_pair_nonzero_train])
    idx_nonzero_test = np.array(
        [u * num_items + v for u, v in re_pair_nonzero_test])

    num_train = len(pairs_nonzero_train)
    num_test = len(pairs_nonzero_test)
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val
    print('pairs nonzero num_train : ', num_train)
    print('pairs nonzero num_test : ', num_test)
    print('pairs nonzero num_val : ', num_val)

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]
    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)
    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]
    # assert (len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]
    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()
    # user the original rating values
    train_ratings = labels[train_idx]
    val_ratings = labels[val_idx]
    test_ratings = labels[test_idx]
    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_ratings = np.hstack([train_ratings, val_ratings])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    '''Note here rating matrix elements' values + 1 !!!'''
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] \
            = np.array(
            [post_rating_map[r] for r in class_values[ratings[train_idx]]]
        ) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))
    return rating_mx_train, train_ratings, u_train_idx, v_train_idx, \
           val_ratings, u_val_idx, v_val_idx, test_ratings, u_test_idx, v_test_idx, \
           class_values, rating_dict

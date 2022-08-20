import os.path
from shutil import copy, rmtree, copytree

from graph_utils import MyDynamicDataset, Dynamic_relative_Feature, Dynamic_self_Feature
from preprocessing import *
from train_eval import *
from model_32 import *
from model_self_att import GT_self_attn

from gensim.models import KeyedVectors

# 왜 random seed 이렇게 많이 필요한지??
torch.manual_seed(seed=1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed=1)
random.seed(1)
np.random.seed(1)

# data_name = 'yahoo_music'
data_name = 'flixster'
# data_name = 'douban'
# data_name = 'ml_100k'
# data_name = 'ml_1m'

print('Using ' + data_name + ' dataset ...')
home = os.path.expanduser('~')
data_root = os.path.join(home, 'link_prediction/raw_data/')
data_dir = data_root + data_name

print('node embedding loading from ', data_name)
dimensions = 32
# walk_length = 80
walk_length = 20
# window_size = 10
window_size = 5

# top_wv = np.load(os.path.join(data_dir,
top_wv = KeyedVectors.load(os.path.join(data_dir,
                                        'emb/top_' +
                                        'dim_' + str(dimensions) +
                                        '_walklen_' + str(walk_length) +
                                        '_window_' + str(window_size) +
                                        '.wv'))
# '.npy'))
# btm_wv = np.load(os.path.join(data_dir,
btm_wv = KeyedVectors.load(os.path.join(data_dir,
                                        'emb/btm_' +
                                        'dim_' + str(dimensions) +
                                        '_walklen_' + str(walk_length) +
                                        '_window_' + str(window_size) +
                                        '.wv'))
# '.npy'))
# top_emb = nn.Embedding.from_pretrained(torch.FloatTensor(top_wv))
# btm_emb = nn.Embedding.from_pretrained(torch.FloatTensor(btm_wv))
top_emb = nn.Embedding.from_pretrained(torch.FloatTensor(top_wv.vectors))
btm_emb = nn.Embedding.from_pretrained(torch.FloatTensor(btm_wv.vectors))

u_emb, v_emb = top_emb, btm_emb
u_dict, v_dict = top_wv.key_to_index, btm_wv.key_to_index
re_u_dict = {}
for k, v in u_dict.items():
    re_u_dict[int(k)] = v
u_dict = re_u_dict
re_v_dict = {}
for k, v in v_dict.items():
    re_v_dict[int(k) - 10000] = v
u_wv_keys = np.array([int(i) for i in u_dict.keys()])
v_wv_keys = np.array(list(re_v_dict.keys()))

if data_name in ['yahoo_music', 'flixster', 'douban']:
    (
        adj_train,
        train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices,
        test_labels, test_u_indices, test_v_indices,
        class_values, rating_dict
    ) = load_data(data_name,
                  u_wv_keys, v_wv_keys)

elif data_name == 'ml_100k':
    print("Using official MovieLens split u1.base/u1.test with 20% validation...")
    (
        adj_train,
        train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices,
        test_labels, test_u_indices, test_v_indices, class_values
    ) = load_ml_100k(
        data_name,
        u_wv_keys, v_wv_keys
    )
else:
    (
        adj_train,
        train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices,
        test_labels, test_u_indices, test_v_indices, class_values
    ) = load_ml_1m(
        data_name,
        u_wv_keys, v_wv_keys,
    )
print(class_values)  # sorted unique ratings values
# flixster : [0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]
print('len(class_values) : ', len(class_values))

train_indices = (train_u_indices, train_v_indices)
val_indices = (val_u_indices, val_v_indices)
test_indices = (test_u_indices, test_v_indices)
print('#train: %d, #val: %d, #test: %d' % (
    len(train_u_indices),
    len(val_u_indices),
    len(test_u_indices),
))

train_graphs, val_graphs, test_graphs = None, None, None
data_combo = (data_name)
reprocess = True
if reprocess:
    # if reprocess=True, delete the previously cached raw_data and reprocess.
    if os.path.isdir('raw_data/{}/train'.format(*data_combo)):
        rmtree('raw_data/{}/train'.format(*data_combo))
    if os.path.isdir('raw_data/{}/val'.format(*data_combo)):
        rmtree('raw_data/{}/val'.format(*data_combo))
    if os.path.isdir('raw_data/{}/test'.format(*data_combo)):
        rmtree('raw_data/{}/test'.format(*data_combo))

dataset_class = 'MyDynamicDataset'
# dataset_class = 'Dynamic_relative_Feature'
# dataset_class = 'Dynamic_self_Feature'
# dataset_class = 'MyDataset'
hop = 1
sample_ratio = 1.0
max_nodes_per_hop = 50
regression = True
label_predict = False
max_train_num = None

train_graphs = eval(dataset_class)(
    'raw_data/{}/train'.format(*data_combo),
    adj_train,
    train_indices,
    train_labels,
    hop,
    sample_ratio,
    max_nodes_per_hop,
    regression,
    label_predict,
    u_emb, v_emb,
    u_dict, re_v_dict,
    class_values,
    max_num=max_train_num,
)
test_graphs = eval(dataset_class)(
    'raw_data/{}/test'.format(*data_combo),
    adj_train,
    test_indices,
    test_labels,
    hop,
    sample_ratio,
    max_nodes_per_hop,
    regression,
    label_predict,
    u_emb, v_emb,
    u_dict, re_v_dict,
    class_values,
    max_num=max_train_num,
)

testing = False
if not testing:
    val_graphs = eval(dataset_class)(
        'raw_data/{}/val'.format(*data_combo),
        adj_train,
        val_indices,
        val_labels,
        hop,
        sample_ratio,
        max_nodes_per_hop,
        regression,
        label_predict,
        u_emb, v_emb,
        u_dict, re_v_dict,
        class_values,
        max_num=max_train_num,
    )
    test_graphs = val_graphs

# model object 생성할 떄는
# train_graph 넘길 필요없음
# train_multiple_epochs() 에서 넘기면
# data_loader 가 feeding 시킴
# model = GT_self_attn(
#     num_classes=len(class_values),
#     regression=regression,
# )
model = GT32dim(
    num_classes=len(class_values),
    regression=regression,
)
total_params = sum(p.numel() for param in model.parameters() for p in param)
print(f'Total number of parameters is {total_params}')
print('model : ', model)

train_epoch = 80
batch = 50

train_multiple_epochs(
    train_graphs,
    test_graphs,
    model,
    epochs=train_epoch,
    batch_size=batch,
    lr=1e-3,
    lr_decay_factor=0.1,
    lr_decay_step_size=50,
    weight_decay=0,
    regression=regression,
)

from shutil import rmtree
from gensim.models import KeyedVectors

from graph_utils import DynamicFeatures
from preprocessing import *
from train_eval import train_multiple_epochs

from model_2_modes_3_layers import *


torch.manual_seed(seed=1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed=1)
random.seed(1)
np.random.seed(1)

# data_name = 'yahoo'
# data_name = 'flixster'
# data_name = 'douban'
# data_name = 'ml_100k'
data_name = 'ml_1m'
# max_nodes_per_hop = 500 도 안 되면 어쩌라는 건지 -_
# 결국 max_nodes_per_hop = 100 으로 ;;

print('Using ' + data_name + ' dataset ...')
home = os.path.expanduser('~')
data_root = os.path.join(home, 'ALP/raw_data/')
data_dir = data_root + data_name

# max_nodes_per_hop = 100
# max_nodes_per_hop = 500
# max_nodes_per_hop = 1000
max_nodes_per_hop = 10000
# max_nodes_per_hop = None
print('max_nodes_per_hop : ', max_nodes_per_hop)

print('node embedding loading from ', data_name)
walk_length = 80
num_walks = 10
window_size = 10
dimensions = 32
# dimensions = 16
emb_dir = os.path.join(data_dir, data_name + '_emb')

top_wv = KeyedVectors.load(os.path.join(emb_dir,
                                           'weighted_top_dim_' + str(dimensions) +
                                           '_walklen_80_num_walks_10_window_10.wv'))
btm_wv = KeyedVectors.load(os.path.join(emb_dir,
                                           'weighted_btm_dim_' + str(dimensions) +
                                           '_walklen_80_num_walks_10_window_10.wv'))

top_emb = nn.Embedding.from_pretrained(torch.FloatTensor(top_wv.vectors))
btm_emb = nn.Embedding.from_pretrained(torch.FloatTensor(btm_wv.vectors))

# 근데, 왜 평균값을 더했었나... 여기서 avt_expand 더해서 layer 에서 차원 안 맞았음..
# top_avg = torch.mean(top_emb.weight, dim=0)
# btm_avg = torch.mean(btm_emb.weight, dim=0)
# top_avg_expand = torch.tile(top_avg, (top_emb.weight.shape[0], 1))
# btm_avg_expand = torch.tile(btm_avg, (btm_emb.weight.shape[0], 1))
# top_concat = torch.concat((top_avg_expand, top_emb.weight), 1)
# btm_concat = torch.concat((btm_avg_expand, btm_emb.weight), 1)
# u_emb, v_emb = top_concat, btm_concat

u_emb, v_emb = top_emb.weight, btm_emb.weight
u_dict, v_dict = top_wv.key_to_index, btm_wv.key_to_index
re_u_dict = {}
for k, v in u_dict.items():
    re_u_dict[int(k)] = v
u_dict = re_u_dict
re_v_dict = {}
for k, v in v_dict.items():
    re_v_dict[int(k) - 10000] = v
v_dict = re_v_dict
u_wv_keys = np.array([int(i) for i in u_dict.keys()])
v_wv_keys = np.array(list(re_v_dict.keys()))

use_rating_dict = True
# use rating_dict only for classification since it needs continuous class range
print('\nusing re-mapped rating dictionary? ', use_rating_dict)

if data_name in ['yahoo', 'flixster', 'douban']:
    (
        adj_train,
        train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices,
        test_labels, test_u_indices, test_v_indices,
        class_values
    ) = load_data(data_dir,
                  u_wv_keys, v_wv_keys,
                  use_rating_dict)
                # use_rating_dict=False)
                # rating_map=rating_map)

elif data_name == 'ml_100k':
    print("Using official MovieLens split u1.base/u1.test with 20% validation...")
    (
        adj_train,
        train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices,
        test_labels, test_u_indices, test_v_indices, class_values
    ) = load_ml_100k(
        data_dir,
        u_wv_keys, v_wv_keys,
        use_rating_dict
    )
else:
    (
        adj_train,
        train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices,
        test_labels, test_u_indices, test_v_indices, class_values
    ) = load_ml_1m(
        data_dir,
        u_wv_keys, v_wv_keys,
        use_rating_dict
    )

print('original target value', class_values)  # sorted unique (original) ratings values
# flixster : [0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]
print('# of target values : ', len(class_values))

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

dataset_class = 'DynamicFeatures'
print('using data class : ', dataset_class)

hop = 1
sample_ratio = 1.0
regression = True
# regression = False
print('regression ? ', regression)

max_train_num = None
# use_edge_feature = False
edge_feature = True
print('use rating as the edge weight ? ', edge_feature)

train_graphs = eval(dataset_class)(
    'raw_data/{}/train'.format(*data_combo),
    adj_train,
    train_indices,
    train_labels,
    hop,
    sample_ratio,
    max_nodes_per_hop,
    regression,
    u_emb, v_emb,
    u_dict, re_v_dict,
    class_values,
    emb_dim=dimensions,
    use_edge_feature=edge_feature,
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
    u_emb, v_emb,
    u_dict, re_v_dict,
    class_values,
    emb_dim=dimensions,
    use_edge_feature=edge_feature,
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
        u_emb, v_emb,
        u_dict, re_v_dict,
        class_values,
        emb_dim=dimensions,
        use_edge_feature=edge_feature,
        max_num=max_train_num,
    )
    test_graphs = val_graphs

# model = GT32dim_2_modes_dim_1_concat(
#     top_avg.to('cuda'),
#     btm_avg.to('cuda'),
#     num_classes=len(class_values),
#     regression=regression,
#     num_heads=1,
#     dropout=True,
# )

# model = GT32dim_2_modes_4_layers(
#     num_classes=len(class_values),
#     regression=regression,
#     num_heads=1,
#     dropout=True,
# )

model = GT32dim_2_modes_3_layers(
    num_classes=len(class_values),
    regression=regression,
    num_heads=1,
    dropout=True,
)

total_params = sum(p.numel() for param in model.parameters() for p in param)
print(f'\nTotal number of parameters is {total_params}')
print('model : ', model)

# train_epoch = 80
train_epoch = 100
# MUST FIX the batch size RuntimeError !!
# batch = 10
batch = 50
# batch = 60
# batch = 100
print('batch size : ', batch)

train_multiple_epochs(
    train_graphs,
    test_graphs,
    model,
    epochs=train_epoch,
    batch_size=batch,
    lr=1e-3,
    lr_decay_factor=0.1,
    lr_decay_step_size=50,
    # lr_decay_step_size=20,
    weight_decay=0,
    regression=regression,
)

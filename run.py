import os.path
from shutil import copy, rmtree, copytree
from gensim.models import KeyedVectors

from graph_utils import MyDynamicDataset, DynamicFeatures
from graph_util_with_attention import AttentionDataset
from preprocessing import *
from train_eval import *

from model_16_inter_intra_3_layers import GT16dim_3_modes
from model_16_inter_intra_3_layers_4_heads import GT16dim_3_modes_4_heads
from model_16_inter_intra_3_layers_8_heads import GT16dim_3_modes_8_heads
from model_16_inter_intra_3_layers_16_heads import GT16dim_3_modes_16_heads
from model_32 import *
from model_64 import GT64dim
from model_32_inter_intra_2_layers import GT32dim_2_modes
from model_32_inter_intra_3_layers import GT32dim_3_modes
from model_32_inter_intra_3_layers_4_heads import GT32dim_3_modes_4_heads
from model_32_inter_intra_3_layers_8_heads import GT32dim_3_modes_8_heads
from model_32_inter_intra_3_layers_16_heads import GT32dim_3_modes_16_heads
from model_32_inter_intra_3_layers_2_linear import GT32dim_3_modes_2_linear
from model_32_inter_intra_3_layers_3_dec_linear import GT32dim_3_modes_3_dec_linear
from model_32_inter_intra_3_layers_4_dec_linear import GT32dim_3_modes_4_dec_linear
from model_32_inter_intra_3_layers_5_dec_linear import GT32dim_3_modes_5_dec_linear
from model_32_inter_intra_3_layers_6_dec_linear import GT32dim_3_modes_6_dec_linear
from model_32_inter_intra_3_layers_7_dec_linear import GT32dim_3_modes_7_dec_linear
from model_32_inter_intra_4_layers_6_dec_linear import GT32dim_4_modes_6_dec_linear
from model_32_inter_intra_3_layers_new_attention import GT32dim_3_modes_new_attention
from model_32_inter_intra_3_layers_new_6_linear import GT32dim_3_modes_new_6_linear
from model_64_inter_intra_3_layers import GT64dim_3_modes
from model_64_inter_intra_3_layers_4_heads import GT64dim_3_modes_4_heads
from model_64_inter_intra_3_layers_8_heads import GT64dim_3_modes_8_heads
from model_64_inter_intra_3_layers_16_heads import GT64dim_3_modes_16_heads
from model_128_inter_intra_3_layers import GT128dim_3_modes
from model_128_inter_intra_3_layers_4_heads import GT128dim_3_modes_4_heads
from model_128_inter_intra_3_layers_8_heads import GT128dim_3_modes_8_heads
from model_128_inter_intra_3_layers_16_heads import GT128dim_3_modes_16_heads

torch.manual_seed(seed=1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed=1)
random.seed(1)
np.random.seed(1)

# data_name = 'yahoo_music'
# data_name = 'flixster'
# data_name = 'douban'
data_name = 'ml_100k'
# data_name = 'ml_1m'

print('Using ' + data_name + ' dataset ...')
home = os.path.expanduser('~')
data_root = os.path.join(home, 'link_prediction/ALP/raw_data/')
data_dir = data_root + data_name

print('node embedding loading from ', data_name)
# dimensions = 16
dimensions = 32
# dimensions = 64
# dimensions = 128
# walk_length = 100
walk_length = 80
# walk_length = 20
window_size = 10
# window_size = 5
# window_size = 20
print('walk length : ', walk_length)
print('window size : ', window_size)
print('embed dim. : ', dimensions)
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
v_dict = re_v_dict
u_wv_keys = np.array([int(i) for i in u_dict.keys()])
v_wv_keys = np.array(list(re_v_dict.keys()))

# when using rating_map for Flixster !
# rating_map = {x: 2x for x in np.arange(0.5, 5.01, 0.5)}
# rating_map = {x: math.ceil(int(x)) for x in np.arange(0.5, 5.01, 0.5)}
# print('using rating_map : ', rating_map)

# use_rating_dict = False
use_rating_dict = True
# use rating_dict only for classification since it needs continuous class range
print('\nusing re-mapped rating dictionary? ', use_rating_dict)

if data_name in ['yahoo_music', 'flixster', 'douban']:
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

# dataset_class = 'MyDynamicDataset'
dataset_class = 'DynamicFeatures'
# dataset_class = 'MyDataset'
# dataset_class = 'AttentionDataset'
print('using data class : ', dataset_class)

hop = 1
sample_ratio = 1.0
# max_nodes_per_hop = 50
max_nodes_per_hop = 10000
# max_nodes_per_hop = 1000
# max_nodes_per_hop = 100
print('max_nodes_per_hop : ', max_nodes_per_hop)

regression = True
# regression = False
print('regression ? ', regression)

max_train_num = None
use_edge_feature = True
print('use rating as the edge weight ? ', use_edge_feature)

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
    use_edge_feature,
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
    use_edge_feature,
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
        use_edge_feature,
        max_num=max_train_num,
    )
    test_graphs = val_graphs

# model object 생성할 떄는
# train_graph 넘길 필요없음
# train_multiple_epochs() 에서 넘기면
# data_loader 가 feeding 시킴

# model = GT16dim(
#     num_classes=len(class_values),
#     regression=regression,
#     use_edge_feature=True,
# )

# model = GT16dim_3_modes(
#     num_classes=len(class_values),
#     regression=regression,
# )

# model = GT16dim_3_modes_4_heads(
#     num_classes=len(class_values),
#     regression=regression,
# )

# model = GT16dim_3_modes_8_heads(
#     num_classes=len(class_values),
#     regression=regression,
# )

# model = GT16dim_3_modes_16_heads(
#     num_classes=len(class_values),
#     regression=regression,
# )

# model = GT32dim(
#     num_classes=len(class_values),
#     regression=regression,
#     use_edge_feature=False,
# )

# model = GT32dim_2_modes(
#     num_classes=len(class_values),
#     regression=regression,
# )

# model = GT32dim_3_modes(
#     num_classes=len(class_values),
#     regression=regression,
# )

# model = GT32dim_3_modes_4_heads(
#         num_classes=len(class_values),
#         regression=regression,
# )

# model = GT32dim_3_modes_8_heads(
#         num_classes=len(class_values),
#         regression=regression,
# )

# model = GT32dim_3_modes_16_heads(
#     num_classes=len(class_values),
#     regression=regression,
# )

# model = GT32dim_3_modes_2_linear(
#     num_classes=len(class_values),
#     regression=regression,
#     num_heads=1,
#     # num_heads=4,
#     # num_heads=8,
#     # num_heads=16,
#     # dropout=True,
#     dropout=False,
# )

# model = GT32dim_3_modes_3_dec_linear(
#     num_classes=len(class_values),
#     regression=regression,
#     # num_heads=1,
#     # num_heads=4,
#     # num_heads=8,
#     num_heads=16,
#     dropout=True,
#     # dropout=False,
# )

# model = GT32dim_3_modes_4_dec_linear(
#     num_classes=len(class_values),
#     regression=regression,
#     num_heads=1,
#     # num_heads=4,
#     # num_heads=8,
#     # num_heads=16,
#     # dropout=True,
#     dropout=False,
# )

# model = GT32dim_3_modes_5_dec_linear(
#     num_classes=len(class_values),
#     regression=regression,
#     num_heads=1,
#     # num_heads=4,
#     # num_heads=8,
#     # num_heads=16,
#     # dropout=True,
#     dropout=False,
# )

# model = GT32dim_3_modes_6_dec_linear(
#     num_classes=len(class_values),
#     regression=regression,
#     # num_heads=1,
#     # num_heads=4,
#     num_heads=8,
#     # num_heads=16,
#     # dropout=True,
#     dropout=False,
# )

# model = GT32dim_3_modes_7_dec_linear(
#     num_classes=len(class_values),
#     regression=regression,
#     # num_heads=1,
#     # num_heads=4,
#     # num_heads=8,
#     num_heads=16,
#     dropout=True,
#     # dropout=False,
# )

# model = GT32dim_4_modes_6_dec_linear(
#     num_classes=len(class_values),
#     regression=regression,
#     # num_heads=1,
#     # num_heads=4,
#     num_heads=8,
#     # num_heads=16,
#     # dropout=True,
#     dropout=False,
# )

# moved global_mean_pool in the last stage,
# not in after the self.lin1() before
model = GT32dim_3_modes_new_attention(
    num_classes=len(class_values),
    regression=regression,
    # num_heads=1,
    # num_heads=4,
    num_heads=8,
    # num_heads=16,
    dropout=True,
    # dropout=False,
)
#
# model = GT32dim_3_modes_new_6_linear(
#     num_classes=len(class_values),
#     regression=regression,
#     # num_heads=1,
#     # num_heads=4,
#     num_heads=8,
#     # num_heads=16,
#     dropout=True,
#     # dropout=False,
# )

# model = GT64dim_3_modes(
#         num_classes=len(class_values),
#         regression=regression,
# )

# model = GT64dim_3_modes_4_heads(
#         num_classes=len(class_values),
#         regression=regression,
# )

# model = GT64dim_3_modes_8_heads(
#         num_classes=len(class_values),
#         regression=regression,
# )

# model = GT64dim_3_modes_16_heads(
#         num_classes=len(class_values),
#         regression=regression,
# )

# model = GT32dim_2_linear(
#     num_classes=len(class_values),
#     regression=regression,
#     use_edge_feature=True,
# )

# model = GT64dim(
#     num_classes=len(class_values),
#     regression=regression,
#     use_edge_feature=True,
# )

# model = GT128dim_3_modes(
#     num_classes=len(class_values),
#     regression=regression,
# )

# model = GT128dim_3_modes_4_heads(
#     num_classes=len(class_values),
#     regression=regression,
# )

# model = GT128dim_3_modes_8_heads(
#     num_classes=len(class_values),
#     regression=regression,
# )

# model = GT128dim_3_modes_16_heads(
#     num_classes=len(class_values),
#     regression=regression,
# )

total_params = sum(p.numel() for param in model.parameters() for p in param)
print(f'\nTotal number of parameters is {total_params}')
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

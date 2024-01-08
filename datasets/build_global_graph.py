import pickle
from tqdm import tqdm
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str , default='yoochoose1_64', help='')
parser.add_argument('--epsilon', type=int, default=2, help='')
opt = parser.parse_args()

if opt.dataset == 'diginetica':
    n_node = 43097
elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
    n_node = 37483
elif opt.dataset == 'RetailRocket':
    n_node = 36968
elif opt.dataset == 'Tmall' :
    n_node = 40727
elif opt.dataset == 'Nowplaying':
    n_node = 60416
else:
    n_node = 310

train = pickle.load(open(f'./{opt.dataset}/raw/train.txt', 'rb'))

if not os.path.exists(f'./{opt.dataset}/raw/unique_nodes.txt'):
    # unique item in train
    unique_nodes = []
    for seq in tqdm(train[0], leave=False):
        for node in seq:
            if node not in unique_nodes:
                unique_nodes.append(node)
    pickle.dump(unique_nodes, open(f'./{opt.dataset}/raw/unique_nodes.txt', 'wb'))
else:
    unique_nodes = pickle.load(open(f'./{opt.dataset}/raw/unique_nodes.txt', 'rb'))

# 参考GCE-GNN https://github.com/johnny12150/GCE-GNN/blob/master/datasets/build_global_graph.py
graph_node = {k: [] for k in unique_nodes}
for i, seq in tqdm(enumerate(train[0])):
    if len(seq) > 0:
        for j, node in enumerate(seq):
            # 有考慮self-loop
            if j+1+opt.epsilon < len(seq)-1:
                graph_node[node]+=seq[j+1:j+opt.epsilon]
            else:
                graph_node[node]+=seq[j:len(seq)]

# 构造节点的neighbor序列以及对应weight
weight = {k: [] for k in unique_nodes}
for key in graph_node:
    cache = np.array(graph_node[key])
    cache = np.sort(cache)
    seq = []
    cache_node = -1
    for i, node in enumerate(cache):
        if cache_node != node:
            cache_node = node
            seq.append(node)
            weight[key].append(1)
        else:
            weight[key][-1] += 1
    
    graph_node[key] = seq

# 补全部分不在train序列中的item编号
for i in range(1, n_node + 2):
    if i not in graph_node:
        graph_node[i] = []
        weight[i] = []

print(f'Number of Session Graphs in {opt.dataset}', len(train[0]))
pickle.dump((graph_node, weight), open(f'./{opt.dataset}/processed/global_graph.dataset', 'wb'))




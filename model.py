import math
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from torch import nn
from torch.nn import Module
from torch_geometric.utils import to_dense_adj

class GCE_GNN(Module):
    def __init__(self, n_node, embed_dim=100, dataset='yoochoose1_64', dropout_global=0.5, n_neighbor=12, alpha=0.2) -> None:
        super(GCE_GNN, self).__init__()

        self.hidden_size = embed_dim
        self.dropout_global = dropout_global
        self.n_neighbor = n_neighbor
        self.embedding_layer = nn.Embedding(num_embeddings=n_node, embedding_dim=embed_dim)
        self.pos_embeding_layer = nn.Embedding(200, self.hidden_size)
        self.local_gnn = Session_Aggregator(embed_dim=embed_dim, alpha=alpha)
        self.globle_gnn = Globle_Aggregator(embed_dim=embed_dim)
        self.global_graph = pickle.load(open(f'./datasets/{dataset}/processed/global_graph.dataset', 'rb'))

        self.w1 = nn.Linear(self.hidden_size + 1, self.hidden_size + 1)
        self.w3 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.q1 = nn.Linear(self.hidden_size + 1, 1, bias=False)
        self.q2 = nn.Linear(self.hidden_size, 1, bias=False)
        self.w_4 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_5 = nn.Linear(self.hidden_size, self.hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def select_neighbors(self, neighbors, w):
        neighbors = torch.tensor(neighbors, dtype=torch.int)
        w = torch.tensor(w, dtype=torch.float)
        if w.shape[0] > self.n_neighbor:
            topk = w.topk(self.n_neighbor, dim=0)
            neighbors = neighbors[topk[1]]
            w = topk[0]
        return [neighbors.unsqueeze(1).reshape(-1, 1), w.unsqueeze(1).reshape(-1, 1)]

    def pai(self, s, h, w, sess_batch, item_batch):
        # 矩阵形式并行计算pai值
        a = self.q1(F.leaky_relu(self.w1(torch.cat((s * h, w), dim=1)))).view(-1, 1)
        a_spilt = torch.split(a, sess_batch, dim=0)
        a_spilt = tuple(torch.split(sess, item_batch[i], dim=0) for i, sess in enumerate(a_spilt))
        a = []

        # 对每个item的neighbor进行softmax操作
        for sess in a_spilt:
            for nodes in sess:
                a.append(torch.softmax(nodes.view(-1, 1), dim=0))
                
        return torch.cat(a, dim=0)
    
    def conpute_scores(self, hidden, batch):
        pos_emb = self.pos_embeding_layer.weight[:len(batch)]
        pos_emb = tuple(session_pos.view(1, -1).repeat(batch[i], 1) for i, session_pos in enumerate(pos_emb))
        pos_emb = torch.cat(pos_emb, dim=0)

        v_i = torch.split(hidden, batch, dim=0)

        z_i = torch.tanh(self.w3(torch.cat((hidden, pos_emb), dim=1)))

        s_ = tuple(torch.mean(nodes.float(), dim=0, keepdim=True).view(1, -1).repeat(nodes.shape[0], 1) for nodes in v_i)
        s_ = torch.cat(s_, dim=0)

        beta = self.q2(torch.sigmoid(self.w_4(z_i) + self.w_5(s_))).view(-1, 1)

        S = beta * hidden
        S = torch.split(S, batch, dim=0)
        S = tuple(torch.sum(nodes, dim=0, keepdim=True).view(1, -1) for nodes in S)
        S = torch.cat(S, dim=0)

        scores = torch.mm(S, self.embedding_layer.weight.transpose(1, 0))
        return scores

    def forward(self, data):
        x, edge_index, batch, seqs = data.x - 1, data.edge_index, data.batch, data.seq

        hidden = self.embedding_layer(x).squeeze()
        sections = torch.bincount(batch.cpu())
        v_i = torch.split(x, tuple(sections.cpu().numpy()))
        batch = list(int(nodes.shape[0]) for nodes in v_i)

        # globle
        # get neighbors and neighbors weigth
        item_neighbors = []
        neighbors_weight = []
        neighbor_batch = []
        item_neighbor_batch = []
        for seq in seqs:
            seq = np.unique(np.array(seq))
            
            # 取n_neighbor个top weight的邻居
            n_w = tuple([self.global_graph[0][node], self.global_graph[1][node]] for node in seq)
            n_w = tuple(self.select_neighbors(li[0], li[1]) for li in n_w)
            neighbors = tuple(li[0] for li in n_w)
            weight = tuple(li[1] for li in n_w)

            # 将邻居嵌入为向量表征
            neighbors = self.embedding_layer(torch.cat(neighbors, dim=0) - 1).squeeze().view(-1, self.hidden_size)
            item_neighbors.append(neighbors)

            weight = tuple(torch.softmax(w, dim=0) for w in weight)
            item_neighbor_batch.append(list(int(w.shape[0]) for w in weight))
            weight = torch.cat(weight, dim=0)
            neighbors_weight.append(weight)
            neighbor_batch.append(weight.shape[0])

        s_i_mean = tuple(torch.mean(nodes.float(), dim=0, keepdim=True) for nodes in v_i)

        # 通过矩阵连接并行计算多个session的hidden_neighbor
        s_mean = torch.cat(tuple(s.view(1, -1).repeat(neighbor_batch[i], 1) for i, s in enumerate(s_i_mean)), dim=0)
        del s_i_mean
        item_neighbors = torch.cat(item_neighbors, dim=0)
        neighbors_weight = torch.cat(neighbors_weight, dim=0)

        # 计算相似度
        h_n = self.pai(s_mean, item_neighbors, neighbors_weight, neighbor_batch, item_neighbor_batch) * item_neighbors
        del s_mean, item_neighbors, neighbors_weight
        h_n = torch.split(h_n, neighbor_batch, dim=0)
        h_n = tuple(torch.split(sess, item_neighbor_batch[i], dim=0) for i, sess in enumerate(h_n))
        h_neighbor = []

        # 对item的neighbor逐个sum
        for sess in h_n:
            for nodes in sess:
                h_neighbor.append(torch.sum(nodes, dim=0).view(1, -1))

        h_n = torch.cat(h_neighbor, dim=0)
        del h_neighbor
        h_global = self.globle_gnn(hidden, h_n)

        # local
        h_local = self.local_gnn(hidden, edge_index, batch)

        # combine
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)
        hidden = h_local + h_global
        
        return self.conpute_scores(hidden, batch)

class Globle_Aggregator(Module):
    def __init__(self, embed_dim = 100) -> None:
        super(Globle_Aggregator, self).__init__()
        self.hidden_size = embed_dim

        self.w2 = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, hidden, h_n):
        return F.relu(self.w2(torch.cat((hidden, h_n), dim=1)))

class Session_Aggregator(Module):
    def __init__(self, embed_dim=100, alpha=0.2) -> None:
        super(Session_Aggregator, self).__init__()
        self.hidden_size = embed_dim
        self.alpha = alpha

        self.a = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, hidden, edge_index, batch):
        # 对出度与入度边没有赋予初始权重
        adj_matrix = to_dense_adj(edge_index).squeeze()
        neighbors = tuple(torch.unique(torch.cat((adj_matrix[node].nonzero().squeeze(1), adj_matrix[:,node].nonzero().squeeze(1)), dim=0)) for node in range(hidden.shape[0]))
        item_batch = tuple(n.shape[0] for n in neighbors)
        neighbors = tuple(hidden[n] for n in neighbors)
        hidden = tuple(hidden[node_index].unsqueeze(1).view(1, -1).repeat(item_batch[node_index], 1) for node_index in range(hidden.shape[0]))
        neighbors = torch.cat(neighbors, dim=0)
        hidden = torch.cat(hidden, dim=0)
        assert hidden.shape == neighbors.shape
        alpha = F.leaky_relu(self.a(neighbors * hidden), self.alpha)
        assert alpha.shape[1] == 1
        alpha = torch.split(alpha, item_batch, dim=0)
        alpha = tuple(torch.softmax(node, dim=0) for node in alpha)
        alpha = torch.cat(alpha, dim=0)
        assert alpha.shape[0] == sum(item_batch)
        assert alpha.shape[1] == 1
        assert alpha.shape[0] == neighbors.shape[0]
        hidden = torch.split(alpha * neighbors, item_batch, dim=0)
        hidden = torch.cat(tuple(torch.sum(neighbor, dim=0).view(1, -1) for neighbor in hidden), dim=0)
        assert hidden.shape[1] == self.hidden_size
        assert hidden.shape[0] == sum(batch)
        return hidden



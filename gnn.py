import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import dgl
import matplotlib.pyplot as plt
from itertools import chain
from time import time

from dgl.nn.pytorch import SAGEConv

G = nx.Graph()

for i in range(30):
    G.add_node(i)   
                  


elist = [(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18),
(1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 29), (2, 3), (2, 4), (2, 5),
(2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17), (2, 18), (3, 4), (3, 5), (3, 6),
(3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (3, 16), (3, 17), (3, 18), (4, 5), (4, 6), (4, 7),
(4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (4, 17), (4, 18), (5, 6), (5, 7), (5, 8), (5, 9),
(5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12),
(6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16),
(7, 17), (7, 18), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (9, 10), (9, 11), (9, 12),
(9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18),
(11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (12, 13), (12, 14), (12, 15), (12, 16), (12, 17), (12, 18),
(13, 14), (13, 15), (13, 16), (13, 17), (13, 18), (14, 15), (14, 16), (14, 17), (14, 18), (15, 16), (15, 17), (15, 18), (16, 17),
(16, 18), (17, 18), (18, 19), (18, 20), (18, 21), (18, 22), (18, 23), (18, 24), (18, 25), (18, 26), (18, 27), (18, 28), (18, 29),
(18, 29), (19, 20), (19, 21), (19, 22), (19, 23), (19, 24), (19, 25), (19, 26), (19, 27), (19, 28), (19, 29), (19, 29), (20, 21),
(20, 22), (20, 23), (20, 24), (20, 25), (20, 26), (20, 27), (20, 28), (20, 29), (20, 29), (21, 22), (21, 23), (21, 24), (21, 25),
(21, 26), (21, 27), (21, 28), (21, 29), (21, 29), (22, 23), (22, 24), (22, 25), (22, 26), (22, 27), (22, 28), (22, 29), (22, 29),
(23, 24), (23, 25), (23, 26), (23, 27), (23, 28), (23, 29), (23, 29), (24, 25), (24, 26), (24, 27), (24, 28), (24, 29), (24, 29),
(25, 26), (25, 27), (25, 28), (25, 29), (25, 29), (26, 27), (26, 28), (26, 29), (26, 29), (27, 28), (27, 29), (27, 29), (28, 29),
(28, 29)]


TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
G.add_edges_from(elist) 
dgl_graph = dgl.from_networkx(G)
dgl_graph = dgl_graph.to(TORCH_DEVICE)
layout = nx.kamada_kawai_layout(G)

nx.draw(G, with_labels=True, pos=layout, node_color='gray', edge_color='k')
plt.show() 

def get_adjacency_matrix(nx_graph, torch_device, torch_dtype):

    adj = nx.linalg.graphmatrix.adjacency_matrix(nx_graph).todense()
    adj_ = torch.tensor(adj).type(torch_dtype).to(torch_device)
    
    return adj_


class GNNSage(nn.Module): 
  
    def __init__(self, g, in_feats, hidden_size, num_classes, dropout, agg_type='mean'):


        super(GNNSage, self).__init__()

        self.g = g
        self.num_classes = num_classes

        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, hidden_size, agg_type, activation=F.relu)) 
                                                                                        
        self.layers.append(SAGEConv(hidden_size, num_classes, agg_type)) 
        self.dropout = nn.Dropout(p=dropout) 

    def forward(self, features): 

        h = features  
        for i, layer in enumerate(self.layers):
            if i != 0: 
                h = self.dropout(h) 
            h = layer(self.g, h) 

        return h

def get_gnn(g, n_nodes, opt_params, torch_device, torch_dtype, SEED_VALUE):

    try:
        set_seed(SEED_VALUE)
    except KeyError:
        set_seed(0)

    net = GNNSage(g, 50, 60, 18, 0.4, 'mean') 


    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, 50) 
    embed = embed.type(torch_dtype).to(torch_device)

    params = chain(net.parameters(), embed.parameters()) 

    optimizer = torch.optim.AdamW(params, **opt_params, weight_decay=1e-2)

    return net, embed, optimizer



def loss_func_mod(probs, adj_tensor):

    loss_ = torch.mul(adj_tensor, (probs @ probs.T)).sum() / 2 

    return loss_

def loss_func_color_hard(coloring, nx_graph): 

    cost_ = 0
    for (u, v) in nx_graph.edges:
        cost_ += 1*(coloring[u] == coloring[v])

    return cost_




def run_gnn_training(nx_graph, graph_dgl, adj_mat, net, embed, optimizer,
                     number_epochs=int(1e5), patience=500, tolerance=1e-4, seed=1):


    set_seed(seed)

    inputs = embed.weight
    best_cost = torch.tensor(float('Inf'))
    best_loss = torch.tensor(float('Inf'))
    best_coloring = None

    prev_loss = 1.
    cnt = 0
    losses = []

    for epoch in range(number_epochs):

        logits = net(inputs)

        probs = F.softmax(logits, dim=1)

        loss = loss_func_mod(probs, adj_mat)
        losses.append(loss.item())

        coloring = torch.argmax(probs, dim=1)
        cost_hard = loss_func_color_hard(coloring, nx_graph)

        if cost_hard < best_cost:
            best_loss = loss
            best_cost = cost_hard
            best_coloring = coloring

        """Early stopping"""

        if (abs(loss - prev_loss) <= tolerance) | ((loss - prev_loss) > 0):
            cnt += 1
        else:
            cnt = 0

        prev_loss = loss

        if cnt >= patience:
            print(f'Stopping early on epoch {epoch}. Patience count: {cnt}')
            print(cost_hard)
            break


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if epoch % 1000 == 0:
            print('Epoch %d | Soft Loss: %.5f' % (epoch, loss.item()))
            print('Epoch %d | Discrete Cost: %d' % (epoch, cost_hard.item()))


    print('Epoch %d | Final loss: %.5f' % (epoch, loss.item()))
    print('Epoch %d | Lowest discrete cost: %d' % (epoch, best_cost))
    plt.plot(losses)
    plt.show()

    final_loss = loss
    final_coloring = torch.argmax(probs, 1)
    print(f'Final coloring: {final_coloring}, soft loss: {final_loss}')
    print(best_coloring)

    return probs, best_coloring, best_loss, final_coloring, final_loss, epoch



t_start = time()

hypers = {
        'learning_rate': 0.003
    }

opt_hypers = {
    'lr': hypers.get('learning_rate', None)
}

net, embed, optimizer = get_gnn(dgl_graph, 30, opt_hypers, TORCH_DEVICE, TORCH_DTYPE, SEED_VALUE)
adj_ = get_adjacency_matrix(G, TORCH_DEVICE, TORCH_DTYPE)
probs, best_coloring, best_loss, final_coloring, final_loss, epoch_num = run_gnn_training(G, dgl_graph, adj_, net, embed, optimizer, int(5e4), 500, 1e-3, seed=SEED_VALUE)
runtime_gnn = round(time() - t_start, 4)

print(f'GNN runtime: {runtime_gnn}s')


best_cost_hard = loss_func_color_hard(best_coloring, G)

print(f'Best (hard) cost of coloring (n_class={17}): {best_cost_hard}')


color_dict = {0:'red', 1:'blue', 2:'yellow', 3:'lightblue', 4:'orange', 5:'green', 6:'lightgreen', 7:'moccasin', 8:'turquoise', 9:'navy', 10:'lightcoral', 11:'limegreen', 12:'orchid', 13:'mediumpurple', 14:'salmon', 15:'white', 16:'aquamarine', 17:'lavender'}
color_map = np.vectorize(color_dict.get)(best_coloring.cpu())
layout = nx.kamada_kawai_layout(G)
nx.draw(G, with_labels=True, node_color=color_map, pos=layout)

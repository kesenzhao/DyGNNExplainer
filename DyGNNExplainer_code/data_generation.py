import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import shutil
import random
import pandas as pd
from tqdm import tqdm
import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt
import argparse
import scipy.sparse as sp
from tensorboardX import SummaryWriter

import causaleffect
from gae.model import VGAE3MLP
from gae.optimizer import loss_function as gae_loss

sys.path.append('gnnexp')
import models
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils

dataset = 'syn1'
ckpt = torch.load('ckpt.tar'%(dataset))
cg_dict = ckpt["cg"]


import random
adj = cg_dict["adj"][0]
node_num = adj.shape[0]
label = cg_dict["label"][0]

ind = []
for i in range(len(label)):
    if label[i] != 0:
        ind.append(i)
import random
random.shuffle(ind)

feats = cg_dict["feat"][0]
feats = torch.tensor(feats)
hist_ndFeats_list = []
for i in range(5):
    hist_ndFeats_list.append(feats)


roof_cls1 = node_roof[:30]
roof_cls2 = node_roof[30:60]
roof_cls3 = node_roof[60:80]



adj_flag = adj.copy()
timestamp = [[0] * adj.shape[0] for _ in range(adj.shape[0])]

#process 1 from roof to foot
dynamic_label = label.copy()
for i in roof_cls1:
    # print(i)
    dynamic_label[i] = 1
    edge_seq = adj_flag[i,:]
    # print(edge_seq)
    # de
    edge_seq = np.where(edge_seq==1)
    # print(label[edge_seq])
    for j in edge_seq[0]:
        if label[j] == 0:
            timestamp[i][j] = 1
            timestamp[j][i] = 1
        elif label[j] == 1:
            dynamic_label[j] = 1
            timestamp[i][j] = 2
            timestamp[j][i] = 2
            edge_seq_j = adj_flag[j,:]
            edge_seq_j = np.where(edge_seq_j==1)
            for k in edge_seq_j[0]:
                if label[k] == 2:
                    dynamic_label[k] = 1
                    timestamp[j][k] = 3
                    timestamp[k][j] = 3
                elif label[k] != 3:
                    timestamp[j][k] = 4
                    timestamp[k][j] = 4


# process 2 from foot to roof

for i in roof_cls2:
    dynamic_label[i] = 2
    edge_seq = adj_flag[i,:]
    # print(edge_seq)
    # de
    edge_seq = np.where(edge_seq==1)
    # print(label[edge_seq])
    for j in edge_seq[0]:
        if label[j] == 0:
            timestamp[i][j] = 4
            timestamp[j][i] = 4
        elif label[j] == 1:
            dynamic_label[j] = 2
            timestamp[i][j] = 3
            timestamp[j][i] = 3
            edge_seq_j = adj_flag[j,:]
            edge_seq_j = np.where(edge_seq_j==1)
            for k in edge_seq_j[0]:
                if label[k] == 2:
                    dynamic_label[k] = 2
                    timestamp[j][k] = 2
                    timestamp[k][j] = 2
                elif label[k] != 3:
                    timestamp[j][k] = 1
                    timestamp[k][j] = 1

# process 3 from eaves to roof and foot
# dynamic_label = label.copy()
for i in roof_cls3:
    dynamic_label[i] = 3
    edge_seq = adj_flag[i,:]
    # print(edge_seq)
    # de
    edge_seq = np.where(edge_seq==1)
    # print(label[edge_seq])
    for j in edge_seq[0]:
        if label[j] == 0:
            timestamp[i][j] = 2
            timestamp[j][i] = 2
        elif label[j] == 1:
            dynamic_label[j] = 3
            timestamp[i][j] = 1
            timestamp[j][i] = 1
            edge_seq_j = adj_flag[j,:]
            edge_seq_j = np.where(edge_seq_j==1)
            for k in edge_seq_j[0]:
                if label[k] == 2:
                    dynamic_label[k] = 3
                    timestamp[j][k] = 3
                    timestamp[k][j] = 3
                elif label[k] != 3:
                    timestamp[j][k] = 4
                    timestamp[k][j] = 4

hist_adj_list = []
hist_adj = torch.zeros([adj.shape[0],adj.shape[1]])
# snapshot number 5
for j in range(adj.shape[0]):
    for k in range(adj.shape[0]):
        if adj[j][k] == 1 and timestamp[j][k] == 0:
            hist_adj[j][k] = 1
idx = torch.nonzero(hist_adj).T  
data = hist_adj[idx[0],idx[1]]
coo_a = torch.sparse_coo_tensor(idx, data,hist_adj.shape)
hist_adj_list.append(coo_a)

for i in range(4):
    for j in range(adj.shape[0]):
        for k in range(adj.shape[0]):
            if adj[j][k] == 1:
                if timestamp[j][k] == (i+1):
                    hist_adj[j][k] = 1
    print(hist_adj.sum())
    idx = torch.nonzero(hist_adj).T  
    data = hist_adj[idx[0],idx[1]]
    coo_a = torch.sparse_coo_tensor(idx, data,hist_adj.shape)
    # hist_adj = torch.coo
    hist_adj_list.append(coo_a)


hist_mask_list = []
num_nodes = 700
for i in range(5):
    mask = torch.zeros(num_nodes) - float("Inf")
    # hist_adj_list[i]
    non_zero = hist_adj_list[i]._indices().unique()
    mask[non_zero] = 0
    hist_mask_list.append(mask)


import random
random.shuffle(ind)
label_sp = {}
label_sp['idx'] = torch.tensor(ind[:350])
label_sp['vals'] = torch.tensor(dynamic_label[ind[:350]]).type(torch.long)
# label_sp['vals'] = label_sp['vals'].type(torch.long)
label_sp_test = {}
label_sp_test['idx'] = torch.tensor(ind[350:])
label_sp_test['vals'] = torch.tensor(dynamic_label[ind[350:]]).type(torch.long)
spl_train = {'hist_adj_list': hist_adj_list,
				'hist_ndFeats_list': hist_ndFeats_list,
				'label_sp': label_sp,
				'node_mask_list': hist_mask_list}

spl_test = {'hist_adj_list': hist_adj_list,
				'hist_ndFeats_list': hist_ndFeats_list,
				'label_sp': label_sp_test,
				'node_mask_list': hist_mask_list}
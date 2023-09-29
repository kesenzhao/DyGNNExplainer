import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math


class EGCN(torch.nn.Module):
    def __init__(self, args, activation, device='cpu', skipfeats=False):
        super().__init__()
        GRCU_args = u.Namespace({})

        # feats = [args.feats_per_node,
        #          args.layer_1_feats,
        #          args.layer_2_feats]
        feats = [10,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1,len(feats)):
            GRCU_args = u.Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation})

            grcu_i = GRCU(GRCU_args)
            #print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list,nodes_mask_list):
        node_feats= Nodes_list[-1]

        for unit in self.GRCU_layers:
            Nodes_list,mask_adj_list,update_list = unit(A_list,Nodes_list,nodes_mask_list)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
        return out,mask_adj_list,update_list


class GRCU(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)
        #mask
        self.sample_gcn = Parameter(torch.randn(7,30,30))

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def norm(self, Ahat):
        Ahat = Ahat.to_dense() + torch.eye(Ahat.to_dense().shape[0]).to(Ahat.device)
        di = torch.sum(Ahat, dim=0)
        dj = torch.sum(Ahat, dim=1)
        Ahat = Ahat.mul(dj.unsqueeze(1).matmul(di.unsqueeze(0)) ** -0.5)
        return Ahat

    def forward(self,A_list,node_embs_list,mask_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        mask_adj_list = []
        update_list = []
        for t,Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            #first evolve the weights from the initial and use the new weights with the node_embs
            
            GCN_weights, update = self.evolve_weights(GCN_weights,node_embs,mask_list[t])
            mask_adj = torch.sigmoid(self.sample_gcn[t])
            Ahat_m = self.norm(Ahat).to_dense().mul(mask_adj)
            # print(Ahat_m.sum())
            # print(Ahat)
            # print(mask_adj)
            # de
            # Ahat_m = Ahat
            # print(Ahat_m)
            # de
            # print(Ahat_m[:10,:10])
            node_embs = self.activation(Ahat_m.matmul(node_embs.matmul(GCN_weights)))
            
            out_seq.append(node_embs)
            mask_adj_list.append(mask_adj)
            update_list.append(update)
        # print(mask_adj_list[1][:10,:10])
        return out_seq,mask_adj_list,update_list

class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        
        self.choose_topk = TopK(feats = args.rows,
                                k = args.cols)
        # self.choose_topk = TopK(feats = 700,
        #                         k = args.cols)

    def forward(self,prev_Q,prev_Z,mask):
        z_topk = self.choose_topk(prev_Z,mask)

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q, update

        

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self,feats,k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats,1))
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self,node_embs,mask):
        # print(node_embs)
        # print(node_embs.matmul(self.scorer))
        # print(self.scorer.norm())
        # de
        # node_embs = node_embs.float()
        # print(node_embs.type())
        # print(self.scorer.type())
        # print(node_embs.matmul(self.scorer.float()))
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        # mask = torch.unsqueeze(mask,1)

        scores = scores + mask
        
        self.k = min(self.k, scores.shape[0])

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]
        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices,self.k)
            
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()
        # print(node_embs[topk_indices].shape)
        # print(tanh(scores[topk_indices]).shape)
        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1,1))

        #we need to transpose the output
        return out.t()

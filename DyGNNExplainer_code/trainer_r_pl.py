import torch
import utils as u
import logger
import time
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch_geometric

class Trainer():
    def __init__(self,args, splitter, gcn, classifier, gcn2, classifier2, comp_loss, dataset, num_classes):
        self.args = args
        self.splitter = splitter
        self.tasker = splitter.tasker
        self.gcn = gcn
        self.classifier = classifier
        self.gcn2 = gcn2
        self.classifier2 = classifier2
        self.comp_loss = comp_loss

        self.num_nodes = dataset.num_nodes
        self.data = dataset
        self.dataset = []
        self.num_classes = num_classes

        self.logger = logger.Logger(args, self.num_classes)

        self.init_optimizers(args)
        self.best_pl = 0
        self.best = 0
        if self.tasker.is_static:
            adj_matrix = u.sparse_prepare_tensor(self.tasker.adj_matrix, torch_size = [self.num_nodes], ignore_batch_dim = False)
            self.hist_adj_list = [adj_matrix]
            self.hist_ndFeats_list = [self.tasker.nodes_feats.float()]

    def init_optimizers(self,args):
        params = self.gcn.parameters()
        self.gcn_opt = torch.optim.Adam(params, lr = args.learning_rate)
        params = self.classifier.parameters()
        self.classifier_opt = torch.optim.Adam(params, lr = args.learning_rate)
        self.gcn_opt.zero_grad()
        self.classifier_opt.zero_grad()

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    def load_checkpoint(self, filename, model):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']
            self.gcn.load_state_dict(checkpoint['gcn_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_dict'])
            self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
            self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
            self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
            return epoch
        else:
            self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
            return 0

    def extract_neighborhood(self, node_idx, spl):
        """Returns the neighborhood of a given node."""
        n_hops = 3
        hist_adj_list = spl['hist_adj_list'].copy()
        hist_ndFeats_list = spl['hist_ndFeats_list'].copy()
        label_sp = spl['label_sp'].copy()
        node_mask_list = spl['node_mask_list'].copy()
        for i in range(len(hist_adj_list)):
            hist_adj_list[i] = hist_adj_list[i].to(self.args.device)
            hist_ndFeats_list[i] = hist_ndFeats_list[i].float().to(self.args.device)
            node_mask_list[i] = node_mask_list[i].unsqueeze(1).to(self.args.device)
        label_sp['idx'] =  label_sp['idx'].unsqueeze(0).to(self.args.device)
        label_sp['vals'] = label_sp['vals'].to(self.args.device) 

        # 前五个时间片逐步搭房子，接着两个时间片，房子中的一些边消失。
        edge_index = hist_adj_list[-1]._indices()
        num_nodes = 100
        n_hops = 1
        map_pre = []
        node_idx_pre = []
        while n_hops:
            mapping, edge_idxs, node_idx_new, edge_mask = torch_geometric.utils.k_hop_subgraph(int(node_idx), n_hops, edge_index, relabel_nodes=True)
            
            if len(mapping)> num_nodes:
                break
            else:
                map_pre = mapping.cpu().numpy().tolist()
                node_idx_pre = node_idx_new
            n_hops +=1
        map_pad = set(mapping.cpu().numpy().tolist()).difference(set(map_pre))
        pad_len = num_nodes - len(map_pre)
        import random
        map_pad = list(map_pad)
        random.shuffle(map_pad)
        mapping = map_pre + map_pad[:pad_len]
        


        # 节点的新id
        node_idx_new = node_idx_pre.item()
        mapping = torch.tensor(mapping, dtype=torch.long).to(self.args.device)
        # 新的邻接矩阵list
        sub_hist_adj_list = []
        # for i in range(len(hist_adj_list)):
        #     adj = hist_adj_list[i].to_dense()
        #     sub_adj = adj[mapping,:][:,mapping]
        #     # dense转coo
        #     idx = torch.nonzero(sub_adj).T  # 这里需要转置一下
        #     data = sub_adj[idx[0],idx[1]]
        #     coo_a = torch.sparse_coo_tensor(idx, data,sub_adj.shape).to(self.args.device)
        #     # hist_adj = torch.coo
        #     sub_hist_adj_list.append(coo_a)
        for i in range(len(hist_adj_list)):
            indices, values = torch_geometric.utils.subgraph(subset=mapping,edge_index=edge_index,relabel_nodes=True)
            values = torch.ones(indices.shape[1]).to(self.args.device)
            coo_a = torch.sparse_coo_tensor(indices, values)
            sub_hist_adj_list.append(coo_a)
        # 新的特征矩阵list
        sub_hist_ndFeats_list = []
        for i in range(len(hist_adj_list)):
            feats = hist_ndFeats_list[i]
            sub_feats = feats[mapping]
            sub_hist_ndFeats_list.append(sub_feats)

        # 新的node_mask_list
        # 图中出现的节点
        sub_hist_mask_list = []
        # num_nodes = len(mapping)
        for i in range(len(hist_adj_list)):
            mask = torch.zeros(num_nodes) - float("Inf")
            non_zero = sub_hist_adj_list[i]._indices().unique()
            mask[non_zero] = 0

            sub_hist_mask_list.append(mask.unsqueeze(1).to(self.args.device))

        # node_indices
        node_indices = torch.tensor(list(range(num_nodes)), dtype=torch.long).unsqueeze(0).to(self.args.device)

        # 新的label
        index = label_sp['idx'].cpu().numpy()
        y_index=np.argwhere(index==node_idx)
        label = label_sp['vals'][y_index[0][1]]

        # sub_adj = torch_geometric.utils.to_dense_adj(edge_idxs)[0]
        # # print(tg_G.edge_index)
        # # print(tg_G.edge_index.type)
        # # print(tg_G.edge_index.shape)
        # # de
        # adj_norm = preprocess_graph(sub_adj)
        # adj_label = sub_adj + np.eye(sub_adj.shape[0])
        # pos_weight = float(sub_adj.shape[0] * sub_adj.shape[0] - sub_adj.sum()) / sub_adj.sum()
        # pos_weight = torch.from_numpy(np.array(pos_weight))
        # norm = torch.tensor(sub_adj.shape[0] * sub_adj.shape[0] / float((sub_adj.shape[0] * sub_adj.shape[0] - sub_adj.sum()) * 2))
        # # Calculate hop_feat:
        # pow_adj = ((sub_adj @ sub_adj >=1).float() - np.eye(sub_adj.shape[0]) - sub_adj >=1).float()
        # feat = features[mapping]
        # sub_feat = feat
        # one_hot = torch.zeros((sub_adj.shape[0], ), dtype=torch.float)
        # one_hot[node_idx_new] = 1
        # hop_feat = [one_hot, sub_adj[node_idx_new], pow_adj[node_idx_new]]
        # if args.n_hops == 3:
        #     pow3_adj = ((pow_adj @ pow_adj >=1).float() - np.eye(pow_adj.shape[0]) - pow_adj >=1).float()
        #     hop_feat += [pow3_adj[node_idx_new]]
        #     hop_feat = torch.stack(hop_feat).t()
        #     sub_feat = torch.cat((sub_feat, hop_feat), dim=1)
        # if args.graph_labelling:
        #     G = graph_labeling(nx.from_numpy_array(sub_adj.numpy()))
        #     graph_label = np.array([G.nodes[node]['string'] for node in G])
        #     graph_label_onehot = label_onehot[graph_label]
        #     sub_feat = torch.cat((sub_feat, graph_label_onehot), dim=1)
        # sub_label = torch.from_numpy(label[mapping])
        return {
            "node_idx_new": node_idx_new,
            'hist_adj_list':sub_hist_adj_list,
            'hist_ndFeats_list': sub_hist_ndFeats_list,
            'hist_mask_list': sub_hist_mask_list,
            "mapping": mapping,
            'node_indices': node_indices,
            'label': label
            # 'sub_label': label_sp['idx'].unsqueeze(0)
            # "feat": feat.unsqueeze(0).to(device),
            # "sub_adj": sub_adj.unsqueeze(0).to(device),
            # "sub_feat": sub_feat.unsqueeze(0).to(device),
            # "adj_norm": adj_norm.unsqueeze(0).to(device),
            # "sub_label": sub_label.to(device),
            # "mapping": mapping.to(device),
            # "adj_label": adj_label.unsqueeze(0).to(device),
            # "graph_size": mapping.shape[-1],
            # "pos_weight": pos_weight.unsqueeze(0).to(device),
            # "norm": norm.unsqueeze(0).to(device)
        }

    def train(self):
        # for s in self.splitter.train:           
        #     if self.tasker.is_static:
        #         s = self.prepare_static_sample(s)
        #     else:
        #         s = self.prepare_sample(s)
        #     break
        # hist_adj_list = s.hist_adj_list
        # hist_ndFeats_list = s.hist_ndFeats_list
        # hist_mask_list = s.node_mask_list
        # label_sp = {}
        # label_sp['idx'] = s.label_sp['idx']
        # print(label_sp['idx'].shape)
        # de
        # label_sp['vals']= s.label_sp['vals']
        # label_sp_test = {}
        # label_sp_test['idx'] = s.label_sp['idx']
        # label_sp_test['vals']= s.label_sp['vals']

        # spl_train = {'hist_adj_list': hist_adj_list,
		# 		'hist_ndFeats_list': hist_ndFeats_list,
		# 		'label_sp': label_sp,
		# 		'node_mask_list': hist_mask_list}

        # spl_test = {'hist_adj_list': hist_adj_list,
		# 		'hist_ndFeats_list': hist_ndFeats_list,
		# 		'label_sp': label_sp_test,
		# 		'node_mask_list': hist_mask_list}

                                                   

        # spl_train = torch.load('./version4/spl_train') 
        # spl_test = torch.load('./version4/spl_test')
        # spl_train = torch.load('spl_train') 
        # spl_test = torch.load('spl_test')
        spl_train = torch.load('./realdata/spl_train') 
        spl_test = torch.load('./realdata/spl_test')

        train_idxs = spl_train['label_sp']['idx']
        test_idxs = spl_test['label_sp']['idx']
        dataset = dict([[node_idx.item(),self.extract_neighborhood(node_idx.item(), spl_train)] for node_idx in train_idxs])
        dataset.update(dict([[node_idx.item(),self.extract_neighborhood(node_idx.item(), spl_test)] for node_idx in test_idxs]))     
        torch.save(dataset,'./realdata/dataset')
        self.dataset = dataset
        # print(dataset)
        # de


        self.tr_step = 0
        best_eval_valid_pl = 0
        best_eval_valid = 0
        eval_valid = 0
        best_pl_rate = []
        best_rate = []
        epochs_without_impr = 0
        
        for e in range(self.args.num_epochs):
            eval_train = self.run_epoch(self.splitter.train,spl_train,train_idxs, e, 'TRAIN', grad = True)
            if len(self.splitter.dev)>0 :
                eval_valid,rate_acc, rate_acc_pl= self.run_eval(self.splitter.dev,spl_test,test_idxs, e, 'VALID', grad = False)
                if eval_valid[0] > best_eval_valid_pl:
                    best_eval_valid_pl = eval_valid[0]
                    best_pl_rate = rate_acc_pl 
                if eval_valid[1] > best_eval_valid:
                    best_eval_valid = eval_valid[1]
                    best_rate = rate_acc
                # if eval_valid>best_eval_valid:
                #     best_eval_valid = eval_valid
                #     epochs_without_impr = 0
                #     print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Best valid measure:'+str(eval_valid))
                # else:
                #     epochs_without_impr+=1
                #     if epochs_without_impr>self.args.early_stop_patience:
                #         print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Early stop.')
                #         break

            # if len(self.splitter.test)>0 and eval_valid==best_eval_valid and e>self.args.eval_after_epochs:
            #     eval_test, _ = self.run_epoch(self.splitter.test, e, 'TEST', grad = False)

            #     if self.args.save_node_embeddings:
            #         self.save_node_embs_csv(nodes_embs, self.splitter.train_idx, log_file+'_train_nodeembs.csv.gz')
            #         self.save_node_embs_csv(nodes_embs, self.splitter.dev_idx, log_file+'_valid_nodeembs.csv.gz')
            #         self.save_node_embs_csv(nodes_embs, self.splitter.test_idx, log_file+'_test_nodeembs.csv.gz')
        print('best explainer acc is', best_eval_valid_pl)
        print('best label acc is', best_eval_valid)
        print('explainer acc rate', best_pl_rate)
        print('label acc rate', best_rate)

    def run_epoch(self, split,spl,train_idxs, epoch, set_name, grad):
        t0 = time.time()
        log_interval=999
        if set_name=='TEST':
            log_interval=1
        self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)

        torch.set_grad_enabled(grad)
        # for s in split:
        #     print(s.label_sp['idx'].shape)
          
        # for s in split:
            
        #     if self.tasker.is_static:
        #         s = self.prepare_static_sample(s)
        #     else:
        #         s = self.prepare_sample(s)
        #     # print(s.hist_adj_list)
        #     # print(s.hist_ndFeats_list[0].shape)
        #     # print(s.label_sp['idx'].shape)
        #     # print(s.label_sp['vals'].shape)
        #     # de
        #     # print(s.node_mask_list)
        #     predictions, nodes_embs = self.predict(s.hist_adj_list,
        #                                            s.hist_ndFeats_list,
        #                                            s.label_sp['idx'],
        #                                            s.node_mask_list)

        #     loss = self.comp_loss(predictions,s.label_sp['vals'])
        #     # print(loss)
        #     if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
        #         self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
        #     else:
        #         self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach())
        #     if grad:
        #         self.optim_step(loss)
     
        # hist_adj_list = spl['hist_adj_list'].copy()
        # hist_ndFeats_list = spl['hist_ndFeats_list'].copy()
        label_sp = spl['label_sp'].copy()
        # node_mask_list = spl['node_mask_list'].copy()
       
        # for i in range(7):
        #     hist_adj_list[i] = hist_adj_list[i].to(self.args.device)
        #     hist_ndFeats_list[i] = hist_ndFeats_list[i].float().to(self.args.device)
        #     node_mask_list[i] = node_mask_list[i].unsqueeze(1).to(self.args.device)
        label_sp['idx'] =  label_sp['idx'].unsqueeze(0).to(self.args.device)
        label_sp['vals'] = label_sp['vals'].to(self.args.device)
        s_idx = 0
        
        # batch size
        batch_size = 32
        num_train = label_sp['idx'].shape[1]
        perm = np.random.permutation(num_train)
        acc_mean = []
        acc_pl_mean = []
        for beg_ind in range(0, num_train, batch_size):
            end_ind = min(beg_ind+batch_size, num_train)
            perm_train_idxs = train_idxs[perm[beg_ind: end_ind]].cpu().numpy().tolist()
            self.gcn_opt.zero_grad()
            self.classifier_opt.zero_grad()
            predictions, mask_adj_list, loss2,loss3,loss4, label,label_pl, pred_pl = zip(*map(self.predict, perm_train_idxs))
            # #loss 1: 与ground truth CEloss
            # predictions = torch.stack(predictions)
            # label = torch.stack(label)
            # loss1 = self.comp_loss(predictions,label)


            # print(loss1, loss2, loss3, loss4)
            #loss2: 解释子图稀疏性
            #loss3: 时间片稀疏性
            #loss4: 解释子图和原图的相似性
            loss2 = torch.stack(loss2).mean()
            loss3 = 0.01*torch.stack(loss3).mean()
            loss4 = torch.stack(loss4).mean()

            #loss5: 与解释目标label CEloss
            label_pl = torch.stack(label_pl)
            pred_pl = torch.stack(pred_pl)
            loss5 = self.comp_loss(pred_pl,label_pl)
            #loss 1: 与ground truth CEloss
            label = torch.stack(label)
            loss1 = self.comp_loss(pred_pl,label)

            # print(pred_pl)
            # print(label_pl)
            # print(label)
            # loss = loss1 + loss2 + loss3 + loss4 + loss5
            # print('1',loss1)
            # print('5',loss5)
            loss = loss1+loss5
            # print('loss1',loss1.item())
            # print('loss5',loss5.item())
            # print('loss5', loss5)            
            if grad:
                self.optim_step(loss)

            # pred = torch.max(predictions, dim=1)[1]
            # acc = (pred==label)+0
            # acc = sum(acc) / len(perm_train_idxs)
            # acc_mean.append(acc)

            pred_pl = torch.max(pred_pl, dim=1)[1]

            acc = (pred_pl==label)+0
            acc = sum(acc) / len(perm_train_idxs)
            acc_mean.append(acc)

            acc_pl = (pred_pl==label_pl)+0
            acc_pl = sum(acc_pl) / len(perm_train_idxs)
            acc_pl_mean.append(acc_pl) 

        acc_mean = torch.stack(acc_mean).mean()
        acc_pl_mean = torch.stack(acc_pl_mean).mean()
        print('acc is',acc_mean.item())
        print('acc_pl is',acc_pl_mean.item())
        # predictions, nodes_embs,mask_adj_list,update_list = self.predict(hist_adj_list,
        #                                             hist_ndFeats_list,
        #                                             label_sp['idx'],label_sp['vals'],
        #                                            node_mask_list,
        #                                            s_idx,hist_adj_list, set_name, reward)

        # print(predictions.shape)
        # print(label_sp['vals'].shape)
        # if set_name=='VALID':
        # pred = torch.max(predictions, dim=1)[1]
        # acc = (pred==(label_sp['vals']))+0
        # acc = sum(acc) / len(acc)
        # print('acc is',acc.item())

        # loss1 = self.comp_loss(predictions,label_sp['vals'])
        # explain_num = 0
        # original_num = 0
        # update_sum = 0
        # kl_sum = 0
        # explain_graph_adj_list = []

        # for i in range(len(mask_adj_list)):
        #     # print(explain_graph_adj[i].to_dense())
        #     # de
        #     explain_graph_adj = hist_adj_list[i].to_dense().mul(mask_adj_list[i])
        #     # print(explain_graph_adj)
        #     # print(hist_adj_list[i].to_dense())
        #     # de
        #     explain_num += explain_graph_adj.sum()
        #     original_num += hist_adj_list[i].to_dense().sum()
        #     update_sum += torch.sum(abs(update_list[i]))
        #     kl_sum += F.kl_div(explain_graph_adj.to_dense(),hist_adj_list[i].to_dense(),reduction='mean')
        #     explain_graph_adj_list.append(explain_graph_adj)
        # loss2 = explain_num / original_num 
        # loss3 = 0.0001*update_sum / len(mask_adj_list)
        # loss4 = 100* kl_sum / len(mask_adj_list)
                
        # if set_name=='VALID':
        # nodes_embs_pl = self.gcn2(explain_graph_adj_list,
        #                     hist_ndFeats_list,                       
        #                     node_mask_list)
        # nodes_embs_ori = self.gcn2(hist_adj_list,
        #                     hist_ndFeats_list,                       
        #                     node_mask_list)
                    
        # predictions_pl = self.predict_pl(nodes_embs_pl, label_sp['idx'])
        # predictions_ori = self.predict_pl(nodes_embs_ori, label_sp['idx'])
        # pred_pl = torch.max(predictions_pl, dim=1)[1]
        # pred_ori = torch.max(predictions_ori, dim=1)[1]
        # acc_pl = (pred_pl==pred_ori)+0
        # acc_pl = sum(acc_pl) / len(acc_pl)
        # print('explainer acc is',acc_pl.item())
        
        # loss5 = self.comp_loss(predictions_pl, pred_ori)  
        # loss = loss1 + loss2 + loss3 + loss4 + loss5
        

        # rate_acc_pl = []
        # rate_acc = []
        # if set_name=='VALID':
        #     for i in range(1, 11):
        #         explain_graph_adj_list_i = []
        #         for j in range(len(mask_adj_list)):

        #             flatten_mask_adj = hist_adj_list[j].to_dense().mul(mask_adj_list[j]).flatten(0)
        #             topk = hist_adj_list[j]._values().shape[0] * i // 10
        #             # topk = torch.tensor([topk], device=self.args.device).unsqueeze(-1)
        #             # print(flatten_mask_adj.sort(1,descending=True).values.shape)
        #             # print(torch.gather(flatten_mask_adj.sort(1,descending=True).values, 1, topk))
        #             values, indices = torch.topk(flatten_mask_adj, topk)
        #             threshold = values[-1]
        #             # threshold = torch.gather(flatten_mask_adj.sort(0,descending=True).values, 0, topk)
        #             # print('de', threshold)
        #             threshold = max(threshold, 1E-6)
        #             topk_alpha_adj = (flatten_mask_adj > threshold).float().view(mask_adj_list[j].to_dense().shape)
        #             # explain_graph_adj_j = hist_adj_list[j].to_dense().mul(topk_alpha_adj)
        #             explain_graph_adj_j = topk_alpha_adj
        #             # if i == 10:
        #             #     print(explain_graph_adj_j)
        #             #     print(hist_adj_list[j].to_dense())
        #             #     print(explain_graph_adj_j - hist_adj_list[j].to_dense())
        #             #     print((explain_graph_adj_j - hist_adj_list[j].to_dense()).sum())
        #             #     de
        #             explain_graph_adj_list_i.append(explain_graph_adj_j)
        #         nodes_embs_pl_i = self.gcn2(explain_graph_adj_list_i,
        #                         hist_ndFeats_list,                       
        #                         node_mask_list)
        #         predictions_pl_i = self.predict_pl(nodes_embs_pl_i, label_sp['idx'])
        #         pred_pl_i = torch.max(predictions_pl_i, dim=1)[1]
        #         acc_pl_i = (pred_pl_i==pred_ori)+0
        #         acc_pl_i = sum(acc_pl_i) / len(acc_pl_i)
        #         acc_i = (pred_pl_i==(label_sp['vals']))+0
        #         acc_i = sum(acc_i) / len(acc_i)
        #         rate_acc_pl.append(acc_pl_i.item())
        #         rate_acc.append(acc_i.item())

        # print(loss1)
        # print(loss2)
        # print(loss3)
        # print(loss4)
        # print(loss5)
        # de
        # loss = loss1 

        # if set_name=='TRAIN':
        #     c = 2
        #     reward = -loss.item() + c
        print('loss is',loss.item())

        # if grad:
        #     self.optim_step(loss)
        torch.set_grad_enabled(True)
        # eval_measure = self.logger.log_epoch_done()
        eval_measure = [acc_pl.item(), acc.item()]
        return eval_measure

    def run_eval(self, split,spl,test_idxs, epoch, set_name, grad):
        t0 = time.time()
        log_interval=999
        self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)

        torch.set_grad_enabled(grad)
        # for s in split:
        #     print(s.label_sp['idx'].shape)
          
        # for s in split:
            
        #     if self.tasker.is_static:
        #         s = self.prepare_static_sample(s)
        #     else:
        #         s = self.prepare_sample(s)
        #     # print(s.hist_adj_list)
        #     # print(s.hist_ndFeats_list[0].shape)
        #     # print(s.label_sp['idx'].shape)
        #     # print(s.label_sp['vals'].shape)
        #     # de
        #     # print(s.node_mask_list)
        #     predictions, nodes_embs = self.predict(s.hist_adj_list,
        #                                            s.hist_ndFeats_list,
        #                                            s.label_sp['idx'],
        #                                            s.node_mask_list)

        #     loss = self.comp_loss(predictions,s.label_sp['vals'])
        #     # print(loss)
        #     if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
        #         self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
        #     else:
        #         self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach())
        #     if grad:
        #         self.optim_step(loss)
     
        # hist_adj_list = spl['hist_adj_list'].copy()
        # hist_ndFeats_list = spl['hist_ndFeats_list'].copy()
        label_sp = spl['label_sp'].copy()
        # node_mask_list = spl['node_mask_list'].copy()
       
        # for i in range(7):
        #     hist_adj_list[i] = hist_adj_list[i].to(self.args.device)
        #     hist_ndFeats_list[i] = hist_ndFeats_list[i].float().to(self.args.device)
        #     node_mask_list[i] = node_mask_list[i].unsqueeze(1).to(self.args.device)
        label_sp['idx'] =  label_sp['idx'].unsqueeze(0).to(self.args.device)
        label_sp['vals'] = label_sp['vals'].to(self.args.device)
        s_idx = 0
        rate_acc_pl = torch.zeros(10)
        rate_acc = torch.zeros(10)
        acc_mean = 0
        acc_pl_mean = 0
        case = []
        case_weight = []
        for idx in test_idxs:
            
            # data = dataset[idx]
            # org_probs = F.softmax(classifier(data['feat'], data['sub_adj'])[0][:,data['node_idx_new']], dim=1)
            # pred_labels = torch.argmax(org_probs, axis=1)
            # mu, _ = model.encode(data['sub_feat'], data['adj_norm'])
            # alpha_mu = torch.zeros_like(mu)
            # alpha_mu[:,:,:args.K] = mu[:,:,:args.K]
            # alpha_adj = torch.sigmoid(model.dc(alpha_mu))
            # masked_alpha_adj = alpha_adj * data['sub_adj']
            # flatten_alpha_adj = masked_alpha_adj.flatten(1)

            predictions, mask_adj_list, loss2,loss3,loss4, label,label_pl,pred_pl = self.predict(idx.item())
            pred_pl = torch.argmax(pred_pl)
            acc = (pred_pl==label)+0
            acc_pl = (pred_pl==label_pl)+0
            if acc:
                acc_mean += 1
            if acc_pl:
                acc_pl_mean += 1
            # acc_mean = torch.stack(acc_mean).mean()
            # acc_pl_mean = torch.stack(acc_pl_mean).mean()
            

            data=self.dataset[idx.item()]
            node_idx_new = data['node_idx_new']
            hist_adj_list = data['hist_adj_list']
            hist_ndFeats_list = data['hist_ndFeats_list']
            mask_list = data['hist_mask_list']
            mapping = data['mapping']
            node_indices = data['node_indices']
            label = data['label']
            
            case.append(hist_adj_list)
            case_weight.append(mask_adj_list)

            for i in range(1,10):
                explain_graph_adj_list_i = []
                per_num = hist_adj_list[-1]._values().shape[0] // 10 + 1
                for j in range(len(mask_adj_list)):
                    flatten_mask_adj = hist_adj_list[j].to_dense().mul(mask_adj_list[j]).flatten(0)
                    topk = min(i*per_num,hist_adj_list[j]._values().shape[0]) 
                    # topk = torch.tensor([topk], device=self.args.device).unsqueeze(-1)
                    # print(flatten_mask_adj.sort(1,descending=True).values.shape)
                    # print(torch.gather(flatten_mask_adj.sort(1,descending=True).values, 1, topk))
                    values, indices = torch.topk(flatten_mask_adj, topk)
                    threshold = values[-1]
                    # threshold = torch.gather(flatten_mask_adj.sort(0,descending=True).values, 0, topk)
                    # print('de', threshold)
                    threshold = max(threshold, 1E-6)
                    topk_alpha_adj = hist_adj_list[j].to_dense().mul((flatten_mask_adj > threshold).float().view(mask_adj_list[j].to_dense().shape))
                    # explain_graph_adj_j = hist_adj_list[j].to_dense().mul(topk_alpha_adj)
                    explain_graph_adj_j = topk_alpha_adj
                    # print(i,j,topk_alpha_adj.sum())
                    # print(i,j,hist_adj_list[j].to_dense().sum())
                    # if i == 10:
                    #     print(explain_graph_adj_j)
                    #     print(hist_adj_list[j].to_dense())
                    #     print(explain_graph_adj_j - hist_adj_list[j].to_dense())
                    #     print((explain_graph_adj_j - hist_adj_list[j].to_dense()).sum())
                    #     de
                    explain_graph_adj_list_i.append(explain_graph_adj_j)
                nodes_embs_pl_i = self.gcn2(explain_graph_adj_list_i,
                                hist_ndFeats_list,                       
                                mask_list)
                predictions_pl_i = self.predict_pl(nodes_embs_pl_i, node_indices)
                pred_pl_i = torch.max(predictions_pl_i, dim=1)[1][node_idx_new]
                acc_pl_i = (pred_pl_i==label_pl)+0
                # acc_pl_i = sum(acc_pl_i) / len(acc_pl_i)
                acc_i = (pred_pl_i==label)+0
                # acc_i = sum(acc_i) / len(acc_i)
                # rate_acc_pl.append(acc_pl_i.item())
                # rate_acc.append(acc_i.item())
                rate_acc_pl[i]+=acc_pl_i.item()
                rate_acc[i]+=acc_i.item()
        rate_acc_pl /= len(test_idxs)
        rate_acc /= len(test_idxs)
        # print(acc_mean)
        # print(len(test_idxs))
        # acc_mean = acc_mean / len(test_idxs)
        acc_mean /= len(test_idxs)
        acc_pl_mean /= len(test_idxs)
        # print('acc is',acc_mean)
        print('acc_pl is',acc_pl_mean)
        # print('rate_acc_pl', rate_acc_pl)
        print('rate_acc_pl', rate_acc_pl)
        print('rate_acc',rate_acc)
        
        torch.save(case,'case')
        torch.save(case_weight,'case_weight')
        print('save')
        # if set_name=='VALID':
        # nodes_embs_pl = self.gcn2(explain_graph_adj_list,
        #                     hist_ndFeats_list,                       
        #                     node_mask_list)
        # nodes_embs_ori = self.gcn2(hist_adj_list,
        #                     hist_ndFeats_list,                       
        #                     node_mask_list)
                    
        # predictions_pl = self.predict_pl(nodes_embs_pl, label_sp['idx'])
        # predictions_ori = self.predict_pl(nodes_embs_ori, label_sp['idx'])
        # pred_pl = torch.max(predictions_pl, dim=1)[1]
        # pred_ori = torch.max(predictions_ori, dim=1)[1]
        # acc_pl = (pred_pl==pred_ori)+0
        # acc_pl = sum(acc_pl) / len(acc_pl)
        # print('explainer acc is',acc_pl.item())
        
        # loss5 = self.comp_loss(predictions_pl, pred_ori)  
        # loss = loss1 + loss2 + loss3 + loss4 + loss5
        

        # rate_acc_pl = []
        # rate_acc = []
        # if set_name=='VALID':
        #     for i in range(1, 11):
        #         explain_graph_adj_list_i = []
        #         for j in range(len(mask_adj_list)):

        #             flatten_mask_adj = hist_adj_list[j].to_dense().mul(mask_adj_list[j]).flatten(0)
        #             topk = hist_adj_list[j]._values().shape[0] * i // 10
        #             # topk = torch.tensor([topk], device=self.args.device).unsqueeze(-1)
        #             # print(flatten_mask_adj.sort(1,descending=True).values.shape)
        #             # print(torch.gather(flatten_mask_adj.sort(1,descending=True).values, 1, topk))
        #             values, indices = torch.topk(flatten_mask_adj, topk)
        #             threshold = values[-1]
        #             # threshold = torch.gather(flatten_mask_adj.sort(0,descending=True).values, 0, topk)
        #             # print('de', threshold)
        #             threshold = max(threshold, 1E-6)
        #             topk_alpha_adj = (flatten_mask_adj > threshold).float().view(mask_adj_list[j].to_dense().shape)
        #             # explain_graph_adj_j = hist_adj_list[j].to_dense().mul(topk_alpha_adj)
        #             explain_graph_adj_j = topk_alpha_adj
        #             # if i == 10:
        #             #     print(explain_graph_adj_j)
        #             #     print(hist_adj_list[j].to_dense())
        #             #     print(explain_graph_adj_j - hist_adj_list[j].to_dense())
        #             #     print((explain_graph_adj_j - hist_adj_list[j].to_dense()).sum())
        #             #     de
        #             explain_graph_adj_list_i.append(explain_graph_adj_j)
        #         nodes_embs_pl_i = self.gcn2(explain_graph_adj_list_i,
        #                         hist_ndFeats_list,                       
        #                         node_mask_list)
        #         predictions_pl_i = self.predict_pl(nodes_embs_pl_i, label_sp['idx'])
        #         pred_pl_i = torch.max(predictions_pl_i, dim=1)[1]
        #         acc_pl_i = (pred_pl_i==pred_ori)+0
        #         acc_pl_i = sum(acc_pl_i) / len(acc_pl_i)
        #         acc_i = (pred_pl_i==(label_sp['vals']))+0
        #         acc_i = sum(acc_i) / len(acc_i)
        #         rate_acc_pl.append(acc_pl_i.item())
        #         rate_acc.append(acc_i.item())

        # print(loss1)
        # print(loss2)
        # print(loss3)
        # print(loss4)
        # print(loss5)
        # de
        # loss = loss1 

        # if set_name=='TRAIN':
        #     c = 2
        #     reward = -loss.item() + c
        # print('loss is',loss.item())

        # if grad:
        #     self.optim_step(loss)
        torch.set_grad_enabled(True)
        # eval_measure = self.logger.log_epoch_done()
        eval_measure = [acc_pl_mean, acc_mean]
        return eval_measure, rate_acc_pl, rate_acc_pl

    # def predict(self,hist_adj_list,hist_ndFeats_list,node_indices,node_label,mask_list,s_idx,hist_hop,set_name,reward):
    #     nodes_embs,mask_adj_list,update_list = self.gcn(hist_adj_list,
    #                           hist_ndFeats_list,
                              
    #                           mask_list)

    #     predict_batch_size = 100000
    #     gather_predictions=[]
    #     for i in range(1 +(node_indices.size(1)//predict_batch_size)):
    #         cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
    #         predictions = self.classifier(cls_input)
            
    #         gather_predictions.append(predictions)
    #     # gather_predictions=[]
    #     # cls_input = []
    #     # for node_set in node_indices:
    #     #     cls_input.append(nodes_embs[node_set])
    #     # cls_input = torch.tensor(cls_input)
    #     # gather_predictions = self.classifier(cls_input)
    #     gather_predictions=torch.cat(gather_predictions, dim=0)
    #     # print(gather_predictions.shape)
    #     # de
    #     return gather_predictions, nodes_embs,mask_adj_list,update_list
    def predict(self,perm_train_idxs):
        data = self.dataset[perm_train_idxs]
        node_idx_new = data['node_idx_new']
        # 边数 80左右
        hist_adj_list = data['hist_adj_list']
        hist_ndFeats_list = data['hist_ndFeats_list']
        mask_list = data['hist_mask_list']
        mapping = data['mapping']
        node_indices = data['node_indices']
        label = data['label']
        # print(hist_adj_list[0])
        nodes_embs,mask_adj_list,update_list = self.gcn(hist_adj_list,
                              hist_ndFeats_list,
                              
                              mask_list)

        predict_batch_size = 100000

        gather_predictions=[]
        for i in range(1 +(node_indices.size(1)//predict_batch_size)):
            cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
            predictions = self.classifier(cls_input)
            
            gather_predictions.append(predictions)
        # gather_predictions=[]
        # cls_input = []
        # for node_set in node_indices:
        #     cls_input.append(nodes_embs[node_set])
        # cls_input = torch.tensor(cls_input)
        # gather_predictions = self.classifier(cls_input)
        gather_predictions=torch.cat(gather_predictions, dim=0)
        gather_predictions = gather_predictions[node_idx_new,:]
        # print(gather_predictions.shape)
        # de
        

        # loss1 = self.comp_loss(predictions,label_sp['vals'])
        
        #loss2: 解释子图稀疏性
        #loss3: 时间片稀疏性
        #loss4: 解释子图和原图的相似性
        explain_num = 0
        original_num = 0
        update_sum = 0
        kl_sum = 0
        explain_graph_adj_list = []
        for i in range(len(mask_adj_list)):
            # print(explain_graph_adj[i].to_dense())
            # de
            explain_graph_adj = hist_adj_list[i].to_dense().mul(mask_adj_list[i])
            # print(explain_graph_adj)
            # print(hist_adj_list[i].to_dense())
            # de
            explain_num += explain_graph_adj.sum()
            original_num += hist_adj_list[i].to_dense().sum()
            update_sum += torch.sum(abs(update_list[i]))
            kl_sum += F.kl_div(explain_graph_adj.to_dense(),hist_adj_list[i].to_dense(),reduction='mean')
            explain_graph_adj_list.append(explain_graph_adj)
        loss2 = explain_num / original_num 
        loss3 = update_sum / len(mask_adj_list)
        loss4 = kl_sum / len(mask_adj_list)
        
        # explain label
        # print(hist_adj_list[-1])
        nodes_embs2 = self.gcn2(hist_adj_list,
                              hist_ndFeats_list,
                              
                              mask_list)
        gather_predictions2 = self.predict_pl(nodes_embs2, node_indices) 
        label_pl = torch.max(gather_predictions2, dim=1)[1]
        label_pl = label_pl[node_idx_new]

        # explain_pred
        explain_graph_adj_list = []
        for i in range(len(mask_adj_list)):
            explain_graph_adj = hist_adj_list[i].to_dense().mul(mask_adj_list[i])
            explain_graph_adj_list.append(explain_graph_adj)
        nodes_embs3 = self.gcn2(explain_graph_adj_list,
                              hist_ndFeats_list,                              
                              mask_list)
        pred_pl = self.predict_pl(nodes_embs3, node_indices) 
        pred_pl = pred_pl[node_idx_new,:]

        # gather_predictions3=[]
        # for i in range(1 +(node_indices.size(1)//predict_batch_size)):
        #     cls_input3 = self.gather_node_embs(nodes_embs3, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
        #     predictions3 = self.classifier2(cls_input3)  
        #     gather_predictions3.append(predictions3)  
        # pred_pl=torch.cat(gather_predictions3, dim=0) 
        # pred_pl = torch.max(gather_predictions3, dim=1)[1]
        # pred_pl = pred_pl[node_idx_new,:]

        return gather_predictions, mask_adj_list, loss2,loss3,loss4,label,label_pl, pred_pl

    def predict_pl(self, nodes_embs, node_indices):
        predict_batch_size = 100000
        gather_predictions=[]
        for i in range(1 +(node_indices.size(1)//predict_batch_size)):
            cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
            predictions = self.classifier2(cls_input)
            
            gather_predictions.append(predictions)
        gather_predictions=torch.cat(gather_predictions, dim=0)
        return gather_predictions

    def gather_node_embs(self,nodes_embs,node_indices):
        cls_input = []

        for node_set in node_indices:
            cls_input.append(nodes_embs[node_set])
        return torch.cat(cls_input,dim = 1)

    def optim_step(self,loss):
        self.tr_step += 1
        loss.backward()

        if self.tr_step % self.args.steps_accum_gradients == 0:
            self.gcn_opt.step()
            self.classifier_opt.step()

            self.gcn_opt.zero_grad()
            self.classifier_opt.zero_grad()


    def prepare_sample(self,sample):
        sample = u.Namespace(sample)
        for i,adj in enumerate(sample.hist_adj_list):
            adj = u.sparse_prepare_tensor(adj,torch_size = [self.num_nodes])
            sample.hist_adj_list[i] = adj.to(self.args.device)

            nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])

            sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
            node_mask = sample.node_mask_list[i]
            sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer

        label_sp = self.ignore_batch_dim(sample.label_sp)

        if self.args.task in ["link_pred", "edge_cls"]:
            label_sp['idx'] = label_sp['idx'].to(self.args.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
        else:
            label_sp['idx'] = label_sp['idx'].to(self.args.device)

        label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
        sample.label_sp = label_sp

        return sample

    def prepare_static_sample(self,sample):
        sample = u.Namespace(sample)

        sample.hist_adj_list = self.hist_adj_list

        sample.hist_ndFeats_list = self.hist_ndFeats_list

        label_sp = {}
        label_sp['idx'] =  [sample.idx]
        label_sp['vals'] = sample.label
        sample.label_sp = label_sp

        return sample

    def ignore_batch_dim(self,adj):
        if self.args.task in ["link_pred", "edge_cls"]:
            adj['idx'] = adj['idx'][0]
        adj['vals'] = adj['vals'][0]
        return adj

    def save_node_embs_csv(self, nodes_embs, indexes, file_name):
        csv_node_embs = []
        for node_id in indexes:
            orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

            csv_node_embs.append(torch.cat((orig_ID,nodes_embs[node_id].double())).detach().numpy())

        pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')
        #print ('Node embs saved in',file_name)

# import torch
# import utils as u
# import logger
# import time
# import pandas as pd
# import numpy as np
# import random
# from scipy import sparse

# class Trainer():
#     def __init__(self,args, splitter, gcn, classifier, gcn2, classifier2, comp_loss, dataset, num_classes):
#         self.args = args
#         self.splitter = splitter
#         self.tasker = splitter.tasker
#         self.gcn = gcn
#         self.classifier = classifier
#         self.gcn2 = gcn2
#         self.classifier2 = classifier2
#         self.comp_loss = comp_loss

#         self.num_nodes = dataset.num_nodes
#         self.data = dataset
#         self.num_classes = num_classes

#         self.logger = logger.Logger(args, self.num_classes)

#         self.init_optimizers(args)

#         if self.tasker.is_static:
#             adj_matrix = u.sparse_prepare_tensor(self.tasker.adj_matrix, torch_size = [self.num_nodes], ignore_batch_dim = False)
#             self.hist_adj_list = [adj_matrix]
#             self.hist_ndFeats_list = [self.tasker.nodes_feats.float()]

#     def init_optimizers(self,args):
#         params = self.gcn.parameters()
#         self.gcn_opt = torch.optim.Adam(params, lr = args.learning_rate)
#         params = self.classifier.parameters()
#         self.classifier_opt = torch.optim.Adam(params, lr = args.learning_rate)
#         self.gcn_opt.zero_grad()
#         self.classifier_opt.zero_grad()

#     def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
#         torch.save(state, filename)

#     def load_checkpoint(self, filename, model):
#         if os.path.isfile(filename):
#             print("=> loading checkpoint '{}'".format(filename))
#             checkpoint = torch.load(filename)
#             epoch = checkpoint['epoch']
#             self.gcn.load_state_dict(checkpoint['gcn_dict'])
#             self.classifier.load_state_dict(checkpoint['classifier_dict'])
#             self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
#             self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
#             self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
#             return epoch
#         else:
#             self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
#             return 0

#     def train(self):
#         self.tr_step = 0
#         best_eval_valid = 0
#         eval_valid = 0
#         epochs_without_impr = 0
#         spl_train = torch.load('spl_train') 
#         spl_test = torch.load('spl_test')
#         for e in range(self.args.num_epochs):
#             eval_train, nodes_embs = self.run_epoch(spl_train, e, 'TRAIN', grad = True)
#             if len(self.splitter.dev)>0 and e>self.args.eval_after_epochs:
#                 eval_valid, _ = self.run_epoch(spl_test, e, 'VALID', grad = False)
#                 if eval_valid>best_eval_valid:
#                     best_eval_valid = eval_valid
#                     epochs_without_impr = 0
#                     print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Best valid measure:'+str(eval_valid))
#                 else:
#                     epochs_without_impr+=1
#                     if epochs_without_impr>self.args.early_stop_patience:
#                         print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Early stop.')
#                         break
#             # if len(self.splitter.test)>0 and e>self.args.eval_after_epochs:
#             # if len(self.splitter.test)>0 and eval_valid==best_eval_valid and e>self.args.eval_after_epochs:
#             #     eval_test, _ = self.run_epoch(self.splitter.test, e, 'TEST', grad = False)

#             #     if self.args.save_node_embeddings:
#             #         self.save_node_embs_csv(nodes_embs, self.splitter.train_idx, log_file+'_train_nodeembs.csv.gz')
#             #         self.save_node_embs_csv(nodes_embs, self.splitter.dev_idx, log_file+'_valid_nodeembs.csv.gz')
#             #         self.save_node_embs_csv(nodes_embs, self.splitter.test_idx, log_file+'_test_nodeembs.csv.gz')


#     def run_epoch(self, split, epoch, set_name, grad):
#         t0 = time.time()
#         log_interval=999
#         if set_name=='TEST':
#             log_interval=1
#         self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)

#         torch.set_grad_enabled(grad)
#         # for s in split:
#         #     if self.tasker.is_static:
#         #         s = self.prepare_static_sample(s)
#         #     else:
#         #         s = self.prepare_sample(s)
#         #     print(s.idx.item())
#         # debug

#         # 结点统计
#         # for s in split:
#         # de
#         # for s in split:
#         #     if self.tasker.is_static:
#         #         s = self.prepare_static_sample(s)
#         #     else:
#         #         s = self.prepare_sample(s)
#         #     # print(str(s.idx.item()))
#         #     # de
#         #     # # random sample
#         #     # for i in range(len(s.hist_adj_list)):
#         #     #     s_dense = np.array(s.hist_adj_list[i].to_dense())
#         #     #     sample_index=list(range(1000))
#         #     #     random.shuffle(sample_index)
#         #     #     for j in sample_index[:900]:
#         #     #         for k in sample_index[:900]:
#         #     #             s_dense[j,k]=0
#         #     #     edge_index_temp = sparse.coo_matrix(s_dense)

#         #     #     values = edge_index_temp.data  # 边上对应权重值weight
#         #     #     indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
#         #     #     # edge_index_A = torch.LongTensor(indices)  # 我们真正需要的coo形式
#         #     #     indices_i = torch.LongTensor(indices)  # 转tensor
#         #     #     values_v = torch.FloatTensor(values)  # 转tensor
#         #     #     edge_index = torch.sparse_coo_tensor(indices_i, values_v , edge_index_temp.shape)
#         #     #     s.hist_adj_list[i]=edge_index
#         #     # print(s.hist_ndFeats_list[0])
#         #     # debug
#         #     # print('d',s.hist_ndFeats_list[0])

#         #     predictions, nodes_embs = self.predict(s.hist_adj_list,
#         #                                            s.hist_ndFeats_list,
#         #                                            s.hist_label_sp,
#         #                                            s.node_mask_list, s.idx.item(),s.hist_hop, set_name)


#         #     loss = self.comp_loss(predictions,s.hist_label_sp[-1]['vals'])
#         #     # print(loss)
#         #     if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
#         #         self.logger.log_minibatch(predictions, s.hist_label_sp[-1]['vals'], loss.detach(), adj = s.hist_label_sp[-1]['idx'])
#         #     else:
#         #         self.logger.log_minibatch(predictions, s.hist_label_sp[-1]['vals'], loss.detach())
#         #     if grad:
#         #         self.optim_step(loss)
#         hist_adj_list = spl['hist_adj_list'].copy()
#         hist_ndFeats_list = spl['hist_ndFeats_list'].copy()
#         label_sp = spl['label_sp'].copy()
#         node_mask_list = spl['node_mask_list'].copy()

#         for i in range(5):
#             hist_adj_list[i] = hist_adj_list[i].to(self.args.device)
#             hist_ndFeats_list[i] = hist_ndFeats_list[i].float().to(self.args.device)
#             node_mask_list[i] = node_mask_list[i].unsqueeze(1).to(self.args.device)
#         label_sp['idx'] =  label_sp['idx'].unsqueeze(0).to(self.args.device)
#         s_idx = 0
#         predictions, nodes_embs = self.predict(hist_adj_list,
#                                                     hist_ndFeats_list,
#                                                     label_sp['idx'],
#                                                    node_mask_list,
#                                                    s_idx,hist_adj_list, set_name)
#         # print(predictions.shape)
#         # print(label_sp['vals'].shape)
#         if set_name=='VALID':
#             pred = torch.max(predictions, dim=1)[1]
#             acc = (pred==(label_sp['vals']-1))+0
#             acc = sum(acc) / len(acc)
#             print('acc is',acc)
#         loss = self.comp_loss(predictions,label_sp['vals']-1)
#         print('loss is',loss.item())
#         if grad:
#                 self.optim_step(loss)
#         torch.set_grad_enabled(True)
#         eval_measure = self.logger.log_epoch_done()

#         return eval_measure, nodes_embs

#     def predict(self,hist_adj_list,hist_ndFeats_list,node_indices,mask_list,s_idx,hist_hop,set_name):
#         nodes_embs = self.gcn(hist_adj_list,
#                               hist_ndFeats_list,
#                               node_indices,
#                               mask_list,self.classifier,s_idx,hist_hop,set_name)

#         predict_batch_size = 100000
#         gather_predictions=[]
#         for i in range(1 +(node_indices[-1]['idx'].size(1)//predict_batch_size)):
#             cls_input = self.gather_node_embs(nodes_embs, node_indices[-1]['idx'][:, i*predict_batch_size:(i+1)*predict_batch_size])
#             predictions = self.classifier(cls_input)
#             gather_predictions.append(predictions)
#         gather_predictions=torch.cat(gather_predictions, dim=0)
#         return gather_predictions, nodes_embs

#     def gather_node_embs(self,nodes_embs,node_indices):
#         cls_input = []

#         for node_set in node_indices:
#             cls_input.append(nodes_embs[node_set])
#         return torch.cat(cls_input,dim = 1)

#     def optim_step(self,loss):
#         self.tr_step += 1
#         loss.backward()

#         if self.tr_step % self.args.steps_accum_gradients == 0:
#             self.gcn_opt.step()
#             self.classifier_opt.step()

#             self.gcn_opt.zero_grad()
#             self.classifier_opt.zero_grad()


#     def prepare_sample(self,sample):
#         sample = u.Namespace(sample)
#         for i,adj in enumerate(sample.hist_adj_list):
#             adj = u.sparse_prepare_tensor(adj,torch_size = [self.num_nodes])
#             sample.hist_adj_list[i] = adj.to(self.args.device)

#             nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])

#             sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
#             node_mask = sample.node_mask_list[i]
#             sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer

#         for i,adj in enumerate(sample.hist_hop):
#             adj = u.sparse_prepare_tensor(adj,torch_size = [self.num_nodes])
#             sample.hist_hop[i] = adj.to(self.args.device)

#             # nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])

#             # sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
#             # node_mask = sample.node_mask_list[i]
#             # sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer

#         for i in range(len(sample.hist_label_sp)):
#             label_sp = self.ignore_batch_dim(sample.hist_label_sp[i])
#             if self.args.task in ["link_pred", "edge_cls"]:
#                 label_sp['idx'] = label_sp['idx'].to(self.args.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
#             else:
#                 label_sp['idx'] = label_sp['idx'].to(self.args.device)

#             label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
#             sample.hist_label_sp[i] = label_sp     
#         # label_sp = self.ignore_batch_dim(sample.label_sp)

#         # if self.args.task in ["link_pred", "edge_cls"]:
#         #     label_sp['idx'] = label_sp['idx'].to(self.args.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
#         # else:
#         #     label_sp['idx'] = label_sp['idx'].to(self.args.device)

#         # label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
#         # sample.label_sp = label_sp

#         return sample

#     def prepare_static_sample(self,sample):
#         sample = u.Namespace(sample)

#         sample.hist_adj_list = self.hist_adj_list

#         sample.hist_ndFeats_list = self.hist_ndFeats_list

#         label_sp = {}
#         label_sp['idx'] =  [sample.idx]
#         label_sp['vals'] = sample.label
#         sample.label_sp = label_sp

#         return sample

#     def ignore_batch_dim(self,adj):
#         if self.args.task in ["link_pred", "edge_cls"]:
#             adj['idx'] = adj['idx'][0]
#         adj['vals'] = adj['vals'][0]
#         return adj

#     def save_node_embs_csv(self, nodes_embs, indexes, file_name):
#         csv_node_embs = []
#         for node_id in indexes:
#             orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

#             csv_node_embs.append(torch.cat((orig_ID,nodes_embs[node_id].double())).detach().numpy())

#         pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')
#         #print ('Node embs saved in',file_name)


# import torch
# import utils as u
# import logger
# import time
# import pandas as pd
# import numpy as np

# class Trainer():
#     def __init__(self,args, splitter, gcn, classifier,gcn2, classifier2, comp_loss, dataset, num_classes):
#         self.args = args
#         self.splitter = splitter
#         self.tasker = splitter.tasker
#         self.gcn = gcn
#         self.classifier = classifier
#         self.gcn2 = gcn2
#         self.classifier2 = classifier2
#         self.comp_loss = comp_loss

#         self.num_nodes = dataset.num_nodes
#         self.data = dataset
#         self.num_classes = num_classes

#         self.logger = logger.Logger(args, self.num_classes)

#         self.init_optimizers(args)

#         if self.tasker.is_static:
#             adj_matrix = u.sparse_prepare_tensor(self.tasker.adj_matrix, torch_size = [self.num_nodes], ignore_batch_dim = False)
#             self.hist_adj_list = [adj_matrix]
#             self.hist_ndFeats_list = [self.tasker.nodes_feats.float()]

#     def init_optimizers(self,args):
#         params = self.gcn.parameters()
#         self.gcn_opt = torch.optim.Adam(params, lr = args.learning_rate)
#         params = self.classifier.parameters()
#         self.classifier_opt = torch.optim.Adam(params, lr = args.learning_rate)
#         self.gcn_opt.zero_grad()
#         self.classifier_opt.zero_grad()

#     def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
#         torch.save(state, filename)

#     def load_checkpoint(self, filename, model):
#         if os.path.isfile(filename):
#             print("=> loading checkpoint '{}'".format(filename))
#             checkpoint = torch.load(filename)
#             epoch = checkpoint['epoch']
#             self.gcn.load_state_dict(checkpoint['gcn_dict'])
#             self.classifier.load_state_dict(checkpoint['classifier_dict'])
#             self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
#             self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
#             self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
#             return epoch
#         else:
#             self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
#             return 0

#     def train(self):
#         self.tr_step = 0
#         best_eval_valid = 0
#         eval_valid = 0
#         epochs_without_impr = 0

#         for e in range(self.args.num_epochs):
#             eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, 'TRAIN', grad = True)
#             # if len(self.splitter.dev)>0 and e>self.args.eval_after_epochs:
#             eval_valid, _ = self.run_epoch(self.splitter.dev, e, 'VALID', grad = False)
#             if eval_valid>best_eval_valid:
#                 best_eval_valid = eval_valid
#                 epochs_without_impr = 0
#                 print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Best valid measure:'+str(eval_valid))
#             else:
#                 epochs_without_impr+=1
#                 if epochs_without_impr>self.args.early_stop_patience:
#                     print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Early stop.')
#                     break

#             # if len(self.splitter.test)>0 and eval_valid==best_eval_valid and e>self.args.eval_after_epochs:
#             #     eval_test, _ = self.run_epoch(self.splitter.test, e, 'TEST', grad = False)

#             #     if self.args.save_node_embeddings:
#             #         self.save_node_embs_csv(nodes_embs, self.splitter.train_idx, log_file+'_train_nodeembs.csv.gz')
#             #         self.save_node_embs_csv(nodes_embs, self.splitter.dev_idx, log_file+'_valid_nodeembs.csv.gz')
#             #         self.save_node_embs_csv(nodes_embs, self.splitter.test_idx, log_file+'_test_nodeembs.csv.gz')


#     def run_epoch(self, split, epoch, set_name, grad):
#         t0 = time.time()
#         log_interval=999
#         if set_name=='TEST':
#             log_interval=1
#         self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)

#         torch.set_grad_enabled(grad)
#         acc_batch = []
#         for s in split:
#             if self.tasker.is_static:
#                 s = self.prepare_static_sample(s)
#             else:
#                 s = self.prepare_sample(s)
            
#             label_sp = s.label_sp.copy()
#             label_sp['idx'] =  label_sp['idx'].unsqueeze(0).to(self.args.device)
#             label_sp['vals'] = label_sp['vals'].to(self.args.device)
#             s_idx = 0
#             batch_size = 32
#             num_train = label_sp['idx'].shape[1]
#             perm = np.random.permutation(num_train)
#             acc_mean = []
#             acc_pl_mean = []
#             train_idxs = label_sp['idx']
#             for beg_ind in range(0, num_train, batch_size):
#                 end_ind = min(beg_ind+batch_size, num_train)
#                 perm_train_idxs = train_idxs[perm[beg_ind: end_ind]].cpu().numpy().tolist()
#                 self.gcn_opt.zero_grad()
#                 self.classifier_opt.zero_grad()
#                 predictions, mask_adj_list, loss2,loss3,loss4, label,label_pl, pred_pl = zip(*map(self.predict, perm_train_idxs))

#             # predictions, nodes_embs = self.predict(s.hist_adj_list,
#             #                                        s.hist_ndFeats_list,
#             #                                        s.label_sp['idx'],
#             #                                        s.node_mask_list)
#             pred = torch.max(predictions, dim=1)[1]
#             acc = (pred==(s.label_sp['vals']))+0
#             acc = sum(acc) / len(acc)
#             acc_batch.append(acc)
            
            
#             loss = self.comp_loss(predictions,s.label_sp['vals'])
#             # print(loss)
#             # if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
#             #     self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
#             # else:
#             #     self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach())
#             if grad:
#                 self.optim_step(loss)
#         acc_batch_mean = sum(acc_batch) / len(acc_batch) 
#         print('acc is',acc_batch_mean.item())
#         torch.set_grad_enabled(True)
#         # eval_measure = self.logger.log_epoch_done()
#         eval_measure = acc_batch_mean.item()
#         return eval_measure, nodes_embs

#     def predict(self,hist_adj_list,hist_ndFeats_list,node_indices,mask_list):
#         nodes_embs,mask_adj_list,update_list = self.gcn(hist_adj_list,
#                               hist_ndFeats_list,
#                               mask_list)

#         predict_batch_size = 100000
#         gather_predictions=[]
#         for i in range(1 +(node_indices.size(1)//predict_batch_size)):
#             cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
#             predictions = self.classifier(cls_input)
#             gather_predictions.append(predictions)
#         gather_predictions=torch.cat(gather_predictions, dim=0)
#         return gather_predictions, nodes_embs

#     def gather_node_embs(self,nodes_embs,node_indices):
#         cls_input = []

#         for node_set in node_indices:
#             cls_input.append(nodes_embs[node_set])
#         return torch.cat(cls_input,dim = 1)

#     def optim_step(self,loss):
#         self.tr_step += 1
#         loss.backward()

#         if self.tr_step % self.args.steps_accum_gradients == 0:
#             self.gcn_opt.step()
#             self.classifier_opt.step()

#             self.gcn_opt.zero_grad()
#             self.classifier_opt.zero_grad()


#     def prepare_sample(self,sample):
#         sample = u.Namespace(sample)
#         for i,adj in enumerate(sample.hist_adj_list):
#             adj = u.sparse_prepare_tensor(adj,torch_size = [self.num_nodes])
#             sample.hist_adj_list[i] = adj.to(self.args.device)

#             nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])

#             sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
#             node_mask = sample.node_mask_list[i]
#             sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer

#         label_sp = self.ignore_batch_dim(sample.label_sp)

#         if self.args.task in ["link_pred", "edge_cls"]:
#             label_sp['idx'] = label_sp['idx'].to(self.args.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
#         else:
#             label_sp['idx'] = label_sp['idx'].to(self.args.device)

#         label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
#         sample.label_sp = label_sp

#         return sample

#     def prepare_static_sample(self,sample):
#         sample = u.Namespace(sample)

#         sample.hist_adj_list = self.hist_adj_list

#         sample.hist_ndFeats_list = self.hist_ndFeats_list

#         label_sp = {}
#         label_sp['idx'] =  [sample.idx]
#         label_sp['vals'] = sample.label
#         sample.label_sp = label_sp

#         return sample

#     def ignore_batch_dim(self,adj):
#         if self.args.task in ["link_pred", "edge_cls"]:
#             adj['idx'] = adj['idx'][0]
#         adj['vals'] = adj['vals'][0]
#         return adj

#     def save_node_embs_csv(self, nodes_embs, indexes, file_name):
#         csv_node_embs = []
#         for node_id in indexes:
#             orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

#             csv_node_embs.append(torch.cat((orig_ID,nodes_embs[node_id].double())).detach().numpy())

#         pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')
#         #print ('Node embs saved in',file_name)

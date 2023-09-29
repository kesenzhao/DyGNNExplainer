import torch
import utils as u
import logger
import time
import pandas as pd
import numpy as np
import torch.nn.functional as F

class Trainer():
    def __init__(self,args, splitter, gcn, classifier, comp_loss, dataset, num_classes):
        self.args = args
        self.splitter = splitter
        self.tasker = splitter.tasker
        self.gcn = gcn
        self.classifier = classifier

        self.comp_loss = comp_loss

        self.num_nodes = dataset.num_nodes
        self.data = dataset
        self.num_classes = num_classes

        self.logger = logger.Logger(args, self.num_classes)

        self.init_optimizers(args)

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

    def train(self):
        spl_train = torch.load('./version3/spl_train') 
        spl_test = torch.load('./version3/spl_test')
        # spl_train = torch.load('spl_train_new') 
        # spl_test = torch.load('spl_test_new')
        # de


        self.tr_step = 0
        best_eval_valid = 0
        eval_valid = 0
        epochs_without_impr = 0
        reward = 2
        for e in range(self.args.num_epochs):
            eval_train, nodes_embs,reward = self.run_epoch(self.splitter.train,spl_train,reward, e, 'TRAIN', grad = True)
            if len(self.splitter.dev)>0 and e>self.args.eval_after_epochs:
                eval_valid, _,_ = self.run_epoch(self.splitter.dev,spl_test,reward, e, 'VALID', grad = False)
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


    def run_epoch(self, split,spl,reward, epoch, set_name, grad):
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
     
        hist_adj_list = spl['hist_adj_list'].copy()
        hist_ndFeats_list = spl['hist_ndFeats_list'].copy()
        label_sp = spl['label_sp'].copy()
        node_mask_list = spl['node_mask_list'].copy()
        # print(label_sp)
        # de

        for i in range(5):
            hist_adj_list[i] = hist_adj_list[i].to(self.args.device)
            hist_ndFeats_list[i] = hist_ndFeats_list[i].float().to(self.args.device)
            node_mask_list[i] = node_mask_list[i].unsqueeze(1).to(self.args.device)
        label_sp['idx'] =  label_sp['idx'].unsqueeze(0).to(self.args.device)
        label_sp['vals'] = label_sp['vals'].to(self.args.device)
        s_idx = 0
        
        predictions, nodes_embs = self.predict(hist_adj_list,
                                                    hist_ndFeats_list,
                                                    label_sp['idx'],label_sp['vals'],
                                                   node_mask_list,
                                                   s_idx,hist_adj_list, set_name, reward)
        # print(predictions.shape)
        # de
        # print(predictions.shape)
        # print(label_sp['vals'].shape)
        # if set_name=='VALID':

        pred = torch.max(predictions, dim=1)[1]
        # print(pred)
        # print(label_sp['vals'])
        
        # print(predictions.shape, label_sp['vals'])
        acc = (pred==(label_sp['vals']))+0
        acc = sum(acc) / len(acc)
        print('acc is',acc.item())
        loss1 = self.comp_loss(predictions,label_sp['vals'])
        # explain_num = 0
        # original_num = 0
        # update_sum = 0
        # kl_sum = 0
        # for i in range(len(explain_graph_adj)):
        #     # print(explain_graph_adj[i].to_dense())
        #     # de
        #     explain_num += explain_graph_adj[i].to_dense().sum()
        #     original_num += hist_adj_list[i].to_dense().sum()
        #     update_sum += torch.sum(abs(update_list[i]))
        #     kl_sum += F.kl_div(explain_graph_adj[i].to_dense(),hist_adj_list[i].to_dense(),reduction='mean')
        # loss2 = explain_num / original_num 
        # loss3 = 0.001*update_sum / len(explain_graph_adj)
        # loss4 = kl_sum / len(explain_graph_adj)
        # print(loss1)
        # print(loss2)
        # print(loss3)
        # print(loss4)
        # de
        loss = loss1 
        if set_name=='TRAIN':
            c = 2
            reward = -loss.item() + c
        print('loss is',loss.item())
        if grad:
            self.optim_step(loss)
        torch.set_grad_enabled(True)
        # eval_measure = self.logger.log_epoch_done()
        eval_measure = []
        return eval_measure, nodes_embs,reward

    def predict(self,hist_adj_list,hist_ndFeats_list,node_indices,node_label,mask_list,s_idx,hist_hop,set_name,reward):
        nodes_embs = self.gcn(hist_adj_list,
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
        # print(gather_predictions.shape)
        # de
        return gather_predictions, nodes_embs

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


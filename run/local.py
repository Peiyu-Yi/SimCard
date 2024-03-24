#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2023/7/18 15:21
@desc:
"""
import pdb

import numpy as np
import pickle
import time
from tqdm import tqdm
import random
import torch

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

print('Start load dataloaders')
# start_time = time.time()
# with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/deep_96_euclidean/clusters_deep_96_euclidean.pkl', 'rb') as f:
#     clusters = pickle.load(f)
# centroids = []
# for cluster in tqdm(clusters):
#     centroids.append(np.mean(cluster, axis=0))   #  np.mean (cluster, axis=0)?
# end_time = time.time()
# print(f'succ clusters, cost:{end_time  - start_time}')

#start_time = time.time()
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/deep_96_euclidean/local_train_loaders.pkl', 'rb') as f:
    train_loaders = pickle.load(f)
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/deep_96_euclidean/local_test_loaders.pkl', 'rb') as f:
    test_loaders = pickle.load(f)
#end_time = time.time()
#print(f'succ data loaders, cost:{end_time  - start_time}')


#import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


queries_dimension = 96
hidden_num = 128


class Threshold_Model(nn.Module):

    def __init__(self):
        super(Threshold_Model, self).__init__()
        self.fc1 = nn.Linear(1, hidden_num)
        self.fc2 = nn.Linear(hidden_num, 1)

    def forward(self, threshold):
        t1 = F.relu(self.fc1(threshold))
        t2 = self.fc2(t1)
        return t2

class CNN_Model(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, pool_type, pool_size):
        super(CNN_Model, self).__init__()
        if pool_type == 0:
            pool_layer = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        elif pool_type == 1:
            pool_layer = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
        else:
            print ('CNN_Model Init Error, invalid pool_type {}'.format(pool_type))
            return
        self.layer = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            pool_layer)

    def forward(self, inputs):
        hid = self.layer(inputs)
#         print (hid.shape)
#         hid = F.relu(self.n3(hid))
#         hid = F.relu(self.n4(hid))
#         hid = self.norm2(hid)
#         print (hid.shape)
#         out2 = self.fc(hid.view(out1.shape[0], -1))
        return hid

#
class Output_Model(nn.Module):
    def __init__(self, inputs_dim):
        super(Output_Model, self).__init__()
        self.fc1 = nn.Linear(inputs_dim, hidden_num)
        self.fc2 = nn.Linear(hidden_num, 1)

    def forward(self, queries, threshold):
        out1 = F.relu(self.fc1(queries))
        out2 = out1 + threshold
#         print ('out2: {0}, threshold: {1}'.format(out2.shape, threshold.shape))
        out3 = self.fc2(out2)
        return out3

# class Output_Model(nn.Module):
#     def __init__(self, inputs_dim):
#         super(Output_Model, self).__init__()
#         self.fc1 = nn.Linear(inputs_dim, hidden_num)
#         self.fc2 = nn.Linear(hidden_num+1, 1)
#
#     def forward(self, queries, threshold):
#         out1 = F.relu(self.fc1(queries))
#         out2 = torch.cat((out1, threshold), dim=1)
#         #out2 = out1 + threshold
# #         print ('out2: {0}, threshold: {1}'.format(out2.shape, threshold.shape))
#         out3 = self.fc2(out2)
#         return out3


# class Model(nn.Module):
#
#     def __init__(self):
#         super(Model, self).__init__()
#         self.nn1 = nn.Linear(queries_dimension+1, hidden_num)
#         self.n1 = nn.Linear(hidden_num, hidden_num)
#         self.n2 = nn.Linear(hidden_num, hidden_num)
# #         self.n3 = nn.Linear(hidden_num, hidden_num)
# #         self.n4 = nn.Linear(hidden_num, hidden_num)
#         self.nn2 = nn.Linear(hidden_num, 1)
#
#     def forward(self, queries, threshold):
#         out1 = F.relu(self.nn1(torch.cat([queries, threshold],1)))
#         hid = out1
#         hid = F.relu(self.n1(hid))
#         hid = F.relu(self.n2(hid))
# #         hid = F.relu(self.n3(hid))
# #         hid = F.relu(self.n4(hid))
# #         hid = self.norm2(hid)
#         out2 = self.nn2(hid)
#         return out2


# def loss_fn(estimates, targets, mini, maxi):
#     est = unnormalize(estimates, mini, maxi)
#     print (torch.cat((est, targets), 1))
#     return F.mse_loss(est, targets)

def l1_loss(estimates, targets, eps=1e-5):
    estimates = torch.exp(estimates)
    mape = torch.mean(torch.abs((estimates - targets) * 1.0 / (targets + 0.000001)))
    qerror = 0.0
    for i in range(estimates.shape[0]):
        if estimates[i] > targets[i] + 0.1:
            qerror += ((estimates[i] / (targets[i] + 0.1)))
        else:
            qerror += (((targets[i] + 0.1) / estimates[i]))
    return qerror / estimates.shape[0] + mape


# def mse_loss(estimates, targets, eps=1e-5):
# #     print (torch.cat((estimates, targets), 1))
#     return F.mse_loss(estimates, torch.log(targets))

def print_loss(estimates, targets):
    esti = torch.exp(estimates)
#     print (torch.cat((estimates, esti, targets), 1))
    qerror = []
    for i in range(esti.shape[0]):
        if esti[i] > targets[i] + 0.1:
            qerror.append((esti[i] / (targets[i] + 0.1)).item())
        else:
            qerror.append(((targets[i] + 0.1) / esti[i]).item())

    return F.mse_loss(esti, targets), np.mean(qerror), np.max(qerror)


#from random import sample
#from tqdm import tqdm

def construct_model():
    #cost_times = None
    errors = []
    next_cnn_parameterss = []
    next_cnn_modelss = []
    next_output_models = []
    threshold_models = []

    times = []
    for clus in tqdm(range(100)):  # cluster 20 24 97 MSE error inf nan
        start_time = time.time()

        print ('Begin Cluster: {}'.format(clus))
        idx = clus
        train = train_loaders[idx]  #
        test = test_loaders[idx]
        #mini = min_cards[idx]
        #maxi = max_cards[idx]
        prev_best_error = 100000.0
        cnn_parameters, cnn_models = [], []
        episode = 5
        queries_dimension = 96
        threshold_model = Threshold_Model()
        error, next_cnn_parameters, next_cnn_models,next_output_model = select_best_layer(prev_best_error, cnn_parameters, cnn_models, threshold_model, train, test, episode, queries_dimension,idx)
        saved_error, saved_next_cnn_parameters, saved_next_cnn_models,saved_next_output_model = error, next_cnn_parameters, next_cnn_models,next_output_model
        while error is not None:
            saved_error, saved_next_cnn_parameters, saved_next_cnn_models,saved_next_output_model = error, next_cnn_parameters, next_cnn_models,next_output_model
            #print ('Cluster: {}, Error: {}, CNN Layer Num: {}, Added CNN Layer: {}'.format(clus, error, len(next_cnn_parameters), next_cnn_parameters[-1]))
            error, next_cnn_parameters, next_cnn_models,next_output_model = select_best_layer(error, next_cnn_parameters, next_cnn_models, threshold_model, train, test, episode, queries_dimension,idx)

        #pdb.set_trace()
        #print(f'+++++++++, clus: {clus}, len next_cnn_model : {len(saved_next_cnn_models)}++++++++++++++++++++++++++++++++++++++++++')
        errors.append(saved_error)
        next_cnn_parameterss.append(saved_next_cnn_parameters)
        next_cnn_modelss.append(saved_next_cnn_models)
        next_output_models.append(saved_next_output_model)
        threshold_models.append(threshold_model)

        end_time = time.time()
        times.append(end_time - start_time)


    return errors, next_cnn_parameterss, next_cnn_modelss, next_output_models, threshold_models, np.array(times)


class TunableParameters():

    def __init__(self, out_channel, kernel_size, stride, padding, pool_size, pool_type):
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_size = pool_size
        self.pool_type = pool_type

    def __repr__(self):
        return str(self.out_channel) +' '+ str(self.kernel_size) +' '+ str(self.stride) +' '+ str(self.padding) +' '+ str(self.pool_size) +' '+ str(self.pool_type)

    def __str__(self):
        return str(self.out_channel) +' '+ str(self.kernel_size) +' '+ str(self.stride) +' '+ str(self.padding) +' '+ str(self.pool_size) +' '+ str(self.pool_type)


def select_best_layer(prev_best_error, cnn_parameters, cnn_models, threshold_model, train, test, episode, queries_dimension, clus_id):
    #print ('Input Model Size: {}'.format(len(cnn_parameters)))
    if len(cnn_parameters) > 0:
        in_channel = cnn_parameters[-1].out_channel
    else:
        in_channel = 1
    in_size = queries_dimension
    for para in cnn_parameters:
        in_size = int((int((in_size - para.kernel_size + 2*(para.padding)) / para.stride) + 1) / para.pool_size)
        #print(para.kernel_size, para.padding, para.stride, para.pool_size, in_size)

    if in_size < 10 or len(cnn_parameters) > 5:
        #print('=======================None None None None=========================================')
        return None, None, None, None

    current_paras = []
    current_paras.append(TunableParameters(8,10,1,3,10,0))
    current_paras.append(TunableParameters(4,5,3,2,5,0))
#     current_paras.append(TunableParameters(4,3,1,0,3,0))
    current_paras.append(TunableParameters(2,2,1,0,2,0))

#     for out_channel in [2,4,8]:
#         for kernel_size in [2,4,8]:
#             for stride in range(1, min(4, kernel_size)):
#                 for padding in [0,2]:
#                     for pool_size in [kernel_size,]:
#                         for pool_type in [0,]:
#                             current_paras.append(TunableParameters(out_channel,kernel_size,stride,padding,pool_size,pool_type))
    #print ('Group of parameters: {}'.format(len(current_paras)))
    next_cnn_models = []
    next_cnn_parameters = []
    next_output_model = None

    #cost_times = None
#     current_paras = sample(current_paras, 2)
    for para in current_paras:
        #print (para)
        in_size_local = int((int((in_size - para.kernel_size + 2*(para.padding)) / para.stride) + 1) / para.pool_size)
        if in_size_local < 10:
            continue
        #print (in_size_local, para.out_channel)
        output_model = Output_Model(in_size_local * para.out_channel)
        added_cnn_layer = CNN_Model(in_channel, para.out_channel, para.kernel_size, para.stride, para.padding, para.pool_type, para.pool_size)
        paras = [{"params": model.parameters()} for model in cnn_models]
        paras.append({"params": added_cnn_layer.parameters()})
        paras.append({"params": threshold_model.parameters()})
        paras.append({"params": output_model.parameters()})

        opt = optim.Adam(paras, lr=0.001)
        new_cnn_models = []
        for model in cnn_models:
            new_cnn_models.append(model)
        new_cnn_models.append(added_cnn_layer)
        error = train_and_test(new_cnn_models, threshold_model, output_model, opt, train, test, episode)

        # if clus_id == 20 or clus_id == 24 or clus_id == 96 or clus_id == 97:
        if np.isnan(error) or np.isinf(error) or error > 1000000:

            print(f'!!!!!This is cluster id {clus_id}, q_mean error is nan or inf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #print(f'############ERROR:{error} < prev:{prev_best_error} - 0.1 ##############')
            #prev_best_error = error
            new_cnn_parameters = []
            for para_old in cnn_parameters:
                new_cnn_parameters.append(para_old)
            next_output_model = output_model
            new_cnn_parameters.append(para)
            #print('Update layer: {}'.format(len(new_cnn_parameters)))
            next_cnn_parameters = new_cnn_parameters
            next_cnn_models = new_cnn_models
        else:
            if error < prev_best_error - 0.1:
                #print(f'############ERROR:{error} < prev:{prev_best_error} - 0.1 ##############')
                prev_best_error = error
                new_cnn_parameters = []
                for para_old in cnn_parameters:
                    new_cnn_parameters.append(para_old)
                next_output_model = output_model
                new_cnn_parameters.append(para)
                #print ('Update layer: {}'.format(len(new_cnn_parameters)))
                next_cnn_parameters = new_cnn_parameters
                next_cnn_models = new_cnn_models
    if len(next_cnn_models) == 0:
        #print("===================next_cnn_model is None None None None None============================")
        return None, None, None, None
    return prev_best_error, next_cnn_parameters, next_cnn_models, next_output_model


def only_test(cnn_models, threshold_model, output_model, test):
    for model in cnn_models:
        model.eval()
    threshold_model.eval()
    output_model.eval()
    q_errors = []
    for batch_idx, (queries, _, thresholds, targets) in enumerate(test):
        queries = Variable(queries)
        thresholds = Variable(thresholds)
        targets = Variable(targets)

        queries = queries.unsqueeze(2).permute(0,2,1)
        for model in cnn_models:
            queries = model(queries)
        threshold = threshold_model(thresholds)
        queries = queries.view(queries.shape[0], -1)
        estimates = output_model(queries, threshold)

        #loss = l1_loss(estimates, targets)

        esti = torch.exp(estimates)
        for i in range(esti.shape[0]):
            if esti[i] > targets[i] + 0.1:
                q_errors.append((esti[i] / (targets[i] + 0.1)).item())
            else:
                q_errors.append(((targets[i] + 0.1) / esti[i]).item())
    mean = np.mean(q_errors)
    percent90 = np.percentile(q_errors, 90)
    percent95 = np.percentile(q_errors, 95)
    percent99 = np.percentile(q_errors, 99)
    median = np.median(q_errors)
    maxi = np.max(q_errors)
    print ('Testing: Mean Error {}, Median Error {}, 90 Percent {}, 95 Percent {}, 99 Percent {}, Max Percent {}'
           .format(mean, median, percent90, percent95, percent99, maxi))


def train_and_test(cnn_models, threshold_model, output_model, opt, train, test, episode):
    print ('size: {}'.format(len(train)))
    test_errors = []
    #cost_times = []
    for e in range(episode):
        #start_epoch_time = time.time()
        for model in cnn_models:
            model.train()
        threshold_model.train()
        output_model.train()
        for batch_idx, (queries, _, thresholds, targets) in enumerate(train):

            #pdb.set_trace()
    #         print (torch.cat((queries, thresholds), 1)[0])
            queries = Variable(queries)
            thresholds = Variable(thresholds)
            targets = torch.where(targets == 0, 1, targets)
            targets = Variable(targets)
    #         print (targets)
            opt.zero_grad()
            queries = queries.unsqueeze(2).permute(0,2,1)
            for model in cnn_models:
                queries = model(queries)
            threshold = threshold_model(thresholds)
            queries = queries.view(queries.shape[0], -1)
            estimates = output_model(queries, threshold)

            loss = l1_loss(estimates, targets)

            loss.backward()
            opt.step()
            for p in model.parameters():
                p.data.clamp_(-10, 10)
            next(threshold_model.fc1.parameters()).data.clamp_(0)  # 权重参数限制为非负值
            next(threshold_model.fc2.parameters()).data.clamp_(0)
            next(output_model.fc2.parameters()).data.clamp_(0)
#             if batch_idx % 100 == 0:
#                 print('Training: Iteration {0}, Batch {1}, Loss {2}'.format(e, batch_idx, loss.item()))
#                 print(cnn_models[0].layer[0].weight.grad)
        for model in cnn_models:
            model.eval()
        threshold_model.eval()
        output_model.eval()
        test_loss = 0.0
        mse_error = 0.0
        q_mean = 0.0
        q_max = 0.0
        for batch_idx, (queries, _, thresholds, targets) in enumerate(test):
            if batch_idx % 100 != 0:
                continue

            queries = Variable(queries)
            thresholds = Variable(thresholds)
            targets = torch.where(targets == 0, 1, targets)
            targets = Variable(targets)

            queries = queries.unsqueeze(2).permute(0,2,1)
            for model in cnn_models:
                queries = model(queries)
            threshold = threshold_model(thresholds)
            queries = queries.view(queries.shape[0], -1)
            estimates = output_model(queries, threshold)

            loss = l1_loss(estimates, targets)
            mse, qer_mean, qer_max = print_loss(estimates, targets)

            #print(f"mse: {mse}, qer_mean:{qer_mean}, qer_max:{qer_max}")
            #pdb.set_trace()

            test_loss += loss.item()
            mse_error += mse.item()
            q_mean += qer_mean
            if qer_max > q_max:
                q_max = qer_max
        test_loss /= len(test)
        mse_error /= len(test)
        q_mean /= len(test)
        test_errors.append(q_mean)
        print ('Testing: Iteration {0}, Loss {1}, MSE_error {2}, Q_error_mean {3}, Q_error_max {4}'.format(e, test_loss, mse_error, q_mean, q_max))

        #end_epoch_time = time.time()
        #cost_times.append(end_epoch_time - start_epoch_time)
        #print(f"####################epoch time cost : {end_epoch_time - start_epoch_time}########################################")

    return np.mean(test_errors[-3:])
    #return np.mean(test_errors)

local_start_time = time.time()
errors, next_cnn_parameterss, next_cnn_modelss, next_output_models, threshold_models, times_array = construct_model()
local_end_time = time.time()
cost_time = local_end_time - local_start_time

np.save('/research/remote/petabyte/users/peiyu/SimCard/cost_time/deep_96_euclidean_local_time.npy', times_array)
print(f'average of clusters: {np.average(times_array)}')
# 打开文件并写入值
with open("/research/remote/petabyte/users/peiyu/SimCard/cost_time/local_train_model_time.txt", "a") as file:
    file.write('deep_96_euclidean:' + str(cost_time) + "\n")
print(f'total local time: {cost_time}')

#print(f'average epoch cost time: {np.mean(cost_times)} ')



print(f"start save cnn hyperpara")
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/deep_96_euclidean/saved_models_s2/cnn_hyper_parameters.hyperpara', 'w') as handle:
    for idx in tqdm(range(100)):
        handle.write(';'.join(str(x) for x in next_cnn_parameterss[idx]))
        handle.write('\n')
print(f"end save cnn hyperpara")

print(f"start save cnn local models")
for idx in tqdm(range(100)):
    states = {}
    for idd, cnn_model in enumerate(next_cnn_modelss[idx]):
        states['cnn_model_state_dict_' + str(idd)] = cnn_model.state_dict()
    states['threshold_model_state_dict'] = threshold_models[idx].state_dict()
    states['output_model_state_dict'] = next_output_models[idx].state_dict()
    torch.save(states, '/research/remote/petabyte/users/peiyu/SimCard/dataset/deep_96_euclidean/saved_models_s2/local_deep_96_euclidean_cluster_' + str(idx) + '.model')
print(f"end save cnn local models")

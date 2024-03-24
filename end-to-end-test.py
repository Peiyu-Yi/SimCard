#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2023/7/18 20:22
@desc:
"""
import pickle
from tqdm import tqdm
from sklearn.metrics import *
import pdb
#from torchsummary import summary

with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/youtube_1770_cosine/final_test_loaders.pkl', 'rb') as f:
    test_loader = pickle.load(f)

f.close()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
input_dimension = 1770
cluster_dimension = 100
hidden_num = 256
output_num = 100


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.nn1 = nn.Linear(input_dimension, hidden_num)
        self.nn2 = nn.Linear(hidden_num, hidden_num)
#         self.nn3 = nn.Linear(hidden_num, hidden_num)

        self.dist1 = nn.Linear(cluster_dimension, hidden_num)
        self.dist2 = nn.Linear(hidden_num, hidden_num)

        self.nn4 = nn.Linear(hidden_num, hidden_num)
        self.nn5 = nn.Linear(hidden_num, output_num)

        self.thres1 = nn.Linear(1, hidden_num)
        self.thres2 = nn.Linear(hidden_num, 1)

    def forward(self, x, distances, thresholds):
        out1 = F.relu(self.nn1(x))
        out2 = F.relu(self.nn2(out1))
#         out3 = F.relu(self.nn3(out2))
#         print (distances.shape)
        distance1 = F.relu(self.dist1(distances))
        distance2 = F.relu(self.dist2(distance1))

        thresholds_1 = F.relu(self.thres1(thresholds))
        thresholds_2 = self.thres2(thresholds_1)

        out4 = F.relu(self.nn4((out2 + distance2) / 2))
        out5 = self.nn5(out4 + thresholds_2)

        probability = torch.sigmoid(out5)
        return probability


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.nn1 = nn.Linear(input_dimension, hidden_num)
#         self.nn2 = nn.Linear(hidden_num, hidden_num)
#         #         self.nn3 = nn.Linear(hidden_num, hidden_num)
#
#         self.dist1 = nn.Linear(cluster_dimension, hidden_num)
#         self.dist2 = nn.Linear(hidden_num, hidden_num)
#
#         self.nn4 = nn.Linear(hidden_num*2, hidden_num)
#         self.nn5 = nn.Linear(hidden_num+1, output_num)
#
#         self.thres1 = nn.Linear(1, hidden_num)
#         self.thres2 = nn.Linear(hidden_num, 1)
#
#     def forward(self, x, distances, thresholds):
#         out1 = F.relu(self.nn1(x))
#         out2 = F.relu(self.nn2(out1))
#         #         out3 = F.relu(self.nn3(out2))
#         #         print (distances.shape)
#         distance1 = F.relu(self.dist1(distances))
#         distance2 = F.relu(self.dist2(distance1))
#
#         thresholds_1 = F.relu(self.thres1(thresholds))
#         thresholds_2 = self.thres2(thresholds_1)
#
#         out4 = torch.relu(self.nn4(torch.cat((out2, distance2), dim=1)))
#         #out4 = F.relu(self.nn4((out2 + distance2) / 2))
#         out5 = self.nn5(torch.cat((out4, thresholds_2), dim=1))
#         #out5 = self.nn5(out4 + thresholds_2)
#
#         probability = torch.sigmoid(out5)
#         return probability

queries_dimension = 1770
hidden_num_2 = 128

class Threshold_Model(nn.Module):

    def __init__(self):
        super(Threshold_Model, self).__init__()
        self.fc1 = nn.Linear(1, hidden_num_2)
        self.fc2 = nn.Linear(hidden_num_2, 1)

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
        return hid

class Output_Model(nn.Module):

    def __init__(self, inputs_dim):
        super(Output_Model, self).__init__()
        self.fc1 = nn.Linear(inputs_dim, hidden_num_2)
        self.fc2 = nn.Linear(hidden_num_2, 1)

    def forward(self, queries, threshold):
        out1 = F.relu(self.fc1(queries))
        out2 = out1 + threshold
        out3 = self.fc2(out2)
        return out3

# class Output_Model(nn.Module):
#     def __init__(self, inputs_dim):
#         super(Output_Model, self).__init__()
#         self.fc1 = nn.Linear(inputs_dim, hidden_num_2)
#         self.fc2 = nn.Linear(hidden_num_2+1, 1)
#
#     def forward(self, queries, threshold):
#         out1 = F.relu(self.fc1(queries))
#         out2 = torch.cat((out1, threshold), dim=1)
#         #out2 = out1 + threshold
# #         print ('out2: {0}, threshold: {1}'.format(out2.shape, threshold.shape))
#         out3 = self.fc2(out2)
#         return out3


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


hyper_parameterss = []
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/youtube_1770_cosine/saved_models/cnn_hyper_parameters.hyperpara', 'r') as handle:
    for paras in handle.readlines():
        hyper_parameters = []
        for para in paras.split(';'):
            para = para.split(' ')
            hyper_parameters.append(TunableParameters(int(para[0]), int(para[1]), int(para[2]),
                                                      int(para[3]), int(para[4]), int(para[5])))
        hyper_parameterss.append(hyper_parameters)

print("succ load hyper_parameters")

cnn_modelss = []
threshold_models = []
output_models = []
for idx in range(100):
    states = torch.load('/research/remote/petabyte/users/peiyu/SimCard/dataset/youtube_1770_cosine/saved_models/local_youtube_1770_cosine_cluster_' + str(idx) + '.model')
    hyper_para = hyper_parameterss[idx]
    cnn_models = []
    weights = [None for _ in range(len(hyper_para))]
    for key, value in states.items():
        if key != 'threshold_model_state_dict' and key != 'output_model_state_dict':
#             print (key)
            layer_id = int(key.split('_')[-1])
#             print (layer_id)
            weights[layer_id] = value
    in_channel = 1
    in_size = queries_dimension
    for weight_idx, weight in enumerate(weights):
        hyper = hyper_para[weight_idx]
        cnn_model = CNN_Model(in_channel, hyper.out_channel, hyper.kernel_size,
                              hyper.stride, hyper.padding, hyper.pool_type, hyper.pool_size)
        in_size = int((int((in_size - hyper.kernel_size + 2*(hyper.padding)) / hyper.stride) + 1) / hyper.pool_size)
        in_channel = hyper.out_channel
        cnn_model.load_state_dict(weight)
        cnn_model.eval()
        cnn_models.append(cnn_model)
    cnn_modelss.append(cnn_models)

    threshold_model_state_dict = states['threshold_model_state_dict']
    threshold_model = Threshold_Model()
    threshold_model.load_state_dict(threshold_model_state_dict)
    threshold_model.eval()
    threshold_models.append(threshold_model)

    output_model_state_dict = states['output_model_state_dict']
    output_model = Output_Model(in_size * in_channel)
    output_model.load_state_dict(output_model_state_dict)
    output_model.eval()
    output_models.append(output_model)

print("succ generate cnn_models, threshold_models, output_models")

def get_local_cardinality(cnn_models, threshold_model, output_model, queries, thresholds):
    queries = queries.unsqueeze(2).permute(0,2,1)
    for model in cnn_models:
        queries = model(queries)
    threshold = threshold_model(thresholds)
    queries = queries.view(queries.shape[0], -1)
    estimates = output_model(queries, threshold)
    esti = torch.exp(estimates)
    return esti.detach()

def print_qerror(estimates, targets):
    qerror = []
    for i in range(estimates.shape[0]):
        left = estimates[i] + 1
        right = targets[i] + 1
        if left > right:
            qerror.append((left / right).item())
        else:
            qerror.append((right / left).item())
    return qerror

# for cluster_id in range(100):
#     print (cluster_id)
#     total = 0.0
#     for batch_idx, (features, thresholds, distances, targets, cards) in enumerate(test_loader):
#         estimates = get_local_cardinality(cnn_modelss[cluster_id], threshold_models[cluster_id],
#                                              output_models[cluster_id], features, thresholds)
#         print (estimates.shape, cards[:,cluster_id].shape)
#         print (torch.cat((estimates, cards[:,cluster_id].unsqueeze(1)), axis=1))
#         errors = print_qerror(estimates, cards[:,cluster_id])
#         total += np.mean(errors)
#         print (np.mean(errors))
#     print (total / len(test_loader))

def calculate_model_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_size_MB = total_size_bytes / (1024**2)

    return total_params, total_size_MB

print("start testing")
model = Model()
model.load_state_dict(torch.load('/research/remote/petabyte/users/peiyu/SimCard/dataset/youtube_1770_cosine/saved_models/global_youtube_1770_cosine_punish_query_threshold_monotonic.model'))
model.eval()

#summary(model, ())
# 计算模型参数数量和内存大小
global_total_params, global_total_size_MB = calculate_model_memory(model)

print(f"global parameters: {global_total_params}")
print(f"global model size: {global_total_size_MB:.2f} MB")

local_params = 0
local_size_MB = 0
for cluster_id in range(100):
    model_cnn = cnn_modelss[cluster_id]
    model_threshold = threshold_models[cluster_id]
    model_output = output_models[cluster_id]

    for model in model_cnn:
        params, size_MB = calculate_model_memory(model)
        local_params += params
        local_size_MB += size_MB

    params, size_MB = calculate_model_memory(model_threshold)
    local_params += params
    local_size_MB += size_MB

    params, size_MB = calculate_model_memory(model_output)
    local_params += params
    local_size_MB += size_MB
print(f"local parameters: {local_params}")
print(f"local model size: {local_size_MB:.2f} MB")

print(f"total parameters: {global_total_params + local_params}")
print(f"total model size: {global_total_size_MB + local_size_MB:.2f} MB")


test_loss = 0.0
precision = 0.0
recall = 0.0
miss_rate = 0.0
estimatesss = []
q_errors = []
mses = []
maes = []
mapes = []

predictions = None
true_cards = None


def mean_absolute_percentage_error(labels, predictions):
    return np.mean(np.abs((predictions - labels) * 1.0 / (labels + 0.000001))) * 100


def qerror_minmax(labels, predictions):
    max_values = np.maximum(labels, predictions)
    min_values = np.minimum(labels, predictions)
    q_error = (max_values+0.6) / (min_values+0.6)
    return np.mean(q_error)


def qerror(labels, predictions):
    return (predictions+0.6) / (labels+0.6)


def eval(predictions, labels):
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    mape = mean_absolute_percentage_error(labels, predictions)
    q_error_minmax = qerror_minmax(labels, predictions)

    q_error = qerror(labels, predictions)
    underestimate_ratio = np.sum(q_error < 1) / len(q_error)
    overestimate_ratio = np.sum(q_error > 1) / len(q_error)
    average_overestimate = np.mean(q_error[q_error > 1]) - 1

    return (mse, mae, mape, q_error_minmax, underestimate_ratio, overestimate_ratio, average_overestimate)


def find_nan_inf(arrayA, arrayB):
    nan_inf_mask_A = np.logical_or(np.isnan(arrayA), np.isinf(arrayA))
    nan_inf_mask_B = np.logical_or(np.isnan(arrayB), np.isinf(arrayB))
    filtered_arrayA = arrayA[~nan_inf_mask_A]
    filtered_arrayB = arrayB[~nan_inf_mask_B]
    return filtered_arrayA, filtered_arrayB

# count=0
for batch_idx, (features, thresholds, distances, targets, cards) in tqdm(enumerate(test_loader)):
    if batch_idx % 100 == 0:
        print (batch_idx)
    estimates = model(features, distances, thresholds)
    global_indicator = (estimates >= 0.5).float()
    local_estimates = []
    for cluster_id in range(100):
        local_estimates.append(get_local_cardinality(cnn_modelss[cluster_id], threshold_models[cluster_id],
                                                     output_models[cluster_id], features, thresholds))
    localss = torch.cat(local_estimates, dim = 1)

    # nan_column_indices = np.where(np.isnan(localss).any(axis=0))[0]
    # localss = np.delete(localss, nan_column_indices, axis=1)
    # global_indicator = np.delete(global_indicator, nan_column_indices, axis=1)

#     print (localss.shape, global_indicator.shape)
    cards_estimates = (localss * global_indicator).sum(dim=1).unsqueeze(1)


    cards = cards.sum(dim=1).unsqueeze(1)
    cards = torch.where(cards == 0, 1, cards)

    if batch_idx == 0:
        predictions = cards_estimates.numpy()
        true_cards = cards.numpy()
    else:
        predictions = np.concatenate((predictions, cards_estimates.numpy()), axis=0)
        true_cards = np.concatenate((true_cards, cards.numpy()), axis=0)

print(f'len testloaders: {len(test_loader)}')
print(f"prediction shape: {predictions.shape}")
print(f"true shape: {true_cards.shape}")
#pdb.set_trace()

#pdb.set_trace()
predictions = np.hstack(predictions)
true_cards = np.hstack(true_cards)
np.save('/research/remote/petabyte/users/peiyu/SimCard/saved_results/deep_true_card.npy', true_cards)
np.save('/research/remote/petabyte/users/peiyu/SimCard/saved_results/deep_est_card.npy', predictions)
print(f'max true cards:{np.max(true_cards)}')
print(f'max pre cards: {np.max(predictions)}')
print(f'min true cards:{np.min(true_cards)}')
print(f'min pre cards: {np.min(predictions)}')
#pdb.set_trace()
index_gt_10000 = np.where(predictions > np.max(true_cards))[0]
print(index_gt_10000.shape)

pre_10000 = np.delete(predictions, index_gt_10000)
true_10000 = np.delete(true_cards, index_gt_10000)
#pdb.set_trace()
print('Loss {}'.format(eval(pre_10000, true_10000)))

#pdb.set_trace()

max_values = np.maximum(true_10000, pre_10000)
min_values = np.minimum(true_10000, pre_10000)

q_errors = (max_values+0.6) / (min_values + 0.6)

mean = np.mean(q_errors)
median = np.median(q_errors)
maxi = np.max(q_errors)
min = np.min(q_errors)
percent25 = np.percentile(q_errors, 25)
percent50 = np.percentile(q_errors, 50)
percent75 = np.percentile(q_errors, 75)
percent90 = np.percentile(q_errors, 90)
percent95 = np.percentile(q_errors, 95)
percent99 = np.percentile(q_errors, 99)

print(f"num of test samples: {len(test_loader)}")
print('q_errors Testing:  Mean {}, Median {}, Max {}, Min {}, 25 Percent {}, 50 Percent {}, 75 Percent {}, 90 Percent {}, 95 Percent {}, 99 Percent {}'
       .format(mean, median, maxi, min, percent25, percent50, percent75, percent90, percent95, percent99))
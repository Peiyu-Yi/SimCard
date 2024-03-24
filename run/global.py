#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2023/7/18 17:24
@desc:
"""
#import numpy as np
import pickle
import time
from tqdm import tqdm
import random
import torch
import numpy as np

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

print('Start load dataloaders')
# with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/deep_96_euclidean/clusters_deep_96_euclidean.pkl', 'rb') as f:
#     clusters = pickle.load(f)
# centroids = []
# for cluster in tqdm(clusters):
#     centroids.append(np.mean(cluster, axis=0))   #  np.mean (cluster, axis=0)?
# end_time = time.time()
# print(f'succ clusters, cost:{end_time  - start_time}')

start_time = time.time()
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/deep_96_euclidean/global_train_loaders.pkl', 'rb') as f:
    train_loaders = pickle.load(f)
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/deep_96_euclidean/global_test_loaders.pkl', 'rb') as f:
    test_loaders = pickle.load(f)
end_time = time.time()
print(f'succ data loaders, cost:{end_time  - start_time}')

#import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms

input_dimension = 96
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


# class Model_concate(nn.Module):
#     def __init__(self):
#         super(Model_concate, self).__init__()
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


def loss_fn(estimates, targets, cards):
    punish_idx = (estimates < 0.5).float()
    return F.mse_loss(estimates, targets) + 0.02 * torch.log(((0.5 - estimates) * cards * punish_idx).mean() + 1.0)


def loss_fn2(estimates, targets, cards):
    punish_idx = (estimates < 0.5).float()
    min_v, _ = torch.min(cards, dim=1)
    max_v, _ = torch.max(cards, dim=1)
    min_v = min_v.unsqueeze(dim=1)
    max_v = max_v.unsqueeze(dim=1)
    range_v = max_v - min_v
    normalized_cards = (cards - min_v) / (range_v + 0.01)
    loss = ((F.relu(estimates - targets) + F.relu(targets - estimates) * (normalized_cards + 1.0)) ** 2).sum(
        dim=1).mean()
    return loss


def print_loss(estimates, targets, cards):
    true_positive = 0.0
    true_negative = 0.0
    false_positive = 0.0
    false_negative = 0.0
    num_elements = estimates.shape[1]
    for est, tar in zip(estimates, targets):
        for i in range(num_elements):
            if est[i] < 0.5 and tar[i] == 0:
                true_negative += 1
            elif est[i] < 0.5 and tar[i] == 1:
                false_negative += 1
            elif est[i] >= 0.5 and tar[i] == 0:
                false_positive += 1
            else:
                true_positive += 1
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    total_card = cards.sum(dim=1)
#     print ('total_card: ', total_card.shape)
    miss_card = torch.FloatTensor([cards[i][((estimates[i] < 0.5).nonzero())].sum() for i in range(cards.shape[0])])
#     print ('miss_card: ', miss_card.shape)
    miss_rate = (miss_card / (total_card + 0.1)).mean()
    return precision, recall, miss_rate

model = Model()
opt = optim.Adam(model.parameters(), lr=0.001)

print('Start Training Global Model')

global_start_time = time.time()
for e in tqdm(range(1)):
    model.train()
    for batch_idx, (features, thresholds, distances, targets, cards) in enumerate(train_loaders):
        x = Variable(features)
        y = Variable(targets.unsqueeze(1))
        z = Variable(thresholds)
        dists = Variable(distances)
        opt.zero_grad()
        estimates = model(x, dists, z)
        cards = torch.where(cards == 0, 1, cards)
        loss = loss_fn(estimates, targets, cards)
        if batch_idx % 1000 == 0:
            print('Training: Iteration {0}, Batch {1}, Loss {2}'.format(e, batch_idx, loss.item()))
        loss.backward()
        opt.step()
        next(model.thres1.parameters()).data.clamp_(0)
        next(model.thres2.parameters()).data.clamp_(0)

    # model.eval()
    # test_loss = 0.0
    # precision = 0.0
    # recall = 0.0
    # miss_rate = 0.0
    # for batch_idx, (features, thresholds, distances, targets, cards) in enumerate(test_loaders):
    #     x = Variable(features)
    #     y = Variable(targets.unsqueeze(1))
    #     z = Variable(thresholds)
    #     dists = Variable(distances)
    #     estimates = model(x, dists, z)
    #     loss = loss_fn(estimates, targets, cards+1)
    #     test_loss += loss.item()
    #     prec, rec, miss = print_loss(estimates, targets, cards+1)
    #     precision += prec
    #     recall += rec
    #     miss_rate += miss
    #     if batch_idx % 100 == 0:
    #         print ('Testing: Iteration {0}, Batch {1}, Loss {2}, Precision {3}, Recall {4}, Miss {5}'.format(e, batch_idx, loss.item(), prec, rec, miss))
    # test_loss /= len(test_loaders)
    # precision /= len(test_loaders)
    # recall /= len(test_loaders)
    # miss_rate /= len(test_loaders)
    # print ('Testing: Loss {0}, Precision {1}, Recall {2}, Miss {3}'.format(test_loss, precision, recall, miss_rate))
global_end_time = time.time()

cost_time = global_end_time - global_start_time
# 打开文件并写入值
with open("/research/remote/petabyte/users/peiyu/SimCard/cost_time/global_train_model_time.txt", "a") as file:
    file.write('deep_96_euclidean_1epoch_09:' + str(cost_time) + "\n")
print(cost_time)
print('Training Finished, Start Save Global Model')
torch.save(model.state_dict(), '/research/remote/petabyte/users/peiyu/SimCard/dataset/deep_96_euclidean/saved_models_s2/global_deep_96_euclidean_punish_query_threshold_monotonic.model')
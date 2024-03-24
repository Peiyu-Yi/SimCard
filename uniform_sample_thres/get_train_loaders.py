#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2023/7/31 10:28
@desc:
"""
import time
from uniform_utils import euclidean_dist, cosine_dist, calculate_consine_distance_matrix
import pickle
from tqdm import tqdm
import numpy as np
import sys
import torch.utils.data
import random

print("---------Start get train dataloaders--------------")
# clusters
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/youtube_1770_cosine/clusters_youtube_1770_cosine.pkl', 'rb') as f:
    clusters_points = pickle.load(f)
print('succ load clusters')

# train_dictionary
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/youtube_1770_cosine/train_dictionary.pkl', 'rb') as f:
    train_dictionary = pickle.load(f)
print('succ load train_dictionary')
f.close()


train_ground_truth_total = []
train_queries = np.array(list(train_dictionary.keys()))
print(f"shape of train queries: {train_queries.shape}")

# slot = 0.04675
# thres_max = 0.4657
# thres_min = 0.0001

# get train_ground_truth_total
print('--------------start get train total----------------')
start_time = time.time()
for clus in tqdm(range(100)):
    dataset = clusters_points[clus]  # N-d L
    ground_truth = []
    # distances between all queries and database
    distances = calculate_consine_distance_matrix(dataset, train_queries)
    # print(f"query与数据集的距离矩阵维度：{distances.shape}")

    sorted_indices = np.argsort(distances, axis=1)
    sorted_distences = np.take_along_axis(distances, sorted_indices, axis=1)
    # print(f"从小到大排序后的距离矩阵：{sorted_distences.shape}")

    for idxx, q in enumerate(train_queries):
        thresholds = train_dictionary.get(tuple(q))
        thres_min = np.min(thresholds)
        thres_max = np.max(thresholds)
        thresholds = np.linspace(thres_min, thres_max, 10)
        for idx, threshold in enumerate(thresholds):
            index = np.searchsorted(sorted_distences[idxx], threshold)
            ground_truth.append((clus, idxx, threshold, threshold, index))
    train_ground_truth_total.append(ground_truth)
end_time = time.time()
print(f'end get tain total, size:{sys.getsizeof(train_ground_truth_total)}, cost: {end_time - start_time}')

print('-------------------start get train level-----------------')
train_ground_truth_total_level = [[[] for _ in range(train_queries.shape[0])] for _ in range(100)]  #  train_queries 的数量
for clus in tqdm(range(100)):
    for t in train_ground_truth_total[clus]:
        train_ground_truth_total_level[t[0]][t[1]].append(t)
print(f'end get train level, size:{sys.getsizeof(train_ground_truth_total_level)}')

centroids = []
for cluster in tqdm(clusters_points):
    centroids.append(np.mean(cluster, axis=0))   #  np.mean (cluster, axis=0)?

print('-----------start get global train loader----------------------')
start_time = time.time()
train_features = []
train_thresholds = []
train_distances = []
train_targets = []
train_cards = []
#slot = 0.008
for query_id in tqdm(range(train_queries.shape[0])):
    query = train_queries[query_id]
    thresholds = train_dictionary.get(tuple(query))
    thres_min = np.min(thresholds)
    thres_max = np.max(thresholds)
    thresholds = np.linspace(thres_min, thres_max, 10)
    cardinality = [0 for _ in range(100)]
    distances2centroids = []
    for cc in centroids:
        #pdb.set_trace()
        distances2centroids.append(cosine_dist(train_queries[query_id], cc))
    for threshold_id, threshold in enumerate(thresholds):
        indicator = []
        cards = []
        for cluster_id in range(100):
            cardinality[cluster_id] = train_ground_truth_total_level[cluster_id][query_id][threshold_id][-1]
            if cardinality[cluster_id] > 0:
                indicator.append(1)
            else:
                indicator.append(0)
            cards.append(cardinality[cluster_id])
        feature = train_queries[query_id]
        train_features.append(feature)
        train_distances.append(distances2centroids)
        train_thresholds.append([threshold])
        train_targets.append(indicator)
        train_cards.append(cards)
batch_size = 128
train_loaders = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(np.array(train_features)), torch.FloatTensor(np.array(train_thresholds)), torch.FloatTensor(np.array(train_distances)), torch.FloatTensor(np.array(train_targets)), torch.FloatTensor(np.array(train_cards))), batch_size=batch_size, shuffle=True)
print(f'end get global train loader, size:{sys.getsizeof(train_loaders)}')
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/youtube_1770_cosine_uniform/global_train_loaders.pkl', 'wb') as f:
    pickle.dump(train_loaders, f, protocol=4)
f.close()
end_time = time.time()
print(f'end get global train loader, size:{sys.getsizeof(train_loaders)}, cost:{end_time-start_time}')


batch_size=128
print('--------------------start get local train loader---------------------')
start_time = time.time()
train_loaders = []
for cluster_id in tqdm(range(100)):
    train_queries_l = []
    train_distances_l = []
    train_thresholds_l = []
    train_targets_l = []
    for query_id in range(train_queries.shape[0]):
        query = train_queries[query_id]
        thresholds = train_dictionary.get(tuple(query))
        thres_min = np.min(thresholds)
        thres_max = np.max(thresholds)
        thresholds = np.linspace(thres_min, thres_max, 10)
        cardinality = 0
        for threshold_id, threshold in enumerate(thresholds):
            cardinality = train_ground_truth_total_level[cluster_id][query_id][threshold_id][-1]
            if cardinality > 0 and random.random() < 0.33:
                train_queries_l.append(train_queries[query_id])
                train_distances_l.append([cosine_dist(train_queries[query_id], centroids[cluster_id])])
                train_thresholds_l.append([threshold])
                train_targets_l.append([cardinality])
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(np.array(train_queries_l)),
                                       torch.FloatTensor(np.array(train_distances_l)),
                                       torch.FloatTensor(np.array(train_thresholds_l)),
                                       torch.FloatTensor(np.array(train_targets_l))), batch_size=batch_size, shuffle=True)

    train_loaders.append(train_loader)
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/youtube_1770_cosine_uniform/local_train_loaders.pkl', 'wb') as f:
    pickle.dump(train_loaders, f, protocol=4)
f.close()
end_time = time.time()
print(f'end get local train loader, size:{sys.getsizeof(train_loaders)}, cost:{end_time-start_time}')




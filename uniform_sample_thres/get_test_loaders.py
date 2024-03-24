#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2023/7/31 17:03
@desc:
"""
import numpy as np
import pickle
import time
from tqdm import tqdm
import sys
import random
import torch.utils.data
from uniform_utils import euclidean_dist, custom_searchsorted, calculate_consine_distance_matrix, cosine_dist
import pdb

print('-------------------------Start get test dataloaders------------------')
# clusters
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/youtube_1770_cosine/clusters_youtube_1770_cosine.pkl', 'rb') as f:
    clusters_points = pickle.load(f)
print('succ load clusters')
# test dictionary
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/youtube_1770_cosine/test_dictionary.pkl', 'rb') as f:
    test_dictionary = pickle.load(f)
print('succ load test_dictionary')

test_ground_truth_total = []
test_queries = np.array(list(test_dictionary.keys()))
print(f"shape of test queries: {test_queries.shape}")

# slot = 0.04
# thres_max = 0.4
# thres_min = 0.0
# thresholds = np.arange(thres_min, thres_max+slot, slot)
# print(thresholds)

# get test_ground_truth_total
print('start get test total')
start_time = time.time()
for clus in tqdm(range(100)):
    dataset = clusters_points[clus]  # N-d L
    #print(f"clus: {clus}, num :{dataset.shape[0]}")
    ground_truth = []
    # distances between all queries and database
    distances = calculate_consine_distance_matrix(dataset, test_queries)
    # print(f"query与数据集的距离矩阵维度：{distances.shape}")

    sorted_indices = np.argsort(distances, axis=1)
    sorted_distences = np.take_along_axis(distances, sorted_indices, axis=1)
    # print(f"从小到大排序后的距离矩阵：{sorted_distences.shape}")

    for idxx, q in enumerate(test_queries):
        thresholds = test_dictionary.get(tuple(q))
        thres_min = np.min(thresholds)
        thres_max = np.max(thresholds)
        thresholds = np.linspace(thres_min, thres_max, 10)
        for idx, threshold in enumerate(thresholds):
            index = np.searchsorted(sorted_distences[idxx], threshold)
            #index_slot = np.searchsorted(sorted_distences[idxx], threshold+slot)
            # if threshold == sorted_distences[idxx][0]:
            #     index = 1
            #     count = count + 1
            ground_truth.append((clus, idxx, threshold, threshold, index))
    test_ground_truth_total.append(ground_truth)
end_time = time.time()
print(f'end get test total, size:{sys.getsizeof(test_ground_truth_total)}, cost: {end_time - start_time}')

print('start get test level')
test_ground_truth_total_level = [[[] for _ in range(test_queries.shape[0])] for _ in range(100)]  #  test_queries 的数量
for clus in tqdm(range(100)):
    for t in test_ground_truth_total[clus]:
        test_ground_truth_total_level[t[0]][t[1]].append(t)
print('end get test level')

centroids = []
for cluster in tqdm(clusters_points):
    centroids.append(np.mean(cluster, axis=0))   #  np.mean (cluster, axis=0)?


print('------------------start get global test loaders----------------------')
start_time = time.time()
test_features = []
test_thresholds = []
test_distances = []
test_targets = []
test_cards = []
#slot = 0.002


for query_id in tqdm(range(test_queries.shape[0])):
    query = test_queries[query_id]
    thresholds = test_dictionary.get(tuple(query))
    thres_min = np.min(thresholds)
    thres_max = np.max(thresholds)
    thresholds = np.linspace(thres_min, thres_max, 10)
    cardinality = [0 for _ in range(100)]
    distances2centroids = []
    for cc in centroids:
        #pdb.set_trace()
        distances2centroids.append(cosine_dist(test_queries[query_id], cc))
    for threshold_id, threshold in enumerate(thresholds):
        indicator = []
        cards = []
        for cluster_id in range(100):
            cardinality[cluster_id] = test_ground_truth_total_level[cluster_id][query_id][threshold_id][-1]
            if cardinality[cluster_id] > 0:
                indicator.append(1)
            else:
                indicator.append(0)
            cards.append(cardinality[cluster_id])
        feature = test_queries[query_id]
        test_features.append(feature)
        test_distances.append(distances2centroids)
        test_thresholds.append([threshold])
        test_targets.append(indicator)
        test_cards.append(cards)
batch_size = 128
test_loaders = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(np.array(test_features)), torch.FloatTensor(np.array(test_thresholds)),
                                       torch.FloatTensor(np.array(test_distances)), torch.FloatTensor(np.array(test_targets)),
                                       torch.FloatTensor(np.array(test_cards))), batch_size=batch_size, shuffle=True)
print(f'end get global test loader, size:{sys.getsizeof(test_loaders)}')
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/youtube_1770_cosine_uniform/global_test_loaders.pkl', 'wb') as f:
    pickle.dump(test_loaders, f)
f.close()
end_time = time.time()
print(f'end get global test loader, size:{sys.getsizeof(test_loaders)}, cost:{end_time-start_time}')


print('----------------tart get local test loaders-----------------')
batch_size=128
start_time = time.time()
test_loaders = []
for cluster_id in tqdm(range(100)):
    test_queries_l = []
    test_distances_l = []
    test_thresholds_l = []
    test_targets_l = []
    for query_id in range(test_queries.shape[0]):
        query = test_queries[query_id]
        thresholds = test_dictionary.get(tuple(query))
        thres_min = np.min(thresholds)
        thres_max = np.max(thresholds)
        thresholds = np.linspace(thres_min, thres_max, 10)

        cardinality = 0
        for threshold_id, threshold in enumerate(thresholds):
            cardinality = test_ground_truth_total_level[cluster_id][query_id][threshold_id][-1]
            if cardinality > 0 and random.random() < 0.33:
                test_queries_l.append(test_queries[query_id])
                test_distances_l.append([cosine_dist(test_queries[query_id], centroids[cluster_id])])
                test_thresholds_l.append([threshold])
                test_targets_l.append([cardinality])
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(np.array(test_queries_l)),
                                       torch.FloatTensor(np.array(test_distances_l)),
                                       torch.FloatTensor(np.array(test_thresholds_l)),
                                       torch.FloatTensor(np.array(test_targets_l))), batch_size=batch_size, shuffle=True)

    test_loaders.append(test_loader)
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/youtube_1770_cosine_uniform/local_test_loaders.pkl', 'wb') as f:
    pickle.dump(test_loaders, f)
end_time = time.time()
print(f'end get local test loader, size:{sys.getsizeof(test_loaders)}, cost:{end_time-start_time}')
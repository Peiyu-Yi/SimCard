#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2023/9/7 10:28
@desc:
"""
import time
#from utils import calculate_consine_distance_matrix, cosine_dist
from utils import calculate_ed_distance_matrix, euclidean_dist
import pickle
from tqdm import tqdm
import numpy as np
#import sys
import torch.utils.data
from multiprocessing import Pool

print("---------Start get train dataloaders--------------")
# clusters
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/fasttext_300_euclidean/clusters_fasttext_300_euclidean.pkl', 'rb') as f:
    clusters_points = pickle.load(f)
print('succ load clusters')
f.close()

# train_dictionary
with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/fasttext_300_euclidean/train_dictionary.pkl', 'rb') as f:
    train_dictionary = pickle.load(f)
print('succ load train_dictionary')
f.close()



train_queries = np.array(list(train_dictionary.keys()))
print(train_queries.shape)
#train_queries = train_queries[60000:80000, :]

print('--------------start get train total----------------')
def run_proc(clus):
    dataset = clusters_points[clus]  # N-d L
    ground_truth = []
    # distances between all queries and database
    distances = calculate_ed_distance_matrix(dataset, train_queries)
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
    return ground_truth

# num_batch = 4
# batch_size = len(train_queries) // num_batch
# query_batches = [train_queries[i:i+batch_size] for i in range(0, len(train_queries), batch_size)]

train_ground_truth_total = []
start_time = time.time()
processes1 = []
pool1 = Pool(processes=40)
# for clus in range(100):
#     for batch in query_batches:
#         processes1.append(pool1.apply_async(run_proc, args=(clus, batch)))
for clus in range(100):
    processes1.append(pool1.apply_async(run_proc, args=(clus,)))
pool1.close()
pool1.join()
for i in processes1:
    train_ground_truth_total.append(i.get())

end_time = time.time()
time1_cost_time = end_time - start_time
print(f'times1: {time1_cost_time}')

# 初始化结果列表
# start_time = time.time()
# train_ground_truth_total = []
# # 遍历每一列
# for i in range(len(train_ground_truth_total0[0])):
#     column = []  # 存储当前列的元素
#     for j in range(len(train_ground_truth_total0)):
#         column.extend(train_ground_truth_total0[j][i])  # 将每行的元组扩展到列中
#     train_ground_truth_total.append(column)
# end_time = time.time()
# print(f'transpose time: {end_time - start_time}')

print('-------------------start get train level-----------------')
train_ground_truth_total_level = [[[] for _ in range(train_queries.shape[0])] for _ in range(100)]  #  train_queries 的数量
# def run_proc2(clus):
#     for t in train_ground_truth_total[clus]:
#         train_ground_truth_total_level[t[0]][t[1]].append(t)
# start_time = time.time()
# pool2 = Pool(processes=40)
# pool2.map(run_proc2, range(100))
# pool2.close()
# pool2.join()
# end_time = time.time()
# cost_time = end_time - start_time
# print(f'time2: {cost_time}')


times2 = []
for clus in tqdm(range(100)):
    start_time = time.time()
    for t in train_ground_truth_total[clus]:
        train_ground_truth_total_level[t[0]][t[1]].append(t)
    end_time = time.time()

    cost_time = end_time - start_time
    times2.append(cost_time)
times2_array = np.array(times2)
print(f'sum times2:{np.sum(times2_array)}, average times2:{np.average(times2_array)}')


centroids = []
for cluster in clusters_points:
    centroids.append(np.mean(cluster, axis=0))


batch_size=128
print('--------------------start get local train loader---------------------')
def run_proc3(cluster_id):
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
            if cardinality > 0:
                train_queries_l.append(train_queries[query_id])
                train_distances_l.append([euclidean_dist(train_queries[query_id], centroids[cluster_id])])
                train_thresholds_l.append([threshold])
                train_targets_l.append([cardinality])
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(np.array(train_queries_l)),
                                       torch.FloatTensor(np.array(train_distances_l)),
                                       torch.FloatTensor(np.array(train_thresholds_l)),
                                       torch.FloatTensor(np.array(train_targets_l))), batch_size=batch_size, shuffle=True)
    return train_loader

start_time = time.time()
train_loaders2 = []
processes3 = []
pool3 = Pool(processes=40)
for clus in range(100):
    processes3.append(pool3.apply_async(run_proc3, args=(clus,)))
pool3.close()
pool3.join()
for i in processes3:
    train_loaders2.append(i.get())
end_time = time.time()
time3_cost_time = end_time - start_time
print(f'time3: {time3_cost_time}')

total_time = time1_cost_time + np.sum(times2_array) + time3_cost_time
print(f'total time: {total_time}')
# 打开文件并写入值
with open("/research/remote/petabyte/users/peiyu/SimCard/cost_time/local_prepare_data_time.txt", "a") as file:
    file.write('fasttext_300_euclidean_2:' + str(total_time) + "\n")

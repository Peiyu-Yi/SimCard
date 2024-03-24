#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2023/7/25 16:32
@desc:
"""
import pickle
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import random

fix_seed = 2023
random.seed(fix_seed)
np.random.seed(fix_seed)

print('start loading datasets')
# datasets = np.load('/research/remote/petabyte/users/peiyu/TSCE/data/seismic/seismic_1m_wo_nan.npy')
datasets = np.load('/research/remote/petabyte/users/peiyu/TSCE/data/deep/deep_10m.npy')
# datasets = np.load('/research/remote/petabyte/users/s3852583/dataset4simcard/youtube_originalData.npy')
# print(f'shape: {datasets.shape}')

start_time = time.time()
pca = PCA(n_components=3)
new_X = pca.fit_transform(datasets)
kmeans = MiniBatchKMeans(n_clusters=100, random_state=0, batch_size=100).fit(new_X)
clusters = kmeans.predict(new_X)
clusters_points = []
for cluster_id in tqdm(range(100)):
    clusters_points.append(datasets[(clusters == cluster_id).nonzero()])

end_time = time.time()

for cc in clusters_points:
    print(f"num: {len(cc)}")

print(f'clustering time : {end_time - start_time}')

# cost_time = end_time - start_time
# 打开文件并写入值
# with open("/research/remote/petabyte/users/peiyu/SimCard/cost_time/clustering_time.txt", "a") as file:
#     file.write('deep:' + str(cost_time) + "\n")

with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/seismic_256_euclidean/clusters_seismic_256_euclidean.pkl', 'wb') as f:
    pickle.dump(clusters_points, f)
f.close()

print("succ save clusters")
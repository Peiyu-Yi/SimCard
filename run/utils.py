#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2023/7/18 10:23
@desc:
"""
import pickle

import numpy as np
import time
import sys


def generate_dictionary(data_file,len):
    start_time = time.time()
    dictionary = {}
    for row in data_file:
        key = tuple(row[:len])
        value = row[len]
        if key in dictionary:
            dictionary[key].append(value)
        else:
            dictionary[key] = [value]
    queries = np.array(list(dictionary.keys()))
    end_time = time.time()
    print(f"num of queries: {queries.shape}")
    print(f"succ dictionary, cost:{end_time - start_time}")
    return dictionary


def load_labeled_data(ts_size, data_file, refine=True, shuffle=True):
    #O = np.load(data_file)
    O = data_file
    if shuffle:
        np.random.shuffle(O)
    X = np.array(O[:, :ts_size], dtype=np.float32)
    T = []
    for rid in range(O.shape[0]):
        t = O[rid, ts_size]
        T.append([t])
    T = np.array(T, dtype=np.float32)
    C = np.array(O[:, -1], dtype=np.float32)
    C.resize((X.shape[0], 1))
    if refine:
        indexes = (C > 1.0)
        T = T[indexes]
        C = C[indexes]
        repeated_values = np.tile(indexes, (1, 300))
        X = X[repeated_values]
        return X.reshape((-1, ts_size)), T.reshape((-1, 1)), C.reshape((-1, 1)), indexes
    else:
        return X, T, C, None


def count_distinct(arrays):
    from collections import Counter

    # Example 2D NumPy array
    # array = np.array([[1, 2, 3, 4], [5, 6, 6, 7], [1, 2, 3, 4]])

    # Convert the 2D NumPy array to tuples
    tuples_list = [tuple(row) for row in arrays]

    # Use Counter to count the occurrence of each tuple
    counter = Counter(tuples_list)

    # Extract the distinct 1D arrays and their occurrence
    distinct_arrays = counter.keys()
    occurrences = counter.values()
    distinct_arrays = [np.array(row) for row in distinct_arrays]
    return distinct_arrays, occurrences


def calculate_consine_distance_matrix(dataset, queries):
    dataset_norms = np.linalg.norm(dataset, axis=1)  # 数据集每条序列的范数，维度为（N，）
    queries_norms = np.linalg.norm(queries, axis=1)  # 查询序列每条序列的范数，维度为（Q，）

    dot_product = np.dot(queries, dataset.T)  # 点积，维度为（Q，N）
    cosine_similarity = dot_product / (np.outer(queries_norms, dataset_norms) + 1e-8)  # 余弦相似度，维度为（Q，N）
    cosine_dist = 1 - cosine_similarity  # 转换为余弦距离，维度为（Q，N）

    return cosine_dist


def calculate_ed_distance_matrix(dataset, queries):
    dataset_squared = np.sum(dataset ** 2, axis=1)  # 数据集每条序列的平方和，维度为（N，）
    queries_squared = np.sum(queries ** 2, axis=1)  # 查询序列每条序列的平方和，维度为（Q，）

    dot_product = np.dot(queries, dataset.T)  # 点积，维度为（Q，N）

    distances_squared = queries_squared[:, np.newaxis] + dataset_squared - 2 * dot_product  # 欧几里得距离平方，维度为（Q，N）
    distances_squared = np.maximum(distances_squared, 0)  # 将小于0的距离设为0，避免负数平方根

    distances = np.sqrt(distances_squared)  # 欧几里得距离，维度为（Q，N）

    return distances


def custom_searchsorted(arr, val):
    if val in arr:
        return np.searchsorted(arr, val, side='right')
    else:
        return np.searchsorted(arr, val, side='left')

def euclidean_dist(x1, x2=None, eps=1e-8):
    # if np.isnan(x2):
    #     return 1.0
    left = x1
    right = x2
    return np.sqrt(((left - right) ** 2).mean())


def cosine_dist(x1, x2):
    cosine_similarity = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-8)
    return 1 - cosine_similarity


if __name__ == "__main__":
    print('Start loading datasets')
    train_dataset = np.load('/research/remote/petabyte/users/s3852583/dataset4simcard/ecg_trainingDatas2.npy')
    test_dataset = np.load('/research/remote/petabyte/users/s3852583/dataset4simcard/ecg_testingDatas2.npy')
    print(f"train_dataset size: {sys.getsizeof(train_dataset)}")
    print(f"test_dataset size: {sys.getsizeof(test_dataset)}")

    train_dictionary = generate_dictionary(train_dataset, len=320)
    test_dictionary = generate_dictionary(test_dataset, len=320)
    print(f"train_dictionary size: {sys.getsizeof(train_dictionary)}")
    print(f"test_dictionary size: {sys.getsizeof(test_dictionary)}")

    start_time = time.time()
    with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/ecg_320_euclidean/train_dictionary_s2.pkl',
              'wb') as f:
        pickle.dump(train_dictionary, f)
    end_time = time.time()
    print(f'pickle dump train_dictionary cost : {end_time - start_time}')

    start_time = time.time()
    with open('/research/remote/petabyte/users/peiyu/SimCard/dataset/ecg_320_euclidean/test_dictionary_s2.pkl',
              'wb') as f:
        pickle.dump(test_dictionary, f)
    end_time = time.time()
    print(f'pickle dump test_dictionary cost : {end_time - start_time}')
    f.close()
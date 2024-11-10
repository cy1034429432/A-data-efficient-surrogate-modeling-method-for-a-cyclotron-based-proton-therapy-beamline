"""
Author: Yu Chen
Email: yu_chen2000@hust.edu.cn
Website: hustyuchen.github.io
"""


import numpy as np
import random

import pandas as pd


def query_max_uncertainty(y_pred, n_instances=100):

    y_pred_max = np.max(y_pred, axis=-1)
    return np.argsort(y_pred_max)[:n_instances]



def query_margin_prob(y_pred, n_instances=100):

    y_pred_sort = np.sort(y_pred, axis=1)
    y_pred_margin = y_pred_sort[:, 2]-y_pred_sort[:, 1]
    return np.argsort(y_pred_margin)[:n_instances]


def query_max_entropy(y_pred, n_instances=100):

    # just normal (0, 1)
    y_pred = 1/(1 + np.e**(-y_pred))
    y_pred_log = np.log(y_pred)
    entropy = np.sum(-y_pred * y_pred_log, axis=1)

    return np.argsort(entropy)[-n_instances:]




def query_margin_kmeans(y_pred, feature, n_instances=100):


    y_pred = 1/(1 + np.e**(-y_pred))
    from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
    y_pred_log = np.log(y_pred)
    entropy = np.sum(-y_pred * y_pred_log, axis=1)
    Cluster = KMeans(n_clusters=n_instances).fit(feature, sample_weight=entropy-min(entropy)+0.01)


    distance = Cluster.transform(feature)
    selected = np.argmin(distance, axis=0)
    return selected


def query_margin_kmeans_pure_diversity(feature, n_instances=100):


    from sklearn.cluster import KMeans
    Cluster = KMeans(n_clusters=n_instances).fit(feature)
    labels = Cluster.labels_
    distance = Cluster.transform(feature)
    selected = np.argmin(distance, axis=0)
    return selected


def query_margin_kmeans_2stages(y_pred, feature, n_instances, beta):

    y_pred_sort = np.sort(y_pred, axis=1)
    y_pred_margin = y_pred_sort[:, 1] - y_pred_sort[:,0]

    y_pred_margin_bottom_index = np.argsort(y_pred_margin)[:beta*n_instances]

    from sklearn.cluster import KMeans
    Cluster = KMeans(n_clusters=n_instances).fit(feature[y_pred_margin_bottom_index])
    labels = Cluster.labels_
    distance = Cluster.transform(feature[y_pred_margin_bottom_index])
    selected = np.argmin(distance, axis=0)
    return y_pred_margin_bottom_index[selected]

def random_sampleing(feature, n_instances):
    selected_list = [i for i in range(feature.shape[0])]
    random.shuffle(selected_list)
    return selected_list[-n_instances:]


if __name__ == '__main__':
    a = random_sampleing(np.zeros(shape=(100,3)),32)

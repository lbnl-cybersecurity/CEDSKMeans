import time
import numpy as np
import numpy.typing as npt
import pandas as pd
import ray
from tqdm import tqdm
from ._map_reduce import (
    KMeansMap,
    KMeansReduce,
    create_new_cluster,
    has_cluster_changed,
    initialize_clusters,
    calculate_distance_matrix,
    split_data,
)


class KMeansMapReduce:
    def __init__(self, n_clusters: int = 20, iteration: int = 10, batch_num: int = 5):
        self.n_clusters = n_clusters
        self.iteration = iteration
        self.batch_num = batch_num
        self.cluster_centers_ = None
        self.execution_time = None
        self.cost = None

    def fit(self, X: pd.DataFrame, y: npt.ArrayLike = None):
        self.n_features = X.shape[1]
        batches = split_data(X, num=self.batch_num)
        center = initialize_clusters(X, self.n_clusters)
        distance_matrix = calculate_distance_matrix(center)

        ray.init()
        mappers = [
            KMeansMap.remote(mini_batch.values, k=self.n_clusters)
            for mini_batch in batches[0]
        ]
        reducers = [
            KMeansReduce.remote(i, self.n_features, *mappers)
            for i in range(self.n_clusters)
        ]

        cost = np.empty((self.iteration, 1))

        start = time.time()
        for i in tqdm(range(self.iteration)):
            for mapper in mappers:
                mapper.communicate_centroids.remote(center)
                mapper.communicate_distances.remote(distance_matrix)

            for mapper in mappers:
                mapper.assign_cluster.remote()

            new_center, cost[i] = create_new_cluster(reducers, self.n_features)
            changed, _ = has_cluster_changed(new_center, center)
            if not changed:
                break
            else:
                center = new_center
                distance_matrix = calculate_distance_matrix(center)
        end = time.time()

        self.execution_time = end - start
        self.cluster_centers_ = center
        self.cost = cost[-1]

        return self

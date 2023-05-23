from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import ray
from sklearn.cluster import KMeans
from tqdm import tqdm

from ._map_reduce import (
    KMeansMap,
    KMeansReduce,
    calculate_distance_matrix,
    create_new_cluster,
    has_cluster_changed,
    initialize_clusters,
    split_data,
)


class KMeansMapReduce(KMeans):
    def __init__(
        self,
        n_clusters: int = 20,
        n_mappers: int = 5,
        *,
        init: Literal["random", "k-means++"] = "k-means++",
        n_init: str | int = "warn",
        max_iter: int = 10,
        tol: float = 1e-4,
        verbose: int = 0,
        random_state: int = None,
        copy_x: bool = True,
        algorithm: Literal["lloyd", "elkan", "auto", "full"] = "lloyd",
    ):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm,
        )
        self.n_mappers = n_mappers
        self.cluster_centers_ = None
        self.cost = None

    def fit(self, X: pd.DataFrame, y: npt.ArrayLike = None):
        self.n_features_out = X.shape[1]
        batches = split_data(X, num_splits=self.n_mappers)
        center = initialize_clusters(X.values, self.n_clusters)
        distance_matrix = calculate_distance_matrix(center)

        ray.init()
        mappers = [
            KMeansMap.remote(mini_batch.values, k=self.n_clusters)
            for mini_batch in batches[0]
        ]
        reducers = [
            KMeansReduce.remote(i, self.n_features_out, *mappers)
            for i in range(self.n_clusters)
        ]

        cost = np.empty((self.iteration, 1))
        for i in tqdm(range(self.iteration)):
            for mapper in mappers:
                mapper.communicate_centroids.remote(center)
                mapper.communicate_distances.remote(distance_matrix)

            for mapper in mappers:
                mapper.assign_cluster.remote()

            new_center, cost[i] = create_new_cluster(reducers, self.n_features_out)
            changed, _ = has_cluster_changed(new_center, center)
            if not changed:
                break
            else:
                center = new_center
                distance_matrix = calculate_distance_matrix(center)
        self.cluster_centers_ = center
        self.cost = cost[-1]
        self.n_iter_ = i

        return self

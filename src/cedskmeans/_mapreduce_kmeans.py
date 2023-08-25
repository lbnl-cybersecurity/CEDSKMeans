from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import ray

from ._cedskmeans import CEDSKMeans
from ._map_reduce import (
    KMeansMap,
    KMeansReduce,
    calculate_distance_matrix,
    create_new_cluster,
    has_cluster_changed,
    initialize_clusters,
    split_data,
)


# There is a bug in sklearn get_params() function that prevents us from using
# the ray.remote decorator. As a workaround, I created a runner method. See below.
class KMeansMapReduce(CEDSKMeans):
    """
    K-means clustering using MapReduce.
    """

    def __init__(
        self,
        n_clusters: int = 20,
        dp_epsilon: float = 0.1,
        dp_delta: float = 0.1,
        n_mappers: int = 5,
        *,
        init: Literal["random", "k-means++"] = "k-means++",
        n_init: str | int = "warn",
        max_iter: int = 10,
        tol: float = 1e-4,
        verbose: int = 0,
        random_state: int = None,  # type: ignore
        copy_x: bool = True,
        algorithm: Literal["lloyd", "elkan", "auto", "full"] = "lloyd",
    ) -> None:
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
            dp_epsilon=dp_epsilon,
            dp_delta=dp_delta,
        )
        self.n_mappers = n_mappers
        self.true_cluster_centers_ = None
        self.cluster_centers_ = None
        self.cost = None

    def fit(self, X: pd.DataFrame, y: npt.ArrayLike = None):
        """Compute k-means clustering.

        Args:
            X (npt.NDArray): Training instances to cluster  of shape (n_samples, n_features). It must be noted that the data will be converted to C ordering, which will cause a memory copy if the given data is not C-contiguous. If a sparse matrix is passed, a copy will be made if it's not in CSR format.
            y (npt.NDArray): Ignored. Not used, present here for API consistency by convention.
            sample_weight (npt.ArrayLike): The weights for each observation in X. If None, all observations are assigned equal weight. Defaults to None.

        Returns:
            self (object): Fitted estimator.
        """
        self.n_features_out = X.shape[1]
        batches = split_data(X, num_splits=self.n_mappers)
        center = initialize_clusters(X.values, self.n_clusters)
        distance_matrix = calculate_distance_matrix(center)

        mappers = [
            KMeansMap.remote(mini_batch.values, num_clusters=self.n_clusters)
            for mini_batch in batches
        ]
        reducers = [
            KMeansReduce.remote(i, self.n_features_out, *mappers)
            for i in range(self.n_clusters)
        ]

        cost = np.empty((self.max_iter, 1))
        for i in range(self.max_iter):
            ray.get(
                [mapper.communicate_centroids.remote(center) for mapper in mappers]
                + [
                    mapper.communicate_distances.remote(distance_matrix)
                    for mapper in mappers
                ]
            )
            ray.get([mapper.assign_clusters.remote() for mapper in mappers])

            new_center, cost[i] = create_new_cluster(reducers)
            changed, _ = has_cluster_changed(new_center, center)
            if not changed:
                break
            else:
                center = new_center
                distance_matrix = calculate_distance_matrix(center)
        self.cost = cost[-1]
        self.n_iter_ = i
        self.true_cluster_centers_ = center
        self._n_features_out = center.shape[0]
        self.true_labels_ = self.predict(X)
        self.n_iter_ = i
        self.cluster_centers_ = center
        self._add_dp_noise(X)
        return self

    def predict(self, X: pd.DataFrame, dp: bool = False) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.

        Args:
            X (pd.DataFrame): New data to predict.

        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        if dp:
            if self.cluster_centers_ is None:
                raise ValueError(
                    "The model has not been trained yet. Please call 'fit' first."
                )
            center = self.cluster_centers_
        else:
            if self.true_cluster_centers_ is None:
                raise ValueError(
                    "The model has not been trained yet. Please call 'fit' first."
                )
            center = self.true_cluster_centers_
        if isinstance(X, pd.DataFrame):
            X = X.values
        distances = np.array([euclidean_distance(x, c) for x in X for c in center])
        distances = distances.reshape(X.shape[0], self.n_clusters)
        return np.argmin(distances, axis=1)


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        point1 (np.ndarray): First point.
        point2 (np.ndarray): Second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))


@ray.remote(num_cpus=1)
def run_kmean_map_reduce(X, **kwargs) -> KMeansMapReduce:
    """
    Run the KMeansMapReduce class.
    This is a workaround for the bug in sklearn get_params() function that prevents us from using the ray.remote decorator.

    Args:
        X (pd.DataFrame): Training instances to cluster  of shape (n_samples, n_features). It must be noted that the data will be converted to C ordering, which will cause a memory copy if the given data is not C-contiguous. If a sparse matrix is passed, a copy will be made if it's not in CSR format.

    Returns:
        kmeans (KMeansMapReduce): Fitted estimator.
    """
    kmeans = KMeansMapReduce(
        **kwargs,
    )
    kmeans.fit(X)
    return kmeans

import numpy as np
import pandas as pd
import ray
from sklearn.cluster import kmeans_plusplus

from . import KMeansReduce


def split_data(
    data_frame: pd.DataFrame, num_splits: int = 3, random_seed: int = None
) -> tuple[pd.DataFrame, ...]:
    """
    Splits a pandas DataFrame into multiple subsets.

    Args:
        data_frame: The input DataFrame to be split.
        num_splits: The number of subsets to create (default: 3).
        random_seed: The seed value for random number generation (default: None).

    Returns:
        A tuple of DataFrame subsets obtained by splitting the input DataFrame.

    """
    np.random.seed(random_seed)
    permutation = np.random.permutation(data_frame.index)
    num_rows = len(data_frame.index)
    split_points = np.linspace(0, num_rows, num_splits + 1, dtype=int)
    data_splits = [
        data_frame.iloc[permutation[split_points[i] : split_points[i + 1]]]
        for i in range(num_splits)
    ]
    return tuple(data_splits)


def has_cluster_changed(
    new_centers: np.ndarray, old_centers: np.ndarray, epsilon: float = 1e-4
) -> tuple[bool, float]:
    """
    Checks if the cluster centers have changed based on a given epsilon threshold.

    Args:
        new_centers: The new cluster centers as a NumPy array.
        old_centers: The old cluster centers as a NumPy array.
        epsilon: The threshold value for considering a change (default: 1e-4).

    Returns:
        A tuple containing a boolean flag indicating if the clusters have changed and the total cost.

    Raises:
        ValueError: If the dimensions of new_centers and old_centers do not match.

    """
    if new_centers.shape[0] != old_centers.shape[0]:
        raise ValueError(
            "Error: Dimensions of new_centers and old_centers do not match!"
        )

    n = new_centers.shape[0]
    total_cost = 0.0
    changed = False

    for i in range(n):
        squared_diff = fast_squared_distance(new_centers[i], old_centers[i])
        if squared_diff > epsilon**2:
            changed = True
        total_cost += squared_diff

    return changed, total_cost


def fast_squared_distance(
    center: np.ndarray,
    point: np.ndarray,
    epsilon: float = 1e-4,
    precision: float = 1e-6,
) -> float:
    """
    Calculates the squared distance between two vectors, taking advantage of optimization techniques.

    Args:
        center: The center vector as a NumPy array.
        point: The point vector as a NumPy array.
        epsilon: The epsilon value for the precision bound (default: 1e-4).
        precision: The precision threshold for choosing the calculation method (default: 1e-6).

    Returns:
        The squared distance between the center and point vectors.

    """
    center_norm = np.linalg.norm(center)
    point_norm = np.linalg.norm(point)
    sum_squared_norm = center_norm**2 + point_norm**2
    norm_diff = center_norm - point_norm

    bound = 2.0 * epsilon * sum_squared_norm / (norm_diff**2 + epsilon)
    squared_distance = 0.0

    if bound < precision:
        squared_distance = sum_squared_norm - 2.0 * np.dot(center, point)
    else:
        squared_distance = np.linalg.norm(center - point)

    return squared_distance


def initialize_clusters(
    data: np.ndarray, n_clusters: int, random_state: int = None
) -> np.ndarray:
    """
    Initialize clusters for k-means clustering algorithm.

    Args:
        data (np.ndarray): The input data array or DataFrame.
        n_clusters (int): The number of clusters to initialize.
        random_state (int, optional): The seed value for random number generation (default: None).

    Returns:
        np.ndarray: The initialized cluster centroids.

    Raises:
        TypeError: If the data type is not supported.

    """
    if isinstance(data, np.ndarray):
        data_c = data.copy()
    else:
        raise TypeError("Data type not supported!")

    centroids, _ = kmeans_plusplus(data_c, n_clusters, random_state=random_state)
    return centroids


def calculate_distance_matrix(center: np.ndarray) -> np.ndarray:
    """
    Calculate the distance matrix between points in the given center array.

    Args:
        center (np.ndarray): Array of points representing the center.

    Returns:
        np.ndarray: The distance matrix between points in the center array.

    """
    n = center.shape[0]
    distance_matrix = np.empty((n, n))
    for i in range(n - 1):
        distance_matrix[i, i + 1] = np.linalg.norm(center[i + 1, :] - center[i, :])
    return distance_matrix


def create_new_cluster(
    reducers: list[KMeansReduce], n_features: int
) -> tuple[np.ndarray, float]:
    """
    Creates a new cluster by combining clusters from multiple KMeansReduce objects.

    Args:
        reducers: A list of KMeansReduce objects.
        n_features: The number of features in the dataset.

    Returns:
        A tuple containing the new cluster as a NumPy array and the total cost.

    """
    total_cost = 0
    combined_cluster = np.empty((0, n_features))

    for reducer in reducers:
        partial_cluster = ray.get(reducer.update_cluster.remote())
        combined_cluster = np.vstack((partial_cluster, combined_cluster))
        total_cost += ray.get(reducer.read_cost.remote())

    return combined_cluster, total_cost

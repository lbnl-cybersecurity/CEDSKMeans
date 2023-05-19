import numpy as np
import ray
import sys
import pandas as pd
from sklearn.cluster import kmeans_plusplus


def split_data(df, seed=None, num=3):
    np.random.seed(seed)

    perm = np.random.permutation(df.index)
    m = len(df.index)
    data = np.zeros((1, num), dtype=object)
    data_end = np.zeros((1, num - 1), dtype=int)
    if num == 1:
        data[0, 0] = df.iloc[:, :]
        return tuple(data)
    for i in range(num - 1):
        data_end[0, i] = int(m / num * (i + 1))

    for i in range(num):
        if i == 0:
            data[0, i] = df.iloc[perm[: data_end[0][0]]]
        elif i == num - 1:
            data[0][i] = df.iloc[perm[data_end[0][i - 1] :]]
        else:
            data[0][i] = df.iloc[perm[data_end[0][i - 1] : data_end[0][i]]]
    return tuple(data)


@ray.remote
class KMeans_Map:
    centroids = 0

    def __init__(self, item, k=1):
        self.item = item
        self.k = k
        self.cluster_centers_ = None
        self._distances_matrix = None

    def communicate_centroids(self, centroids):
        self.centroids = centroids

    def communicate_distances(self, distances_matrix):
        self._distances_matrix = distances_matrix

    def _distance(self, a, b):
        return np.linalg.norm(a - b)

    def read_cluster(self):
        return self._cluster_assignment

    def read_item(self):
        return self.item

    def assign_cluster(self):
        m = self.item.shape[0]
        self._cluster_assignment = np.zeros((m, 2))

        for i in range(m):
            min_dist = np.inf
            min_idx = -1

            min_idx, min_dist = find_closest_centroid(
                self.k, self.centroids, self.item, i, self._distances_matrix
            )
            self._cluster_assignment[i, :] = int(min_idx), min_dist
        return self._cluster_assignment


@ray.remote
class KMeans_Reduce:
    def __init__(self, value, n_features, *kmeans_maps):
        self.value = value
        self.kmeans_maps = kmeans_maps
        self.centroids = None
        self.cluster_assignment = None
        self.cluster_output = np.zeros((1, n_features))
        self.cost = 0

    def read(self):
        return self.value

    def read_cost(self):
        return self.cost

    def update_cluster(self):
        self.cost = 0

        for kmeans_map in self.kmeans_maps:
            self.cluster_assignment = ray.get(kmeans_map.assign_cluster.remote())
            index_all = self.cluster_assignment[:, 0]
            self.cost += np.sum(self.cluster_assignment[:, 1])
            value = np.nonzero(index_all == self.value)
            points_in_cluster = ray.get(kmeans_map.read_item.remote())[value[0]]
            self.cluster_output = np.insert(
                self.cluster_output, 0, points_in_cluster, axis=0
            )

        try:
            self.cluster_output = np.delete(self.cluster_output, -1, axis=0)
        except IndexError:
            print("Incorrect mapper data!")
            sys.exit(2)
        else:
            self.centroids = np.mean(self.cluster_output, axis=0)
            return self.centroids


def create_new_cluster(reducers: list[KMeans_Reduce], n_features: int):
    cost = 0
    new_cluster = np.zeros((1, n_features))
    for reducer in reducers:
        tmp = ray.get(reducer.update_cluster.remote())
        new_cluster = np.insert(new_cluster, 0, tmp, axis=0)
        cost += ray.get(reducer.read_cost.remote())
    return np.delete(new_cluster, -1, axis=0), cost


def has_cluster_changed(new_center, old_center, epsilon=1e-4):
    if new_center.shape[0] != old_center.shape[0]:
        print("Error: Dimensions of new_center and old_center do not match!")
        sys.exit(2)

    n = new_center.shape[0]
    total_cost = 0
    changed = False

    for i in range(n):
        diff = fast_squared_distance(new_center[i], old_center[i])
        if diff > epsilon**2:
            changed = True
        total_cost += diff

    return changed, total_cost


def fast_squared_distance(center, point, epsilon=1e-4, precision=1e-6):
    center_norm = np.linalg.norm(center)
    point_norm = np.linalg.norm(point)
    sum_squared_norm = center_norm**2 + point_norm**2
    norm_diff = center_norm - point_norm

    precision_bound1 = 2.0 * epsilon * sum_squared_norm / (norm_diff**2 + epsilon)
    sq_dist = 0.0

    if precision_bound1 < precision:
        sq_dist = sum_squared_norm - 2.0 * np.dot(center, point)
    else:
        sq_dist = np.linalg.norm(center - point)

    return sq_dist


def find_closest_centroid(k, centroids, item, i, distances_matrix):
    best_distance = np.inf
    best_index = -1
    j = 0
    num_centroids = centroids.shape[0]

    while j < num_centroids:
        center = centroids[j]
        distance = np.linalg.norm(center - item[i])

        if distance < best_distance:
            best_distance = distance
            best_index = j

        if j <= k - 2 and 2 * distance <= distances_matrix[j, j + 1]:
            j += 1

        j += 1

    return best_index, best_distance


def initialize_clusters(data, n_clusters):
    n_features = data.shape[1]
    centroids = np.empty((n_clusters, n_features))
    total_samples = data.shape[0]
    sample_indices = np.arange(total_samples)
    sample_probabilities = np.empty(shape=(1, total_samples), dtype=np.float32)

    first_center = np.random.randint(0, data.shape[0])
    centroids[0] = data.loc[first_center]

    for i in range(1, n_clusters):
        index_row = 0
        total_distance = 0

        for row in data.values:
            min_distance = np.inf

            for j in range(i):
                distance_j = np.linalg.norm(row - centroids[j])

                if distance_j < min_distance:
                    min_distance = distance_j

            total_distance += min_distance
            sample_probabilities[0][index_row] = min_distance
            index_row += 1

        sample_probabilities = sample_probabilities / total_distance
        chosen_index = np.random.choice(
            sample_indices, p=sample_probabilities[0].ravel()
        )
        centroids[i] = data.loc[chosen_index]

    return centroids


def initialize_clusters_sklearn(data, n_clusters, random_state=None):
    if isinstance(data, pd.DataFrame):
        data_c = data.copy()
        data_c = data_c.values
    elif isinstance(data, np.ndarray):
        data_c = data.copy()
    else:
        raise TypeError("Data type not supported!")

    centroids, _ = kmeans_plusplus(data_c, n_clusters, random_state=random_state)
    return centroids


def calculate_distance_matrix(center):
    n = center.shape[0]
    distance_matrix = np.empty((n, n))
    # differences = np.diff(center, axis=0)
    # distance_matrix[:-1, 1:] = np.linalg.norm(differences, axis=1).reshape(-1, 1)
    for i in range(n - 1):
        distance_matrix[i, i + 1] = np.linalg.norm(center[i + 1, :] - center[i, :])
    return distance_matrix

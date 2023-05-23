import numpy as np
import ray


@ray.remote
class KMeansReduce:
    """
    Class representing a K-Means reducer.

    Args:
        value (int): The value associated with the reducer.
        n_features (int): The number of features in the data.
        *kmeans_maps (list[ray.actor.ActorHandle]): Variable-length arguments representing K-Means mappers.

    Attributes:
        value (int): The value associated with the reducer.
        kmeans_maps (list[ray.actor.ActorHandle]): List of K-Means mapper handles.
        centroids (None | np.ndarray): The computed centroids of the clusters.
        cluster_output (np.ndarray): The combined cluster output.
        cost (float): The cost of the clustering.
    """

    def __init__(
        self, value: int, n_features: int, *kmeans_maps: list[ray.actor.ActorHandle]
    ) -> None:
        self.value = value
        self.kmeans_maps = kmeans_maps
        self.centroids = None
        self.cluster_output = np.zeros((1, n_features))
        self.cost = 0

    def read(self) -> int:
        """
        Get the value associated with the reducer.

        Returns:
            int: The value associated with the reducer.
        """
        return self.value

    def read_cost(self) -> float:
        """
        Get the cost of the clustering.

        Returns:
            float: The cost of the clustering.
        """
        return self.cost

    def update_cluster(self) -> None | np.ndarray:
        """
        Update the cluster by computing the centroids.

        Returns:
            None | np.ndarray: The computed centroids, or None if an error occurs.
        """
        cost = 0.0
        cluster_output = np.empty((0, self.cluster_output.shape[1]))

        for kmeans_map in self.kmeans_maps:
            cluster_assignment = ray.get(kmeans_map.assign_cluster.remote())
            cost += np.sum(cluster_assignment[:, 1])
            cluster_indices = np.where(cluster_assignment[:, 0] == self.value)[0]
            points_in_cluster = ray.get(kmeans_map.read_item.remote())[cluster_indices]
            cluster_output = np.concatenate((cluster_output, points_in_cluster), axis=0)

        self.cost = cost
        self.cluster_output = cluster_output
        self.centroids = np.mean(cluster_output, axis=0)

        return self.centroids

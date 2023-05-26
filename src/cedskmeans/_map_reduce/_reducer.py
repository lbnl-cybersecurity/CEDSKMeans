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
        centroid (None | np.ndarray): The computed centroid of the cluster.
        cluster_output (np.ndarray): The combined cluster output.
        cost (float): The cost of the clustering.
    """

    def __init__(
        self, value: int, n_features: int, *kmeans_maps: list[ray.actor.ActorHandle]
    ) -> None:
        self.value = value
        self.kmeans_maps = kmeans_maps
        self.centroid = None
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
        Update the cluster by computing the centroid. This method collects all the 
        references to the cluster assignments and mapper items before calling ray.get().

        Returns:
            None | np.ndarray: The computed centroid, or None if an error occurs.
        """
        cluster_assignment_refs = [
            kmeans_map.assign_cluster.remote() for kmeans_map in self.kmeans_maps
        ]
        mapper_items_refs = [
            kmeans_map.read_items.remote() for kmeans_map in self.kmeans_maps
        ]

        cluster_assignments = ray.get(cluster_assignment_refs)

        self.cost = np.sum(
            [
                np.sum(cluster_assignment[:, 1])
                for cluster_assignment in cluster_assignments
            ]
        )
        cluster_indices_all = [
            np.where(cluster_assignment[:, 0] == self.value)[0]
            for cluster_assignment in cluster_assignments
        ]
        mapper_items_all = ray.get(mapper_items_refs)
        self.cluster_output = np.array(
            [
                mapper_items[cluster_indices]
                for mapper_items, cluster_indices, in zip(
                    mapper_items_all, cluster_indices_all
                )
            ]
        )
        self.centroid = np.mean(self.cluster_output, axis=0)
        return self.centroid

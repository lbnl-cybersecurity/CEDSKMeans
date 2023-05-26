import numpy as np
import ray


@ray.remote
class KMeansMap:
    """
    A remote Ray class for performing K-means clustering map tasks.

    Args:
        items (np.ndarray): The input array of items.
        num_clusters (int, optional): The number of clusters (default: 1).

    Attributes:
        items (np.ndarray): The input array of items.
        num_clusters (int): The number of clusters.
        cluster_assignments (np.ndarray): The array to store cluster assignments.
        distances_matrix (np.ndarray): The matrix containing pre-calculated distances between centroids.
    """

    def __init__(self, items: np.ndarray, num_clusters: int = 1):
        self.items = items
        self.num_clusters = num_clusters
        self.cluster_assignments = None
        self.distances_matrix = None

    def communicate_centroids(self, centroids: np.ndarray):
        """
        Communicates the centroids to the KMeansMap instance.

        Args:
            centroids (np.ndarray): The centroids to be communicated.
        """
        self.centroids = centroids

    def communicate_distances(self, distances_matrix: np.ndarray):
        """
        Communicates the distances matrix to the KMeansMap instance.

        Args:
            distances_matrix (np.ndarray): The distances matrix to be communicated.
        """
        self.distances_matrix = distances_matrix

    @staticmethod
    def calculate_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
        """
        Calculates the distance between two points.

        Args:
            point_a (np.ndarray): The first point.
            point_b (np.ndarray): The second point.

        Returns:
            float: The distance between the two points.
        """
        return np.linalg.norm(point_a - point_b)

    def read_cluster_assignments(self) -> np.ndarray:
        """
        Reads the cluster assignments.

        Returns:
            np.ndarray: The cluster assignments.
        """
        return self.cluster_assignments

    def read_items(self):
        """
        Reads the input array of items.

        Returns:
            np.ndarray: The input array of items
        """
        return self.items

    def assign_clusters(self) -> np.ndarray:
        """
        Assigns clusters to each item.

        Returns:
            np.ndarray: The cluster assignments.
        """
        num_items = self.items.shape[0]
        self.cluster_assignments = np.zeros((num_items, 2))

        for i in range(num_items):
            min_distance = np.inf
            min_index = -1

            min_index, min_distance = KMeansMap.find_closest_centroid(
                self.num_clusters, self.centroids, self.items, i, self.distances_matrix
            )
            self.cluster_assignments[i, :] = min_index, min_distance

        return self.cluster_assignments

    @staticmethod
    def find_closest_centroid(
        num_clusters: int,
        centroids: np.ndarray,
        items: np.ndarray,
        item_index: int,
        distances_matrix: np.ndarray,
    ) -> tuple[int, float]:
        """
        Find the index of the closest centroid to the given item.

        Args:
            num_clusters (int): The number of centroids.
            centroids (np.ndarray): An array of centroid coordinates.
            items (np.ndarray): An array of item coordinates.
            item_index (int): Index of the item to compare.
            distances_matrix (np.ndarray): Matrix containing pre-calculated distances between centroids.

        Returns:
            tuple[int, float]: Index of the closest centroid and the corresponding distance.
        """
        best_distance = np.inf
        best_index = -1
        j = 0
        num_centroids = centroids.shape[0]

        while j < num_centroids:
            center = centroids[j]
            distance = np.linalg.norm(center - items[item_index])

            if distance < best_distance:
                best_distance = distance
                best_index = j

            if j <= num_clusters - 2 and 2 * distance <= distances_matrix[j, j + 1]:
                j += 1

            j += 1

        return best_index, best_distance

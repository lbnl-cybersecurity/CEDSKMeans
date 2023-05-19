import time
import ray
from tqdm import tqdm
from ._map_reduce import (
    KMeans_Map,
    KMeans_Reduce,
    create_new_cluster,
    has_cluster_changed,
    initialize_clusters_sklearn,
    calculate_distance_matrix,
    split_data,
)


class KMeans_MR:
    def __init__(self, data, sample=None, n_clusters=20, iteration=10):
        self.sample = sample
        self.n_clusters = n_clusters
        self.iteration = iteration
        self.df = data
        self.n_features = data.shape[1]
        self.cluster_centers_ = None

    def fit(self, batch_num):
        batches = split_data(self.df, num=batch_num)

        center = initialize_clusters_sklearn(self.df, self.n_clusters)
        # print("Init centers: ", center)

        n = center.shape[0]

        distance_matrix = calculate_distance_matrix(center)

        ray.init()
        maps = [
            KMeans_Map.remote(mini_batch.values, k=self.n_clusters)
            for mini_batch in batches[0]
        ]
        reducers = [
            KMeans_Reduce.remote(i, self.n_features, *maps)
            for i in range(self.n_clusters)
        ]
        start = time.time()
        cost = 0

        for i in tqdm(range(self.iteration)):
            for map_ in maps:
                map_.communicate_centroids.remote(center)
                map_.communicate_distances.remote(distance_matrix)

            for map_ in maps:
                map_.assign_cluster.remote()

            new_center, cost = create_new_cluster(reducers, self.n_features)
            changed, cost_1 = has_cluster_changed(new_center, center)
            if not changed:
                break
            else:
                center = new_center
                distance_matrix = calculate_distance_matrix(center)

        end = time.time()
        print(center)
        self.cluster_centers_ = center
        print("Time: ", end - start, ", Cost: ", cost)

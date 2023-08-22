import pandas as pd
import ray
from sklearn.datasets import make_blobs

from cedskmeans import DPKMeans
from cedskmeans import kMeansMapReduceRunner


def get_data():
    x_array, target, centers_ = make_blobs(  # type: ignore
        n_samples=100,
        n_features=3,
        centers=3,
        random_state=0,
        return_centers=True,
    )
    return pd.DataFrame(data=x_array)


def single_process_demo(X) -> None:
    # Create a CEDSKMeans object
    kmeans = DPKMeans(n_clusters=6, dp_epsilon=0.1, dp_delta=1e-5, max_iter=1000)
    kmeans.fit(X)
    # Access the labels
    labels = kmeans.labels_
    # Access the centroids
    centroids = kmeans.cluster_centers_
    # Access the true labels
    true_labels = kmeans.true_labels_
    # Access the true centroids
    true_centroids = kmeans.true_cluster_centers_
    print(kmeans.cluster_centers_)
    print(kmeans.true_cluster_centers_)


def distributed_demo(X):
    # Create a CEDSKMeans object
    kmeans = kMeansMapReduceRunner.remote(
        X=X,
        n_clusters=3,
        n_mappers=2,
        max_iter=1000,
        dp_epsilon=0.1,
        dp_delta=1e-5,
    )
    kmeans = ray.get(kmeans)

    # Access the labels
    labels = kmeans.labels_
    # Access the centroids
    centroids = kmeans.cluster_centers_
    print(kmeans.cluster_centers_)
    print(kmeans.true_cluster_centers_)


def main():
    x = get_data()
    single_process_demo(x)
    ray.init()
    distributed_demo(x)


if __name__ == "__main__":
    main()

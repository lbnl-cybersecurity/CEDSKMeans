import ray
from . import KMeansMapReduce


def main(X):
    ray.init()
    kmeans = KMeansMapReduce(n_clusters=3)
    kmeans.fit(X)
    return kmeans


if __name__ == "__main__":
    main()

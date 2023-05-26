# CEDS DP K-Means Clustering

This repository contains the code for the CEDS Data Product K-Means Clustering. It follows the methodology proposed in papers [1] and [2].

[1] Ravi, Nikhil, et al. "Differentially Private-Means Clustering Applied to Meter Data Analysis and Synthesis." IEEE Transactions on Smart Grid 13.6 (2022): 4801-4814.  
[2] Ravi, Nikhil, Anna Scaglione, and Sean Peisert. "Colored noise mechanism for differentially private clustering." arXiv preprint arXiv:2111.07850 (2021).

## Installation
```shell
pip install git+https://github.com/lbnl-cybersecurity/CEDSKMeans.git
```

## Centralized Usage
```python
from cedskmeans import DPKMeans

# Import the data
X = "Import data here in the form of a numpy ndarray"

# Create a CEDSKMeans object
kmeans = DPKMeans(
    n_clusters=6,
    epsilon=0.1,
    delta=1e-5,
    max_iter=1000
)
kmeans.fit(X)

# Access the labels
labels = kmeans.labels_
# Access the centroids
centroids = kmeans.cluster_centers_

# Access the true labels
true_labels = kmeans.true_labels_
# Access the true centroids
true_centroids = kmeans.true_cluster_centers_
```

## Distributed Map Reduce Usage (requires `ray`)
```python
from cedskmeans import KMeansReduce
import ray

# Import the data
X = "Import data here in the form of a numpy ndarray"

ray.init()
# Create a CEDSKMeans object
kmeans = KMeansReduce(
    n_clusters=6,
    max_iter=1000
    # epsilon=0.1, # TODO: Add support for DP
    # delta=1e-5, # TODO: Add support for DP
)
kmeans = kmeans.fit.remote(X)
kmeans = ray.get(kmeans)


# Access the labels
labels = kmeans.labels_ # TODO: Add support for centralized labels
# Access the centroids
centroids = kmeans.cluster_centers_

# TODO: Add support for DP
# # Access the true labels
# true_labels = kmeans.true_labels_
# # Access the true centroids
# true_centroids = kmeans.true_cluster_centers_
```


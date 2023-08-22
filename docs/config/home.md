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
from cedskmeans import run_kmean_map_reduce
import ray

# Import the data
X = "Import data here in the form of a pandas dataframe"

ray.init()
# Create a CEDSKMeans object
kmeans = run_kmean_map_reduce.remote(
    X=X,
    n_clusters=3,
    n_mappers=2,
    max_iter=1000,
    epsilon=0.1, 
    delta=1e-5, 
)
kmeans = ray.get(kmeans)

# Access the labels
labels = kmeans.labels_ 
# Access the centroids
centroids = kmeans.cluster_centers_
```

## Copyright Notice

CEDS Differential Privacy (CEDSDP) Copyright (c) 2023, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy) and Cornell University. All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
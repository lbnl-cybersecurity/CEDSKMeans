import numpy.typing as npt
from ._cedskmeans import CEDSKMeans


class DPKMeans(CEDSKMeans):
    def fit(
        self, X: npt.NDArray, y: npt.NDArray = None, sample_weight: npt.ArrayLike = None
    ):
        _kmeans_results = super().fit(X=X, y=y, sample_weight=sample_weight)
        self.true_cluster_centers_ = _kmeans_results.cluster_centers_
        self._n_features_out = _kmeans_results.cluster_centers_.shape[0]
        self.true_labels_ = _kmeans_results.labels_
        self.inertia_ = _kmeans_results.inertia_
        self.n_iter_ = _kmeans_results.n_iter_
        self._add_dp_noise(X)
        return self

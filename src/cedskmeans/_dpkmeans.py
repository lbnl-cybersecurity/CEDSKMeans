from typing import Literal
import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
import sympy


class DPKMeans(KMeans):
    def __init__(
        self,
        n_clusters: int = 8,
        dp_epsilon: float = 0.1,
        dp_delta: float = 0.1,
        *,
        init: Literal["random", "k-means++"] = "k-means++",
        n_init: str | int = "warn",
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: int = 0,
        random_state: int = None,
        copy_x: bool = True,
        algorithm: Literal["lloyd", "elkan", "auto", "full"] = "lloyd",
    ):
        """Differentially private k-means clustering.

        Read more in [1].

        Args:
            n_clusters (int, optional): Refer to [2]. Defaults to 8.
            dp_epsilon (float, optional): The differential privacy parameter epsilon [1]. Defaults to 0.1.
            dp_delta (float, optional): The differential private parameter delta [1]. Defaults to 0.1.
            init (Literal["random", "k-means++", optional): Refer to [2]. Defaults to "k-means++".
            n_init (str | int, optional): Refer to [2]. Defaults to "warn".
            max_iter (int, optional): Refer to [2]. Defaults to 300.
            tol (float, optional): Refer to [2]. Defaults to 1e-4.
            verbose (int, optional): Refer to [2]. Defaults to 0.
            random_state (int, optional): Refer to [2]. Defaults to None.
            copy_x (bool, optional): Refer to [2]. Defaults to True.
            algorithm (Literal["lloyd", "elkan", "auto", "full"], optional): Refer to [2]. Defaults to "lloyd".

        [1] Ravi, Nikhil, et al. "Differentially Private-Means Clustering Applied to Meter Data Analysis and Synthesis." IEEE Transactions on Smart Grid 13.6 (2022): 4801-4814.
        [2] https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        """
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm,
        )

        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta

    def fit(
        self, X: npt.NDArray, y: npt.NDArray = None, sample_weight: npt.ArrayLike = None
    ):
        """Compute k-means clustering.

        Args:
            X (npt.NDArray): Training instances to cluster  of shape (n_samples, n_features).
            It must be noted that the data will be converted to C ordering,
            which will cause a memory copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.
            y (npt.NDArray): Ignored. Not used, present here for API consistency by convention.
            sample_weight (npt.ArrayLike): The weights for each observation in X.
            If None, all observations are assigned equal weight. Defaults to None.

        Returns:
            self (object): Fitted estimator.
        """
        _kmeans_results = super().fit(X=X, y=y, sample_weight=sample_weight)
        self.true_cluster_centers_ = _kmeans_results.cluster_centers_
        self._n_features_out = _kmeans_results.cluster_centers_.shape[0]
        self.true_labels_ = _kmeans_results.labels_
        self.inertia_ = _kmeans_results.inertia_
        self.n_iter_ = _kmeans_results.n_iter_
        self.color_cov = self._calculate_color_noise_covariance(X)
        self.cluster_centers_ = (
            self.true_cluster_centers_
            + np.random.multivariate_normal(
                np.zeros(self.n_clusters * X.shape[1]), self.color_cov
            ).reshape(self.n_clusters, X.shape[1])
        )
        self.labels_ = self.true_labels_
        return self

    def _calculate_color_noise_covariance(self, X: npt.NDArray) -> npt.NDArray:
        """Calculates the differentially private color noise covariance matrix [1].

        Args:
            X (npt.NDArray): Training instances to cluster. Of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        Returns:
            npt.NDArray: DP covariance matrix of shape (n_clusters * n_features, n_clusters * n_features).

        [1] Ravi, Nikhil, Anna Scaglione, and Sean Peisert.
        "Colored noise mechanism for differentially private clustering." (2021).
        https://arxiv.org/pdf/2006.03684.pdf
        """

        def __cluster_cardinality(labels: npt.NDArray, n_clusters: int) -> npt.NDArray:
            """Calculates the cardinality of each cluster.

            Args:
                labels (npt.NDArray): Cluster labels for each sample.
                n_clusters (int): Number of clusters.

            Returns:
                npt.NDArray: Cardinality of each cluster of shape (n_clusters, 1).
            """
            return np.bincount(labels, minlength=n_clusters).reshape(-1, 1)

        def __cluster_sum(
            X: npt.NDArray, labels: npt.NDArray, n_clusters: int
        ) -> npt.NDArray:
            """Calculates the sum of data points in each cluster.

            Args:
                X (npt.NDArray): Training instances to cluster.
                labels (npt.NDArray): Cluster labels for each sample.
                n_clusters (int): Number of clusters.

            Returns:
                npt.NDArray: Sum of data points in each cluster of shape (n_clusters, n_features).
            """
            return np.array([np.sum(X[labels == k], axis=0) for k in range(n_clusters)])

        def __cluster_center_without_node(
            X: npt.NDArray,
            labels: npt.NDArray,
            cluster_sum: npt.NDArray,
            cluster_cardinality: npt.NDArray,
        ) -> npt.NDArray:
            """Calculates the cluster center without the data point.

            Args:
                X (npt.NDArray): Training instances to cluster.
                labels (npt.NDArray): Cluster labels for each sample.
                cluster_sum (npt.NDArray): Sum of data points in each cluster.
                cluster_cardinality (npt.NDArray): Cardinality of each cluster.

            Returns:
                np.ndarray: Cluster center without the data point of shape (n_samples, n_features).
            """
            return (cluster_sum[labels] - X) / (cluster_cardinality[labels] - 1)

        def __modified_cluster_centers(
            X: npt.NDArray,
            centers: npt.NDArray,
            labels: npt.NDArray,
            cluster_center_without_node: npt.NDArray,
        ) -> npt.NDArray:
            """For each data point, this method return a flattened version of the
            cluster centers with the cluster center of the data point replaced
            with the cluster center without the data point.

            Args:
                X (npt.NDArray): Training instances to cluster.
                centers (npt.NDArray): Original cluster centers.
                labels (npt.NDArray): Cluster labels for each sample.
                cluster_center_without_node (npt.NDArray): Cluster center without the data point.

            Returns:
                npt.NDArray: Modified cluster centers of shape (n_clusters * n_features, n_samples).
            """
            modified_cluster_centers = np.tile(centers, (X.shape[0], 1, 1))
            modified_cluster_centers[
                np.arange(X.shape[0]), labels
            ] = cluster_center_without_node
            return modified_cluster_centers.reshape(X.shape[0], -1).T

        def __sort_columns_by_norm(matrix: npt.NDArray) -> npt.NDArray:
            """Sorts the columns of a matrix by their norm.

            Args:
                matrix (npt.NDArray): Matrix to sort.

            Returns:
                npt.NDArray: Sorted matrix.
            """
            norm = np.linalg.norm(matrix, axis=0)
            return matrix[:, np.argsort(norm)]

        def __find_independent_columns(matrix: npt.NDArray) -> npt.NDArray:
            """Finds the independent columns of a matrix.

            Args:
                matrix (npt.NDArray): Matrix to find independent columns.

            Returns:
                npt.NDArray: Independent columns of the matrix.
            """
            _, independent_indices = sympy.Matrix(matrix).rref()
            return matrix[:, independent_indices]

        def __svd_transpose(matrix: npt.NDArray) -> npt.NDArray:
            """Calculates the SVD of the transpose of a matrix.

            Args:
                matrix (npt.NDArray): Matrix to calculate SVD.

            Returns:
                npt.NDArray: SVD of the transpose of the matrix.
            """
            U, D, V = np.linalg.svd(matrix)
            return np.square(
                U[: matrix.shape[1], : matrix.shape[1]] @ np.diag(np.sqrt(D)) @ V
            )

        def __calculate_langrange_multiplier(
            matrix: npt.NDArray, gamma_c: float
        ) -> npt.NDArray:
            """Calculates the Langrange multiplier.

            Args:
                matrix (npt.NDArray): A matrix. Refer to eq. 22 of [1]
                gamma_c (float): Differential privacy parameter.

            Returns:
                npt.NDArray: Langrange multiplier.

            [1] Ravi, Nikhil, Anna Scaglione, and Sean Peisert.
            "Colored noise mechanism for differentially private clustering." (2021).
            https://arxiv.org/pdf/2006.03684.pdf
            """
            lambda_star = np.real(
                np.power(
                    gamma_c * (np.linalg.inv(matrix) @ np.ones((len(matrix), 1))),
                    -2,
                )
            )
            return lambda_star

        def __calculate_objective_function(
            matrix: npt.NDArray, lagrange_multiplier: npt.NDArray
        ) -> npt.NDArray:
            """Calculates the objective function.

            Args:
                matrix (npt.NDArray): A matrix. Refer to eq. 13 of [1]
                lagrange_multiplier (npt.NDArray): Langrange multiplier.

            Returns:
                npt.NDArray: Objective function.

            [1] Ravi, Nikhil, Anna Scaglione, and Sean Peisert.
            "Colored noise mechanism for differentially private clustering." (2021).
            https://arxiv.org/pdf/2006.03684.pdf
            """
            return np.sum(
                lagrange_multiplier[i]
                * np.outer(
                    matrix[:, i],
                    matrix[:, i],
                )
                for i in range(matrix.shape[1])
            )

        def __calculate_color_noise_covariance(matrix: npt.NDArray) -> npt.NDArray:
            """Calculates the covariance matrix of the colored noise.

            Args:
                matrix (npt.NDArray): A matrix. Refer to eq. 11 of [1]

            Returns:
                npt.NDArray: Covariance matrix of the colored noise.

            [1] Ravi, Nikhil, Anna Scaglione, and Sean Peisert.
            "Colored noise mechanism for differentially private clustering." (2021).
            https://arxiv.org/pdf/2006.03684.pdf
            """
            U, D, V = np.linalg.svd(matrix)
            return U @ np.diag(np.sqrt(D)) @ V

        cluster_cardinality = __cluster_cardinality(self.true_labels_, self.n_clusters)
        cluster_sum = __cluster_sum(X, self.true_labels_, self.n_clusters)
        cluster_center_without_node = __cluster_center_without_node(
            X, self.true_labels_, cluster_sum, cluster_cardinality
        )
        modified_cluster_centers = __modified_cluster_centers(
            X,
            self.true_cluster_centers_,
            self.true_labels_,
            cluster_center_without_node,
        )
        change_in_cluster_centers = (
            self.true_cluster_centers_.reshape(-1, 1) - modified_cluster_centers
        )  # Form the $\bm{C}_{XX'}$ matrix

        sorted_change_in_cluster_centers = __sort_columns_by_norm(
            change_in_cluster_centers
        )
        independent_columns = __find_independent_columns(
            sorted_change_in_cluster_centers
        )
        independent_columns_T = __svd_transpose(independent_columns)

        dp_epsilon_total = self.dp_epsilon * self.n_clusters * X.shape[1]
        gamma_c = dp_epsilon_total**2 / (2 * np.log(2 / self.dp_delta))

        lambda_star = __calculate_langrange_multiplier(independent_columns_T, gamma_c)
        R_lambda_star = __calculate_objective_function(independent_columns, lambda_star)

        dp_color_covariance = __calculate_color_noise_covariance(R_lambda_star)
        return dp_color_covariance

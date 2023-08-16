from cedskmeans import kMeansMapReduceRunner, KMeansMapReduce
import pandas as pd
from sklearn.datasets import make_blobs
from typing import Literal
import ray
from dataclasses import dataclass
import numpy as np


@dataclass
class Dataset:
    df: pd.DataFrame
    target: np.ndarray
    n_clusters: int
    n_iter: int
    n_mappers: int
    centers_: np.ndarray


def prepare_data(
    type: Literal["mnist", "synthetic"] = "synthetic",
    num_samples: int = 1000,
    n_features: int = 2,
    num_clusters: int = 5,
    num_mappers: int = 5,
) -> Dataset:
    """
    Prepare data for the experiment based on the type of data required  (mnist or synthetic).

    Args:
        type (Literal["mnist", "synthetic"], optional): Type of data to be used. Defaults to "synthetic".
        num_samples (int, optional): Number of samples to be generated. Defaults to 1000.
        n_features (int, optional): Number of features to be generated. Defaults to 2.
        num_clusters (int, optional): Number of clusters to be generated. Defaults to 5.
        num_mappers (int, optional): Number of mappers to be used. Defaults to 5.

    Returns:
        Dataset: Dataset object containing the data and other parameters.
    """
    if type == "mnist":
        number_of_clusters = 20
        number_of_iteration = 10
        number_of_mappers = 5

        df = pd.read_csv("../data/mnist/mnist.csv")
        df = df.fillna(0)
        # df = np.transpose(df.fillna(0))
        target = df.iloc[:, 0]
        df = df.iloc[:, 1:]
        for unique_target in np.unique(target):
            df[target == unique_target] = df[target == unique_target] / np.max(
                df[target == unique_target]
            )
        # Calculate the average of each class
        centers_ = np.zeros((number_of_clusters, df.shape[1]))
        for unique_target in np.unique(target):
            centers_[unique_target, :] = np.mean(df[target == unique_target], axis=0)

        # print("dataframe read")
        return Dataset(
            df=df,
            target=target.values,  # type: ignore
            n_clusters=number_of_clusters,
            n_iter=number_of_iteration,
            n_mappers=number_of_mappers,
            centers_=centers_,
        )
    elif type == "synthetic":
        number_of_clusters = num_clusters
        number_of_iteration = 10
        number_of_mappers = num_mappers
        n_samples = num_samples

        X, target, centers_ = make_blobs(  # type: ignore
            n_samples=n_samples,
            n_features=n_features,
            centers=number_of_clusters,
            random_state=0,
            return_centers=True,
        )
        df = pd.DataFrame(X)
        # print("dataframe read")
        return Dataset(
            df=df,
            target=target,
            n_clusters=number_of_clusters,
            n_iter=number_of_iteration,
            n_mappers=number_of_mappers,
            centers_=centers_,
        )


def run(
    n_samples: int = 1000,
    n_features: int = 2,
    n_clusters: int = 5,
    n_mappers: int = 5,
) -> KMeansMapReduce:
    """
    Run the experiment with the given parameters.

    Args:
        n_samples (int, optional): Number of samples to be generated. Defaults to 1000.
        n_features (int, optional): Number of features to be generated. Defaults to 2.
        n_clusters (int, optional): Number of clusters to be generated. Defaults to 5.
        n_mappers (int, optional): Number of mappers to be used. Defaults to 5.

    Returns:
        KMeansMapReduce: KMeansMapReduce object containing the fitted class.
    """
    dataset = prepare_data(
        "synthetic",
        num_samples=n_samples,
        n_features=n_features,
        num_clusters=n_clusters,
        num_mappers=n_mappers,
    )
    pipeline: KMeansMapReduce = ray.get(
        kMeansMapReduceRunner.remote(
            n_clusters=dataset.n_clusters,  # type: ignore
            max_iter=dataset.n_iter,
            n_mappers=dataset.n_mappers,
            X=dataset.df,
        )
    )
    return pipeline

# poetry install
# From root: poetry run python scripts/hpm.py

import functools
from datetime import datetime

import ray
from hpm_utils import kMeansMapReduceRun
from mytimeit import nice_timeit
from ray import tune
from ray.air.config import RunConfig

REPEAT = 7
NUMBER = 10


def objective(config: dict[str, int]) -> dict[str, float]:
    """
    Objective function for hyperparameter tuning. For a given configuration of hyperparameters, including the number of
    samples, number of features, number of mappers, and number of clusters, run the MapReduce K-Means algorithm `n` times
    and repeat that `r` times.

    Args:
        config (dict[str, int]): Configuration of hyperparameters.

    Returns:
        dict[str, float|list[float]]: Dictionary of results, including the mean, standard deviation, best, worst, and compile time
    """
    n_samples = config["n_samples"]
    n_mappers = config["n_mappers"]
    n_clusters = config["n_clusters"]
    n_features = config["n_features"]
    res = nice_timeit(
        functools.partial(
            kMeansMapReduceRun,
            n_samples=n_samples,
            n_clusters=n_clusters,
            n_mappers=n_mappers,
            n_features=n_features,
        ),
        repeat=REPEAT,
        number=NUMBER,
    )
    return {
        "mean": res.average,
        "std": res.stdev,
        "best": res.best,
        "worst": res.worst,
        "compile_time": res.compile_time,
        "timings": res.timings,
    }


if __name__ == "__main__":
    ray.shutdown()
    ray.init()

    # Search space for hyperparameters
    search_space = {
        "n_samples": tune.grid_search([100, 1000]),
        "n_features": tune.grid_search([1, 24, 24 * 30, 24 * 365]),
        "n_mappers": tune.grid_search([8]),
        "n_clusters": tune.grid_search([5]),
    }
    # Resource allocation for hyperparameter tuning
    resource_group = tune.PlacementGroupFactory([{"CPU": 1.0}] + [{"CPU": 1.0}] * 10)
    trainable_with_resources = tune.with_resources(objective, resource_group)
    timenow = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Run hyperparameter tuning
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=search_space,
        run_config=RunConfig(
            name=f"hpm_r{REPEAT}_n{NUMBER}_{timenow}", local_dir="./.results"
        ),
    )
    results = tuner.fit()

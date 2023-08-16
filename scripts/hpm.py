import functools
import numpy as np
import timeit
import ray
from ray import tune
from ray.air.config import RunConfig
from hpm_utils import run
from mytimeit import nice_timeit
from datetime import datetime

REPEAT = 7
NUMBER = 10


def objective(config):
    n_samples = config["n_samples"]
    n_mappers = config["n_mappers"]
    n_clusters = config["n_clusters"]
    n_features = config["n_features"]
    res = nice_timeit(
        functools.partial(
            run,
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

    # search_space = {
    #     "n_samples": tune.grid_search([100, 1000, 10000, 100000]),
    #     "n_features": tune.grid_search([1, 24, 24 * 30, 24 * 365]),
    #     "n_mappers": tune.grid_search([1, 2, 4, 8, 16, 32]),
    #     "n_clusters": tune.grid_search([5, 10, 15]),
    # }
    search_space = {
        "n_samples": tune.grid_search([100, 1000]),
        "n_features": tune.grid_search([1, 24, 24 * 30, 24 * 365]),
        "n_mappers": tune.grid_search([8]),
        "n_clusters": tune.grid_search([5]),
    }
    resource_group = tune.PlacementGroupFactory([{"CPU": 1.0}] + [{"CPU": 1.0}] * 10)
    trainable_with_resources = tune.with_resources(objective, resource_group)
    timenow = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=search_space,
        run_config=RunConfig(
            name=f"hpm_r{REPEAT}_n{NUMBER}_{timenow}", local_dir="./.results"
        ),
    )
    results = tuner.fit()

from sklearn.cluster import KMeans, MiniBatchKMeans

from .common import Benchmark, Estimator, Predictor, Transformer
from .datasets import _20newsgroups_highdim_dataset, _blobs_dataset
from .utils import neg_mean_inertia


class KMeansBenchmark(Predictor, Transformer, Estimator, Benchmark):
    """
    Benchmarks for KMeans.
    """

    param_names = ["representation", "algorithm", "init"]
    params = (["dense", "sparse"], ["lloyd", "elkan"], ["random", "k-means++"])

    def setup_cache(self):
        pass

    def make_data(self, params):
        representation, algorithm, init = params

        if representation == "sparse":
            data = _20newsgroups_highdim_dataset(n_samples=8000)
        else:
            data = _blobs_dataset(n_clusters=20)

        return data

    def make_estimator(self, params):
        pass

    def make_scorers(self):
        self.train_scorer = lambda _, __: neg_mean_inertia(
            self.X, self.estimator.predict(self.X), self.estimator.cluster_centers_
        )
        self.test_scorer = lambda _, __: neg_mean_inertia(
            self.X_val,
            self.estimator.predict(self.X_val),
            self.estimator.cluster_centers_,
        )


class MiniBatchKMeansBenchmark(Predictor, Transformer, Estimator, Benchmark):
    """
    Benchmarks for MiniBatchKMeans.
    """

    param_names = ["representation", "init"]
    params = (["dense", "sparse"], ["random", "k-means++"])

    def setup_cache(self):
        pass

    def make_data(self, params):
        representation, init = params

        if representation == "sparse":
            data = _20newsgroups_highdim_dataset()
        else:
            data = _blobs_dataset(n_clusters=20)

        return data

    def make_estimator(self, params):
        pass

    def make_scorers(self):
        self.train_scorer = lambda _, __: neg_mean_inertia(
            self.X, self.estimator.predict(self.X), self.estimator.cluster_centers_
        )
        self.test_scorer = lambda _, __: neg_mean_inertia(
            self.X_val,
            self.estimator.predict(self.X_val),
            self.estimator.cluster_centers_,
        )

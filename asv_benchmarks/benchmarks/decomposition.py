from sklearn.decomposition import PCA, DictionaryLearning, MiniBatchDictionaryLearning

from .common import Benchmark, Estimator, Transformer
from .datasets import _mnist_dataset, _olivetti_faces_dataset
from .utils import make_dict_learning_scorers, make_pca_scorers


class PCABenchmark(Transformer, Estimator, Benchmark):
    """
    Benchmarks for PCA.
    """

    param_names = ["svd_solver"]
    params = (["full", "arpack", "randomized"],)

    def setup_cache(self):
        pass

    def make_data(self, params):
        return _mnist_dataset()

    def make_estimator(self, params):
        pass

    def make_scorers(self):
        make_pca_scorers(self)


class DictionaryLearningBenchmark(Transformer, Estimator, Benchmark):
    """
    Benchmarks for DictionaryLearning.
    """

    param_names = ["fit_algorithm", "n_jobs"]
    params = (["lars", "cd"], Benchmark.n_jobs_vals)

    def setup_cache(self):
        pass

    def make_data(self, params):
        return _olivetti_faces_dataset()

    def make_estimator(self, params):
        pass

    def make_scorers(self):
        make_dict_learning_scorers(self)


class MiniBatchDictionaryLearningBenchmark(Transformer, Estimator, Benchmark):
    """
    Benchmarks for MiniBatchDictionaryLearning
    """

    param_names = ["fit_algorithm", "n_jobs"]
    params = (["lars", "cd"], Benchmark.n_jobs_vals)

    def setup_cache(self):
        pass

    def make_data(self, params):
        return _olivetti_faces_dataset()

    def make_estimator(self, params):
        pass

    def make_scorers(self):
        make_dict_learning_scorers(self)

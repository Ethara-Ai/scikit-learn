from sklearn.svm import SVC

from .common import Benchmark, Estimator, Predictor
from .datasets import _synth_classification_dataset
from .utils import make_gen_classif_scorers


class SVCBenchmark(Predictor, Estimator, Benchmark):
    """Benchmarks for SVC."""

    param_names = ["kernel"]
    params = (["linear", "poly", "rbf", "sigmoid"],)

    def setup_cache(self):
        pass

    def make_data(self, params):
        return _synth_classification_dataset()

    def make_estimator(self, params):
        pass

    def make_scorers(self):
        make_gen_classif_scorers(self)

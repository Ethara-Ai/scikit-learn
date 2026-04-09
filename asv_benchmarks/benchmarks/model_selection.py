from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

from .common import Benchmark, Estimator, Predictor
from .datasets import _synth_classification_dataset
from .utils import make_gen_classif_scorers


class CrossValidationBenchmark(Benchmark):
    """
    Benchmarks for Cross Validation.
    """

    timeout = 20000

    param_names = ["n_jobs"]
    params = (Benchmark.n_jobs_vals,)

    def setup(self, *params):
        (n_jobs,) = params

        data = _synth_classification_dataset(n_samples=50000, n_features=100)
        self.X, self.X_val, self.y, self.y_val = data

        self.clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0)

        cv = 16 if Benchmark.data_size == "large" else 4

        self.cv_params = {"n_jobs": n_jobs, "cv": cv}

    def time_crossval(self, *args):
        pass

    def peakmem_crossval(self, *args):
        pass

    def track_crossval(self, *args):
        pass


class GridSearchBenchmark(Predictor, Estimator, Benchmark):
    """
    Benchmarks for GridSearch.
    """

    timeout = 20000

    param_names = ["n_jobs"]
    params = (Benchmark.n_jobs_vals,)

    def setup_cache(self):
        pass

    def make_data(self, params):
        data = _synth_classification_dataset(n_samples=10000, n_features=100)

        return data

    def make_estimator(self, params):
        pass

    def make_scorers(self):
        make_gen_classif_scorers(self)

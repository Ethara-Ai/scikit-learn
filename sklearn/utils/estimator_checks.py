"""Various utilities to check the compatibility of estimators with scikit-learn API."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import pickle
import re
import textwrap
import warnings
from contextlib import nullcontext
from copy import deepcopy
from functools import partial, wraps
from inspect import signature
from numbers import Integral, Real
from typing import Callable, Literal

import joblib
import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from sklearn import config_context
from sklearn.base import (
    BaseEstimator,
    BiclusterMixin,
    ClassifierMixin,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    DensityMixin,
    MetaEstimatorMixin,
    MultiOutputMixin,
    OneToOneFeatureMixin,
    OutlierMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
    is_classifier,
    is_outlier_detector,
    is_regressor,
)
from sklearn.datasets import (
    load_iris,
    make_blobs,
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.exceptions import (
    DataConversionWarning,
    EstimatorCheckFailedWarning,
    NotFittedError,
    SkipTestWarning,
)
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score
from sklearn.metrics.pairwise import linear_kernel, pairwise_distances, rbf_kernel
from sklearn.model_selection import LeaveOneGroupOut, ShuffleSplit, train_test_split
from sklearn.model_selection._validation import _safe_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.utils import _safe_indexing, shuffle
from sklearn.utils._array_api import (
    _atol_for_type,
    get_namespace,
    move_to,
    yield_namespace_device_dtype_combinations,
)
from sklearn.utils._array_api import device as array_device
from sklearn.utils._missing import is_scalar_nan
from sklearn.utils._param_validation import (
    Interval,
    InvalidParameterError,
    StrOptions,
    generate_invalid_param_val,
    make_constraint,
    validate_params,
)
from sklearn.utils._tags import (
    ClassifierTags,
    InputTags,
    RegressorTags,
    TargetTags,
    TransformerTags,
    get_tags,
)
from sklearn.utils._test_common.instance_generator import (
    CROSS_DECOMPOSITION,
    _get_check_estimator_ids,
    _yield_instances_for_check,
)
from sklearn.utils._testing import (
    SkipTest,
    _array_api_for_tests,
    _get_args,
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
    create_memmap_backed_data,
    ignore_warnings,
    raises,
    set_random_state,
)
from sklearn.utils.validation import _num_samples, check_is_fitted, has_fit_parameter

REGRESSION_DATASET = None


def _raise_for_missing_tags(estimator, tag_name, Mixin):
    tags = get_tags(estimator)
    estimator_type = Mixin.__name__.replace("Mixin", "")
    if getattr(tags, tag_name) is None:
        raise RuntimeError(
            f"Estimator {estimator.__class__.__name__} seems to be a {estimator_type},"
            f" but the `{tag_name}` tag is not set. Either set the tag manually"
            f" or inherit from the {Mixin.__name__}. Note that the order of inheritance"
            f" matters, the {Mixin.__name__} should come before BaseEstimator."
        )


def _yield_api_checks(estimator):
    if not isinstance(estimator, BaseEstimator):
        warnings.warn(
            f"Estimator {estimator.__class__.__name__} does not inherit from"
            " `sklearn.base.BaseEstimator`. This might lead to unexpected behavior, or"
            " even errors when collecting tests.",
            category=UserWarning,
        )

    tags = get_tags(estimator)
    # This is commented out since it's the first check both
    # `parametrize_with_checks` and `check_esitmator` do
    # anyway. But leaving it here as commented out to know
    # it's a part of the basic API.
    # yield check_estimator_cloneable
    yield check_estimator_tags_renamed
    yield check_valid_tag_types
    yield check_estimator_repr
    yield check_no_attributes_set_in_init
    yield check_fit_score_takes_y
    yield check_estimators_overwrite_params
    yield check_dont_overwrite_parameters
    yield check_estimators_fit_returns_self
    yield check_readonly_memmap_input
    if tags.requires_fit:
        yield check_estimators_unfitted
    yield check_do_not_raise_errors_in_init_or_set_params
    yield check_n_features_in_after_fitting
    yield check_mixin_order
    yield check_positive_only_tag_during_fit


def _yield_checks(estimator):
    name = estimator.__class__.__name__
    tags = get_tags(estimator)

    yield check_estimators_dtypes
    if has_fit_parameter(estimator, "sample_weight"):
        yield check_sample_weights_pandas_series
        yield check_sample_weights_not_an_array
        yield check_sample_weights_list
        yield check_all_zero_sample_weights_error
        if not tags.input_tags.pairwise:
            # We skip pairwise because the data is not pairwise
            yield check_sample_weights_shape
            yield check_sample_weights_not_overwritten
            yield check_sample_weight_equivalence_on_dense_data
            if tags.input_tags.sparse:
                yield check_sample_weight_equivalence_on_sparse_data

    # Check that all estimator yield informative messages when
    # trained on empty datasets
    if not tags.no_validation:
        yield check_complex_data
        yield check_dtype_object
        yield check_estimators_empty_data_messages

    if name not in CROSS_DECOMPOSITION:
        # cross-decomposition's "transform" returns X and Y
        yield check_pipeline_consistency

    if not tags.input_tags.allow_nan and not tags.no_validation:
        # Test that all estimators check their input for NaN's and infs
        yield check_estimators_nan_inf

    if tags.input_tags.pairwise:
        # Check that pairwise estimator throws error on non-square input
        yield check_nonsquare_error

    if hasattr(estimator, "sparsify"):
        yield check_sparsify_coefficients

    yield check_estimator_sparse_tag
    yield check_estimator_sparse_array
    yield check_estimator_sparse_matrix

    # Test that estimators can be pickled, and once pickled
    # give the same answer as before.
    yield check_estimators_pickle
    yield partial(check_estimators_pickle, readonly_memmap=True)

    for check in _yield_array_api_checks(
        estimator,
        only_numpy=not tags.array_api_support,
    ):
        yield check

    yield check_f_contiguous_array_estimator


def _yield_classifier_checks(classifier):
    _raise_for_missing_tags(classifier, "classifier_tags", ClassifierMixin)
    tags = get_tags(classifier)

    # test classifiers can handle non-array data and pandas objects
    yield check_classifier_data_not_an_array
    # test classifiers trained on a single label always return this label
    yield check_classifiers_one_label
    yield check_classifiers_one_label_sample_weights
    yield check_classifiers_classes
    yield check_estimators_partial_fit_n_features
    if tags.target_tags.multi_output:
        yield check_classifier_multioutput
    # basic consistency testing
    yield check_classifiers_train
    yield partial(check_classifiers_train, readonly_memmap=True)
    yield partial(check_classifiers_train, readonly_memmap=True, X_dtype="float32")
    yield check_classifiers_regression_target
    if tags.classifier_tags.multi_label:
        yield check_classifiers_multilabel_representation_invariance
        yield check_classifiers_multilabel_output_format_predict
        yield check_classifiers_multilabel_output_format_predict_proba
        yield check_classifiers_multilabel_output_format_decision_function
    if not tags.no_validation:
        yield check_supervised_y_no_nan
        if tags.target_tags.single_output:
            yield check_supervised_y_2d
    if "class_weight" in classifier.get_params().keys():
        yield check_class_weight_classifiers

    yield check_non_transformer_estimators_n_iter
    # test if predict_proba is a monotonic transformation of decision_function
    yield check_decision_proba_consistency

    if (
        isinstance(classifier, LinearClassifierMixin)
        and "class_weight" in classifier.get_params().keys()
    ):
        yield check_class_weight_balanced_linear_classifier

    if not tags.classifier_tags.multi_class:
        yield check_classifier_not_supporting_multiclass


def _yield_regressor_checks(regressor):
    _raise_for_missing_tags(regressor, "regressor_tags", RegressorMixin)
    tags = get_tags(regressor)
    # TODO: test with intercept
    # TODO: test with multiple responses
    # basic testing
    yield check_regressors_train
    yield partial(check_regressors_train, readonly_memmap=True)
    yield partial(check_regressors_train, readonly_memmap=True, X_dtype="float32")
    yield check_regressor_data_not_an_array
    yield check_estimators_partial_fit_n_features
    if tags.target_tags.multi_output:
        yield check_regressor_multioutput
    yield check_regressors_no_decision_function
    if not tags.no_validation and tags.target_tags.single_output:
        yield check_supervised_y_2d
    yield check_supervised_y_no_nan
    name = regressor.__class__.__name__
    if name != "CCA":
        # check that the regressor handles int input
        yield check_regressors_int
    yield check_non_transformer_estimators_n_iter


def _yield_transformer_checks(transformer):
    _raise_for_missing_tags(transformer, "transformer_tags", TransformerMixin)
    tags = get_tags(transformer)
    # All transformers should either deal with sparse data or raise an
    # exception with type TypeError and an intelligible error message
    if not tags.no_validation:
        yield check_transformer_data_not_an_array
    # these don't actually fit the data, so don't raise errors
    yield check_transformer_general
    if tags.transformer_tags.preserves_dtype:
        yield check_transformer_preserve_dtypes
    yield partial(check_transformer_general, readonly_memmap=True)
    if get_tags(transformer).requires_fit:
        yield check_transformers_unfitted
    else:
        yield check_transformers_unfitted_stateless
    # Dependent on external solvers and hence accessing the iter
    # param is non-trivial.
    external_solver = [
        "Isomap",
        "KernelPCA",
        "LocallyLinearEmbedding",
        "LogisticRegressionCV",
        "BisectingKMeans",
    ]

    name = transformer.__class__.__name__
    if name not in external_solver:
        yield check_transformer_n_iter


def _yield_clustering_checks(clusterer):
    yield check_clusterer_compute_labels_predict
    name = clusterer.__class__.__name__
    if name not in ("WardAgglomeration", "FeatureAgglomeration"):
        # this is clustering on the features
        # let's not test that here.
        yield check_clustering
        yield partial(check_clustering, readonly_memmap=True)
        yield check_estimators_partial_fit_n_features
    if not hasattr(clusterer, "transform"):
        yield check_non_transformer_estimators_n_iter


def _yield_outliers_checks(estimator):
    # checks for the contamination parameter
    if hasattr(estimator, "contamination"):
        yield check_outlier_contamination

    # checks for outlier detectors that have a fit_predict method
    if hasattr(estimator, "fit_predict"):
        yield check_outliers_fit_predict

    # checks for estimators that can be used on a test set
    if hasattr(estimator, "predict"):
        yield check_outliers_train
        yield partial(check_outliers_train, readonly_memmap=True)
        # test outlier detectors can handle non-array data
        yield check_classifier_data_not_an_array
    yield check_non_transformer_estimators_n_iter


def _yield_array_api_checks(estimator, only_numpy=False):
    if only_numpy:
        # Enabling array API dispatch and using NumPy inputs should not
        # change results, even if the estimator does not explicitly support
        # array API.
        yield partial(
            check_array_api_input,
            array_namespace="numpy",
            expect_only_array_outputs=False,
        )
    else:
        # These extended checks should pass for all estimators that declare
        # array API support in their tags.
        for (
            array_namespace,
            device_name,
            dtype_name,
        ) in yield_namespace_device_dtype_combinations():
            yield partial(
                check_array_api_input,
                array_namespace=array_namespace,
                device_name=device_name,
                dtype_name=dtype_name,
            )
        # Only test with one namespace to keep costs down
        # There should be no dependency on the exact
        # namespace used.
        yield partial(
            check_array_api_same_namespace,
            array_namespace="array_api_strict",
        )


def _yield_all_checks(estimator, legacy: bool):
    name = estimator.__class__.__name__
    tags = get_tags(estimator)
    if not tags.input_tags.two_d_array:
        warnings.warn(
            "Can't test estimator {} which requires input  of type {}".format(
                name, tags.input_tags
            ),
            SkipTestWarning,
        )
        return
    if tags._skip_test:
        warnings.warn(
            "Explicit SKIP via _skip_test tag for estimator {}.".format(name),
            SkipTestWarning,
        )
        return

    for check in _yield_api_checks(estimator):
        yield check

    if not legacy:
        return  # pragma: no cover

    for check in _yield_checks(estimator):
        yield check
    if is_classifier(estimator):
        for check in _yield_classifier_checks(estimator):
            yield check
    if is_regressor(estimator):
        for check in _yield_regressor_checks(estimator):
            yield check
    if hasattr(estimator, "transform"):
        for check in _yield_transformer_checks(estimator):
            yield check
    if isinstance(estimator, ClusterMixin):
        for check in _yield_clustering_checks(estimator):
            yield check
    if is_outlier_detector(estimator):
        for check in _yield_outliers_checks(estimator):
            yield check
    yield check_parameters_default_constructible
    if not tags.non_deterministic:
        yield check_methods_sample_order_invariance
        yield check_methods_subset_invariance
    yield check_fit2d_1sample
    yield check_fit2d_1feature
    yield check_get_params_invariance
    yield check_set_params
    yield check_dict_unchanged
    yield check_fit_idempotent
    yield check_fit_check_is_fitted
    if not tags.no_validation:
        yield check_n_features_in
        yield check_fit1d
        yield check_fit2d_predict1d
        if tags.target_tags.required:
            yield check_requires_y_none
    if tags.input_tags.positive_only:
        yield check_fit_non_negative


def _check_name(check):
    if hasattr(check, "__wrapped__"):
        return _check_name(check.__wrapped__)
    return check.func.__name__ if isinstance(check, partial) else check.__name__


def _maybe_mark(
    estimator,
    check,
    expected_failed_checks: dict[str, str] | None = None,
    mark: Literal["xfail", "skip", None] = None,
    pytest=None,
    xfail_strict: bool | None = None,
):
    """Mark the test as xfail or skip if needed.

    Parameters
    ----------
    estimator : estimator object
        Estimator instance for which to generate checks.
    check : partial or callable
        Check to be marked.
    expected_failed_checks : dict[str, str], default=None
        Dictionary of the form {check_name: reason} for checks that are expected to
        fail.
    mark : "xfail" or "skip" or None
        Whether to mark the check as xfail or skip.
    pytest : pytest module, default=None
        Pytest module to use to mark the check. This is only needed if ``mark`` is
        `"xfail"`. Note that one can run `check_estimator` without having `pytest`
        installed. This is used in combination with `parametrize_with_checks` only.
    xfail_strict : bool, default=None
        Whether to run checks in xfail strict mode. This option is ignored unless
        `mark="xfail"`. If True, checks that are expected to fail but actually
        pass will lead to a test failure. If False, unexpectedly passing tests
        will be marked as xpass. If None, the default pytest behavior is used.

        .. versionadded:: 1.8
    """
    should_be_marked, reason = _should_be_skipped_or_marked(
        estimator, check, expected_failed_checks
    )
    if not should_be_marked or mark is None:
        return estimator, check

    estimator_name = estimator.__class__.__name__
    if mark == "xfail":
        # With xfail_strict=None we want the value from the pytest config to
        # take precedence and that means not passing strict to the xfail
        # mark at all.
        if xfail_strict is None:
            mark = pytest.mark.xfail(reason=reason)
        else:
            mark = pytest.mark.xfail(reason=reason, strict=xfail_strict)
        return pytest.param(estimator, check, marks=mark)
    else:

        @wraps(check)
        def wrapped(*args, **kwargs):
            raise SkipTest(
                f"Skipping {_check_name(check)} for {estimator_name}: {reason}"
            )

        return estimator, wrapped


def _should_be_skipped_or_marked(
    estimator, check, expected_failed_checks: dict[str, str] | None = None
) -> tuple[bool, str]:
    """Check whether a check should be skipped or marked as xfail.

    Parameters
    ----------
    estimator : estimator object
        Estimator instance for which to generate checks.
    check : partial or callable
        Check to be marked.
    expected_failed_checks : dict[str, str], default=None
        Dictionary of the form {check_name: reason} for checks that are expected to
        fail.

    Returns
    -------
    should_be_marked : bool
        Whether the check should be marked as xfail or skipped.
    reason : str
        Reason for skipping the check.
    """

    expected_failed_checks = expected_failed_checks or {}

    check_name = _check_name(check)
    if check_name in expected_failed_checks:
        return True, expected_failed_checks[check_name]

    return False, "Check is not expected to fail"


def estimator_checks_generator(
    estimator,
    *,
    legacy: bool = True,
    expected_failed_checks: dict[str, str] | None = None,
    mark: Literal["xfail", "skip", None] = None,
    xfail_strict: bool | None = None,
):
    """Iteratively yield all check callables for an estimator.

    This function is used by
    :func:`~sklearn.utils.estimator_checks.parametrize_with_checks` and
    :func:`~sklearn.utils.estimator_checks.check_estimator` to yield all check callables
    for an estimator. In most cases, these functions should be used instead. When
    implementing a custom equivalent, please refer to their source code to
    understand how `estimator_checks_generator` is intended to be used.

    .. versionadded:: 1.6

    Parameters
    ----------
    estimator : estimator object
        Estimator instance for which to generate checks.
    legacy : bool, default=True
        Whether to include legacy checks. Over time we remove checks from this category
        and move them into their specific category.
    expected_failed_checks : dict[str, str], default=None
        Dictionary of the form {check_name: reason} for checks that are expected to
        fail.
    mark : {"xfail", "skip"} or None, default=None
        Whether to mark the checks that are expected to fail as
        xfail(`pytest.mark.xfail`) or skip. Marking a test as "skip" is done via
        wrapping the check in a function that raises a
        :class:`~sklearn.exceptions.SkipTest` exception.
    xfail_strict : bool, default=None
        Whether to run checks in xfail strict mode. This option is ignored unless
        `mark="xfail"`. If True, checks that are expected to fail but actually
        pass will lead to a test failure. If False, unexpectedly passing tests
        will be marked as xpass. If None, the default pytest behavior is used.

        .. versionadded:: 1.8

    Returns
    -------
    estimator_checks_generator : generator
        Generator that yields (estimator, check) tuples.
    """
    if mark == "xfail":
        import pytest
    else:
        pytest = None  # type: ignore[assignment]

    name = type(estimator).__name__
    # First check that the estimator is cloneable which is needed for the rest
    # of the checks to run
    yield estimator, partial(check_estimator_cloneable, name)
    for check in _yield_all_checks(estimator, legacy=legacy):
        check_with_name = partial(check, name)
        for check_instance in _yield_instances_for_check(check, estimator):
            yield _maybe_mark(
                check_instance,
                check_with_name,
                expected_failed_checks=expected_failed_checks,
                mark=mark,
                pytest=pytest,
                xfail_strict=xfail_strict,
            )


def parametrize_with_checks(
    estimators,
    *,
    legacy: bool = True,
    expected_failed_checks: Callable | None = None,
    xfail_strict: bool | None = None,
):
    """Pytest specific decorator for parametrizing estimator checks.

    Checks are categorised into the following groups:

    - API checks: a set of checks to ensure API compatibility with scikit-learn.
      Refer to https://scikit-learn.org/dev/developers/develop.html a requirement of
      scikit-learn estimators.
    - legacy: a set of checks which gradually will be grouped into other categories.

    The `id` of each check is set to be a pprint version of the estimator
    and the name of the check with its keyword arguments.
    This allows to use `pytest -k` to specify which tests to run::

        pytest test_check_estimators.py -k check_estimators_fit_returns_self

    Parameters
    ----------
    estimators : list of estimators instances
        Estimators to generated checks for.

        .. versionchanged:: 0.24
           Passing a class was deprecated in version 0.23, and support for
           classes was removed in 0.24. Pass an instance instead.

        .. versionadded:: 0.24


    legacy : bool, default=True
        Whether to include legacy checks. Over time we remove checks from this category
        and move them into their specific category.

        .. versionadded:: 1.6

    expected_failed_checks : callable, default=None
        A callable that takes an estimator as input and returns a dictionary of the
        form::

            {
                "check_name": "my reason",
            }

        Where `"check_name"` is the name of the check, and `"my reason"` is why
        the check fails. These tests will be marked as xfail if the check fails.

        .. versionadded:: 1.6

    xfail_strict : bool, default=None
        Whether to run checks in xfail strict mode. If True, checks that are
        expected to fail but actually pass will lead to a test failure. If
        False, unexpectedly passing tests will be marked as xpass. If None,
        the default pytest behavior is used.

        .. versionadded:: 1.8

    Returns
    -------
    decorator : `pytest.mark.parametrize`

    See Also
    --------
    check_estimator : Check if estimator adheres to scikit-learn conventions.

    Examples
    --------
    >>> from sklearn.utils.estimator_checks import parametrize_with_checks
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeRegressor

    >>> @parametrize_with_checks([LogisticRegression(),
    ...                           DecisionTreeRegressor()])
    ... def test_sklearn_compatible_estimator(estimator, check):
    ...     check(estimator)

    """
    import pytest

    if any(isinstance(est, type) for est in estimators):
        msg = (
            "Passing a class was deprecated in version 0.23 "
            "and isn't supported anymore from 0.24."
            "Please pass an instance instead."
        )
        raise TypeError(msg)

    def _checks_generator(estimators, legacy, expected_failed_checks):
        for estimator in estimators:
            args = {
                "estimator": estimator,
                "legacy": legacy,
                "mark": "xfail",
                "xfail_strict": xfail_strict,
            }
            if callable(expected_failed_checks):
                args["expected_failed_checks"] = expected_failed_checks(estimator)
            yield from estimator_checks_generator(**args)

    return pytest.mark.parametrize(
        "estimator, check",
        _checks_generator(estimators, legacy, expected_failed_checks),
        ids=_get_check_estimator_ids,
    )


@validate_params(
    {
        "legacy": ["boolean"],
        "expected_failed_checks": [dict, None],
        "on_skip": [StrOptions({"warn"}), None],
        "on_fail": [StrOptions({"raise", "warn"}), None],
        "callback": [callable, None],
    },
    prefer_skip_nested_validation=False,
)
def check_estimator(
    estimator=None,
    *,
    legacy: bool = True,
    expected_failed_checks: dict[str, str] | None = None,
    on_skip: Literal["warn"] | None = "warn",
    on_fail: Literal["raise", "warn"] | None = "raise",
    callback: Callable | None = None,
):
    """Check if estimator adheres to scikit-learn conventions.

    This function will run an extensive test-suite for input validation,
    shapes, etc, making sure that the estimator complies with `scikit-learn`
    conventions as detailed in :ref:`rolling_your_own_estimator`.
    Additional tests for classifiers, regressors, clustering or transformers
    will be run if the Estimator class inherits from the corresponding mixin
    from sklearn.base.

    scikit-learn also provides a pytest specific decorator,
    :func:`~sklearn.utils.estimator_checks.parametrize_with_checks`, making it
    easier to test multiple estimators.

    Checks are categorised into the following groups:

    - API checks: a set of checks to ensure API compatibility with scikit-learn.
      Refer to https://scikit-learn.org/dev/developers/develop.html a requirement of
      scikit-learn estimators.
    - legacy: a set of checks which gradually will be grouped into other categories.

    Parameters
    ----------
    estimator : estimator object
        Estimator instance to check.

    legacy : bool, default=True
        Whether to include legacy checks. Over time we remove checks from this category
        and move them into their specific category.

        .. versionadded:: 1.6

    expected_failed_checks : dict, default=None
        A dictionary of the form::

            {
                "check_name": "this check is expected to fail because ...",
            }

        Where `"check_name"` is the name of the check, and `"my reason"` is why
        the check fails.

        .. versionadded:: 1.6

    on_skip : "warn", None, default="warn"
        This parameter controls what happens when a check is skipped.

        - "warn": A :class:`~sklearn.exceptions.SkipTestWarning` is logged
          and running tests continue.
        - None: No warning is logged and running tests continue.

        .. versionadded:: 1.6

    on_fail : {"raise", "warn"}, None, default="raise"
        This parameter controls what happens when a check fails.

        - "raise": The exception raised by the first failing check is raised and
          running tests are aborted. This does not included tests that are expected
          to fail.
        - "warn": A :class:`~sklearn.exceptions.EstimatorCheckFailedWarning` is logged
          and running tests continue.
        - None: No exception is raised and no warning is logged.

        Note that if ``on_fail != "raise"``, no exception is raised, even if the checks
        fail. You'd need to inspect the return result of ``check_estimator`` to check
        if any checks failed.

        .. versionadded:: 1.6

    callback : callable, or None, default=None
        This callback will be called with the estimator and the check name,
        the exception (if any), the status of the check (xfail, failed, skipped,
        passed), and the reason for the expected failure if the check is
        expected to fail. The callable's signature needs to be::

            def callback(
                estimator,
                check_name: str,
                exception: Exception,
                status: Literal["xfail", "failed", "skipped", "passed"],
                expected_to_fail: bool,
                expected_to_fail_reason: str,
            )

        ``callback`` cannot be provided together with ``on_fail="raise"``.

        .. versionadded:: 1.6

    Returns
    -------
    test_results : list
        List of dictionaries with the results of the failing tests, of the form::

            {
                "estimator": estimator,
                "check_name": check_name,
                "exception": exception,
                "status": status (one of "xfail", "failed", "skipped", "passed"),
                "expected_to_fail": expected_to_fail,
                "expected_to_fail_reason": expected_to_fail_reason,
            }

    Raises
    ------
    Exception
        If ``on_fail="raise"``, the exception raised by the first failing check is
        raised and running tests are aborted.

        Note that if ``on_fail != "raise"``, no exception is raised, even if the checks
        fail. You'd need to inspect the return result of ``check_estimator`` to check
        if any checks failed.

    See Also
    --------
    parametrize_with_checks : Pytest specific decorator for parametrizing estimator
        checks.
    estimator_checks_generator : Generator that yields (estimator, check) tuples.

    Examples
    --------
    >>> from sklearn.utils.estimator_checks import check_estimator
    >>> from sklearn.linear_model import LogisticRegression
    >>> check_estimator(LogisticRegression())
    [...]
    """
    pass


def _regression_dataset():
    pass


class _NotAnArray:
    """An object that is convertible to an array.

    Parameters
    ----------
    data : array-like
        The data.
    """

    def __init__(self, data):
        self.data = np.asarray(data)

    def __array__(self, dtype=None, copy=None):
        return self.data

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ == "may_share_memory":
            return True
        raise TypeError("Don't want to call array_function {}!".format(func.__name__))


def _is_pairwise_metric(estimator):
    """Returns True if estimator accepts pairwise metric.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if _pairwise is set to True and False otherwise.
    """
    pass


def _generate_sparse_data(X_csr):
    """Generate sparse matrices or arrays with {32,64}bit indices of diverse format.

    Parameters
    ----------
    X_csr: scipy.sparse.csr_matrix or scipy.sparse.csr_array
        Input in CSR format.

    Returns
    -------
    out: iter(Matrices) or iter(Arrays)
        In format['dok', 'lil', 'dia', 'bsr', 'csr', 'csc', 'coo',
        'coo_64', 'csc_64', 'csr_64']
    """
    pass


@ignore_warnings(category=FutureWarning)
def check_supervised_y_no_nan(name, estimator_orig):
    # Checks that the Estimator targets are not NaN.
    pass


def check_array_api_input(
    name,
    estimator_orig,
    array_namespace,
    device_name=None,
    dtype_name="float64",
    check_values=False,
    check_sample_weight=False,
    expect_only_array_outputs=True,
):
    """Check that the estimator can work consistently with the Array API

    By default, this just checks that the types and shapes of the arrays are
    consistent with calling the same estimator with numpy arrays.

    When check_values is True, it also checks that calling the estimator on the
    array_api Array gives the same results as ndarrays.

    When check_sample_weight is True, dummy sample weights are passed to the
    fit call.

    When expect_only_array_outputs is False, the check is looser: in particular
    it accepts non-array outputs such as sparse data structures. This is
    useful to test that enabling array API dispatch does not change the
    behavior of any estimator fed with NumPy inputs, even for estimators that
    do not support array API.
    """
    pass


def check_array_api_input_and_values(
    name,
    estimator_orig,
    array_namespace,
    device_name=None,
    dtype_name="float64",
    check_sample_weight=False,
):
    pass


def check_array_api_same_namespace(
    name, estimator_orig, array_namespace, device_name=None
):
    """Check that estimator raises when predict/transform namespace differs from fit.

    Array API compatible estimators should call ``check_same_namespace`` in
    their ``predict``, ``transform``, and similar methods to verify that the
    input arrays are from the same namespace and device as the fitted
    attributes.
    """
    pass


def check_estimator_sparse_tag(name, estimator_orig):
    """Check that estimator tag related with accepting sparse data is properly set."""
    pass


def _check_estimator_sparse_container(name, estimator_orig, sparse_type):
    pass


def check_estimator_sparse_matrix(name, estimator_orig):
    pass


def check_estimator_sparse_array(name, estimator_orig):
    pass


def check_f_contiguous_array_estimator(name, estimator_orig):
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/23988
    # https://github.com/scikit-learn/scikit-learn/issues/24013
    pass


@ignore_warnings(category=FutureWarning)
def check_sample_weights_pandas_series(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type pandas.Series in the 'fit' function.
    pass


@ignore_warnings(category=(FutureWarning))
def check_sample_weights_not_an_array(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type _NotAnArray in the 'fit' function.
    pass


@ignore_warnings(category=(FutureWarning))
def check_sample_weights_list(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type list in the 'fit' function.
    pass


@ignore_warnings(category=FutureWarning)
def check_all_zero_sample_weights_error(name, estimator_orig):
    """Check that estimator raises error when all sample weights are 0."""
    pass


@ignore_warnings(category=FutureWarning)
def check_sample_weights_shape(name, estimator_orig):
    # check that estimators raise an error if sample_weight
    # shape mismatches the input
    pass


@ignore_warnings(category=FutureWarning)
def _check_sample_weight_equivalence(name, estimator_orig, sparse_container):
    # check that setting sample_weight to zero / integer is equivalent
    # to removing / repeating corresponding samples.
    pass


def check_sample_weight_equivalence_on_dense_data(name, estimator_orig):
    pass


def check_sample_weight_equivalence_on_sparse_data(name, estimator_orig):
    pass


def check_sample_weights_not_overwritten(name, estimator_orig):
    # check that estimators don't override the passed sample_weight parameter
    pass


@ignore_warnings(category=(FutureWarning, UserWarning))
def check_dtype_object(name, estimator_orig):
    # check that estimators treat dtype object as numeric if possible
    pass


def check_complex_data(name, estimator_orig):
    pass


@ignore_warnings
def check_dict_unchanged(name, estimator_orig):
    pass


def _is_public_parameter(attr):
    pass


@ignore_warnings(category=FutureWarning)
def check_dont_overwrite_parameters(name, estimator_orig):
    # check that fit method only changes or sets private attributes
    pass


@ignore_warnings(category=FutureWarning)
def check_fit2d_predict1d(name, estimator_orig):
    # check by fitting a 2d array and predicting with a 1d array
    pass


def _apply_on_subsets(func, X):
    # apply function on the whole set and on mini batches
    pass


@ignore_warnings(category=FutureWarning)
def check_methods_subset_invariance(name, estimator_orig):
    # check that method gives invariant results if applied
    # on mini batches or the whole set
    pass


@ignore_warnings(category=FutureWarning)
def check_methods_sample_order_invariance(name, estimator_orig):
    # check that method gives invariant results if applied
    # on a subset with different sample order
    pass


@ignore_warnings
def check_fit2d_1sample(name, estimator_orig):
    # Check that fitting a 2d array with only one sample either works or
    # returns an informative message. The error message should either mention
    # the number of samples or the number of classes.
    pass


@ignore_warnings
def check_fit2d_1feature(name, estimator_orig):
    # check fitting a 2d array with only 1 feature either works or returns
    # informative message
    pass


@ignore_warnings
def check_fit1d(name, estimator_orig):
    # check fitting 1d X array raises a ValueError
    pass


@ignore_warnings(category=FutureWarning)
def check_transformer_general(name, transformer, readonly_memmap=False):
    pass


@ignore_warnings(category=FutureWarning)
def check_transformer_data_not_an_array(name, transformer):
    pass


@ignore_warnings(category=FutureWarning)
def check_transformers_unfitted(name, transformer):
    pass


@ignore_warnings(category=FutureWarning)
def check_transformers_unfitted_stateless(name, transformer):
    """Check that using transform without prior fitting
    doesn't raise a NotFittedError for stateless transformers.
    """
    pass


def _check_transformer(name, transformer_orig, X, y):
    pass


@ignore_warnings
def check_pipeline_consistency(name, estimator_orig):
    pass


@ignore_warnings
def check_mixin_order(name, estimator_orig):
    """Check that mixins are inherited in the correct order."""
    pass


@ignore_warnings
def check_fit_score_takes_y(name, estimator_orig):
    # check that all estimators accept an optional y
    # in fit and score so they can be used in pipelines
    pass


@ignore_warnings
def check_estimators_dtypes(name, estimator_orig):
    pass


def check_transformer_preserve_dtypes(name, transformer_orig):
    # check that dtype are preserved meaning if input X is of some dtype
    # X_transformed should be from the same dtype.
    pass


@ignore_warnings(category=FutureWarning)
def check_estimators_empty_data_messages(name, estimator_orig):
    pass


@ignore_warnings(category=FutureWarning)
def check_estimators_nan_inf(name, estimator_orig):
    # Checks that Estimator X's do not contain NaN or inf.
    pass


@ignore_warnings
def check_nonsquare_error(name, estimator_orig):
    """Test that error is thrown when non-square data provided."""
    pass


@ignore_warnings
def check_estimators_pickle(name, estimator_orig, readonly_memmap=False):
    """Test that we can pickle all estimators."""
    pass


@ignore_warnings(category=FutureWarning)
def check_estimators_partial_fit_n_features(name, estimator_orig):
    # check if number of features changes between calls to partial_fit.
    pass


@ignore_warnings(category=FutureWarning)
def check_classifier_multioutput(name, estimator_orig):
    pass


@ignore_warnings(category=FutureWarning)
def check_regressor_multioutput(name, estimator):
    pass


@ignore_warnings(category=FutureWarning)
def check_clustering(name, clusterer_orig, readonly_memmap=False):
    pass
    # else labels should be less than max(labels_) which is necessarily true


@ignore_warnings(category=FutureWarning)
def check_clusterer_compute_labels_predict(name, clusterer_orig):
    """Check that predict is invariant of compute_labels."""
    pass


@ignore_warnings(category=FutureWarning)
def check_classifiers_one_label(name, classifier_orig):
    pass


@ignore_warnings(category=FutureWarning)
def check_classifiers_one_label_sample_weights(name, classifier_orig):
    """Check that classifiers accepting sample_weight fit or throws a ValueError with
    an explicit message if the problem is reduced to one class.
    """
    pass


@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(
    name, classifier_orig, readonly_memmap=False, X_dtype="float64"
):
    pass


def check_outlier_corruption(num_outliers, expected_outliers, decision):
    # Check for deviation from the precise given contamination level that may
    # be due to ties in the anomaly scores.
    pass


def check_outliers_train(name, estimator_orig, readonly_memmap=True):
    pass


def check_outlier_contamination(name, estimator_orig):
    # Check that the contamination parameter is in (0.0, 0.5] when it is an
    # interval constraint.

    pass


@ignore_warnings(category=FutureWarning)
def check_classifiers_multilabel_representation_invariance(name, classifier_orig):
    pass


@ignore_warnings(category=FutureWarning)
def check_classifiers_multilabel_output_format_predict(name, classifier_orig):
    """Check the output of the `predict` method for classifiers supporting
    multilabel-indicator targets."""
    pass


@ignore_warnings(category=FutureWarning)
def check_classifiers_multilabel_output_format_predict_proba(name, classifier_orig):
    """Check the output of the `predict_proba` method for classifiers supporting
    multilabel-indicator targets."""
    pass


@ignore_warnings(category=FutureWarning)
def check_classifiers_multilabel_output_format_decision_function(name, classifier_orig):
    """Check the output of the `decision_function` method for classifiers supporting
    multilabel-indicator targets."""
    pass


@ignore_warnings(category=FutureWarning)
def check_get_feature_names_out_error(name, estimator_orig):
    """Check the error raised by get_feature_names_out when called before fit.

    Unfitted estimators with get_feature_names_out should raise a NotFittedError.
    """
    pass


@ignore_warnings(category=FutureWarning)
def check_estimators_fit_returns_self(name, estimator_orig):
    """Check if self is returned when calling fit."""
    pass


@ignore_warnings(category=FutureWarning)
def check_readonly_memmap_input(name, estimator_orig):
    """Check that the estimator can handle readonly memmap backed data.

    This is particularly needed to support joblib parallelisation.
    """
    pass


@ignore_warnings
def check_estimators_unfitted(name, estimator_orig):
    """Check that predict raises an exception in an unfitted estimator.

    Unfitted estimators should raise a NotFittedError.
    """
    pass


@ignore_warnings(category=FutureWarning)
def check_supervised_y_2d(name, estimator_orig):
    pass


@ignore_warnings
def check_classifiers_predictions(X, y, name, classifier_orig):
    pass


def _choose_check_classifiers_labels(name, y, y_names):
    # Semisupervised classifiers use -1 as the indicator for an unlabeled
    # sample.
    pass


def check_classifiers_classes(name, classifier_orig):
    pass


@ignore_warnings(category=FutureWarning)
def check_regressors_int(name, regressor_orig):
    pass


@ignore_warnings(category=FutureWarning)
def check_regressors_train(
    name, regressor_orig, readonly_memmap=False, X_dtype=np.float64
):
    pass


@ignore_warnings
def check_regressors_no_decision_function(name, regressor_orig):
    # check that regressors don't have a decision_function, predict_proba, or
    # predict_log_proba method.
    pass


@ignore_warnings(category=FutureWarning)
def check_class_weight_classifiers(name, classifier_orig):
    pass


@ignore_warnings(category=FutureWarning)
def check_class_weight_balanced_classifiers(
    name, classifier_orig, X_train, y_train, X_test, y_test, weights
):
    pass


@ignore_warnings(category=FutureWarning)
def check_class_weight_balanced_linear_classifier(name, estimator_orig):
    """Test class weights with non-contiguous class labels."""
    pass


@ignore_warnings(category=FutureWarning)
def check_estimators_overwrite_params(name, estimator_orig):
    pass


@ignore_warnings(category=FutureWarning)
def check_no_attributes_set_in_init(name, estimator_orig):
    """Check setting during init."""
    pass


@ignore_warnings(category=FutureWarning)
def check_sparsify_coefficients(name, estimator_orig):
    pass


@ignore_warnings(category=FutureWarning)
def check_classifier_data_not_an_array(name, estimator_orig):
    pass


@ignore_warnings(category=FutureWarning)
def check_regressor_data_not_an_array(name, estimator_orig):
    pass


@ignore_warnings(category=FutureWarning)
def check_estimators_data_not_an_array(name, estimator_orig, X, y, obj_type):
    pass


def check_estimator_cloneable(name, estimator_orig):
    """Checks whether the estimator can be cloned."""
    pass


def check_estimator_repr(name, estimator_orig):
    """Check that the estimator has a functioning repr."""
    pass


def check_parameters_default_constructible(name, estimator_orig):
    # test default-constructibility
    # get rid of deprecation warnings

    pass


def _enforce_estimator_tags_y(estimator, y):
    # Estimators with a `requires_positive_y` tag only accept strictly positive
    # data
    pass


def _enforce_estimator_tags_X(estimator, X, X_test=None, kernel=linear_kernel):
    # Estimators with `1darray` in `X_types` tag only accept
    # X of shape (`n_samples`,)
    pass


@ignore_warnings(category=FutureWarning)
def check_positive_only_tag_during_fit(name, estimator_orig):
    """Test that the estimator correctly sets the tags.input_tags.positive_only

    If the tag is False, the estimator should accept negative input regardless of the
    tags.input_tags.pairwise flag.
    """
    pass


@ignore_warnings(category=FutureWarning)
def check_non_transformer_estimators_n_iter(name, estimator_orig):
    # Test that estimators that are not transformers with a parameter
    # max_iter, return the attribute of n_iter_ at least 1.

    pass


@ignore_warnings(category=FutureWarning)
def check_transformer_n_iter(name, estimator_orig):
    # Test that transformers with a parameter max_iter, return the
    # attribute of n_iter_ at least 1.
    pass


@ignore_warnings(category=FutureWarning)
def check_get_params_invariance(name, estimator_orig):
    # Checks if get_params(deep=False) is a subset of get_params(deep=True)
    pass


@ignore_warnings(category=FutureWarning)
def check_set_params(name, estimator_orig):
    # Check that get_params() returns the same thing
    # before and after set_params() with some fuzz
    pass


@ignore_warnings(category=FutureWarning)
def check_classifiers_regression_target(name, estimator_orig):
    # Check if classifier throws an exception when fed regression targets

    pass


@ignore_warnings(category=FutureWarning)
def check_decision_proba_consistency(name, estimator_orig):
    # Check whether an estimator having both decision_function and
    # predict_proba methods has outputs with perfect rank correlation.

    pass


def check_outliers_fit_predict(name, estimator_orig):
    # Check fit_predict for outlier detectors.

    pass


def check_fit_non_negative(name, estimator_orig):
    # Check that proper warning is raised for non-negative X
    # when tag requires_positive_X is present
    pass


def check_fit_idempotent(name, estimator_orig):
    # Check that est.fit(X) is the same as est.fit(X).fit(X). Ideally we would
    # check that the estimated parameters during training (e.g. coefs_) are
    # the same, but having a universal comparison function for those
    # attributes is difficult and full of edge cases. So instead we check that
    # predict(), predict_proba(), decision_function() and transform() return
    # the same results.

    pass


def check_fit_check_is_fitted(name, estimator_orig):
    # Make sure that estimator doesn't pass check_is_fitted before calling fit
    # and that passes check_is_fitted once it's fit.

    pass


def check_n_features_in(name, estimator_orig):
    # Make sure that n_features_in_ attribute doesn't exist until fit is
    # called, and that its value is correct.

    pass


def check_requires_y_none(name, estimator_orig):
    # Make sure that an estimator with requires_y=True fails gracefully when
    # given y=None

    pass


@ignore_warnings(category=FutureWarning)
def check_n_features_in_after_fitting(name, estimator_orig):
    # Make sure that n_features_in are checked after fitting
    pass


def check_valid_tag_types(name, estimator):
    """Check that estimator tags are valid."""
    pass


def check_estimator_tags_renamed(name, estimator_orig):
    pass


def check_dataframe_column_names_consistency(name, estimator_orig):
    pass


def check_transformer_get_feature_names_out(name, transformer_orig):
    pass


def check_transformer_get_feature_names_out_pandas(name, transformer_orig):
    pass


def check_param_validation(name, estimator_orig):
    # Check that an informative error is raised when the value of a constructor
    # parameter does not have an appropriate type or value.
    pass


def check_set_output_transform(name, transformer_orig):
    # Check transformer.set_output with the default configuration does not
    # change the transform output.
    pass


def _output_from_fit_transform(transformer, name, X, df, y):
    """Generate output to test `set_output` for different configuration:

    - calling either `fit.transform` or `fit_transform`;
    - passing either a dataframe or a numpy array to fit;
    - passing either a dataframe or a numpy array to transform.
    """
    pass


def _check_generated_dataframe(
    name,
    case,
    index,
    outputs_default,
    outputs_dataframe_lib,
    is_supported_dataframe,
    create_dataframe,
    assert_frame_equal,
):
    """Check if the generated DataFrame by the transformer is valid.

    The DataFrame implementation is specified through the parameters of this function.

    Parameters
    ----------
    name : str
        The name of the transformer.
    case : str
        A single case from the cases generated by `_output_from_fit_transform`.
    index : index or None
        The index of the DataFrame. `None` if the library does not implement a DataFrame
        with an index.
    outputs_default : tuple
        A tuple containing the output data and feature names for the default output.
    outputs_dataframe_lib : tuple
        A tuple containing the output data and feature names for the pandas case.
    is_supported_dataframe : callable
        A callable that takes a DataFrame instance as input and return whether or
        E.g. `lambda X: isintance(X, pd.DataFrame)`.
    create_dataframe : callable
        A callable taking as parameters `data`, `columns`, and `index` and returns
        a callable. Be aware that `index` can be ignored. For example, polars dataframes
        would ignore the idnex.
    assert_frame_equal : callable
        A callable taking 2 dataframes to compare if they are equal.
    """
    pass


def _check_set_output_transform_dataframe(
    name,
    transformer_orig,
    *,
    dataframe_lib,
    is_supported_dataframe,
    create_dataframe,
    assert_frame_equal,
    context,
):
    """Check that a transformer can output a DataFrame when requested.

    The DataFrame implementation is specified through the parameters of this function.

    Parameters
    ----------
    name : str
        The name of the transformer.
    transformer_orig : estimator
        The original transformer instance.
    dataframe_lib : str
        The name of the library implementing the DataFrame.
    is_supported_dataframe : callable
        A callable that takes a DataFrame instance as input and returns whether or
        not it is supported by the dataframe library.
        E.g. `lambda X: isintance(X, pd.DataFrame)`.
    create_dataframe : callable
        A callable taking as parameters `data`, `columns`, and `index` and returns
        a callable. Be aware that `index` can be ignored. For example, polars dataframes
        will ignore the index.
    assert_frame_equal : callable
        A callable taking 2 dataframes to compare if they are equal.
    context : {"local", "global"}
        Whether to use a local context by setting `set_output(...)` on the transformer
        or a global context by using the `with config_context(...)`
    """
    pass


def _check_set_output_transform_pandas_context(name, transformer_orig, context):
    pass


def check_set_output_transform_pandas(name, transformer_orig):
    pass


def check_global_output_transform_pandas(name, transformer_orig):
    pass


def _check_set_output_transform_polars_context(name, transformer_orig, context):
    pass


def check_set_output_transform_polars(name, transformer_orig):
    pass


def check_global_set_output_transform_polars(name, transformer_orig):
    pass


@ignore_warnings(category=FutureWarning)
def check_inplace_ensure_writeable(name, estimator_orig):
    """Check that estimators able to do inplace operations can work on read-only
    input data even if a copy is not explicitly requested by the user.

    Make sure that a copy is made and consequently that the input array and its
    writeability are not modified by the estimator.
    """
    pass


def check_do_not_raise_errors_in_init_or_set_params(name, estimator_orig):
    """Check that init or set_param does not raise errors."""
    pass


def check_classifier_not_supporting_multiclass(name, estimator_orig):
    """Check that if the classifier has tags.classifier_tags.multi_class=False,
    then it should raise a ValueError when calling fit with a multiclass dataset.

    This test is not yielded if the tag is not False.
    """
    pass

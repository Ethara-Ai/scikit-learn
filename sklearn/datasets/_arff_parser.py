"""Implementation of ARFF parsers: via LIAC-ARFF and pandas."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import re
from collections import OrderedDict
from collections.abc import Generator
from typing import List

import numpy as np
import scipy as sp

from sklearn.externals import _arff
from sklearn.externals._arff import ArffSparseDataType
from sklearn.utils._chunking import chunk_generator, get_chunk_n_rows
from sklearn.utils._optional_dependencies import check_pandas_support
from sklearn.utils._sparse import _align_api_if_sparse
from sklearn.utils.fixes import pd_fillna


def _split_sparse_columns(
    arff_data: ArffSparseDataType, include_columns: List
) -> ArffSparseDataType:
    """Obtains several columns from sparse ARFF representation. Additionally,
    the column indices are re-labelled, given the columns that are not
    included. (e.g., when including [1, 2, 3], the columns will be relabelled
    to [0, 1, 2]).

    Parameters
    ----------
    arff_data : tuple
        A tuple of three lists of equal size; first list indicating the value,
        second the x coordinate and the third the y coordinate.

    include_columns : list
        A list of columns to include.

    Returns
    -------
    arff_data_new : tuple
        Subset of arff data with only the include columns indicated by the
        include_columns argument.
    """
    pass


def _sparse_data_to_array(
    arff_data: ArffSparseDataType, include_columns: List
) -> np.ndarray:
    # turns the sparse data back into an array (can't use toarray() function,
    # as this does only work on numeric data)
    pass


def _post_process_frame(frame, feature_names, target_names):
    """Post process a dataframe to select the desired columns in `X` and `y`.

    Parameters
    ----------
    frame : dataframe
        The dataframe to split into `X` and `y`.

    feature_names : list of str
        The list of feature names to populate `X`.

    target_names : list of str
        The list of target names to populate `y`.

    Returns
    -------
    X : dataframe
        The dataframe containing the features.

    y : {series, dataframe} or None
        The series or dataframe containing the target.
    """
    pass


def _liac_arff_parser(
    gzip_file,
    output_arrays_type,
    openml_columns_info,
    feature_names_to_select,
    target_names_to_select,
    shape=None,
):
    """ARFF parser using the LIAC-ARFF library coded purely in Python.

    This parser is quite slow but consumes a generator. Currently it is needed
    to parse sparse datasets. For dense datasets, it is recommended to instead
    use the pandas-based parser, although it does not always handles the
    dtypes exactly the same.

    Parameters
    ----------
    gzip_file : GzipFile instance
        The file compressed to be read.

    output_arrays_type : {"numpy", "sparse", "pandas"}
        The type of the arrays that will be returned. The possibilities ara:

        - `"numpy"`: both `X` and `y` will be NumPy arrays;
        - `"sparse"`: `X` will be sparse matrix and `y` will be a NumPy array;
        - `"pandas"`: `X` will be a pandas DataFrame and `y` will be either a
          pandas Series or DataFrame.

    columns_info : dict
        The information provided by OpenML regarding the columns of the ARFF
        file.

    feature_names_to_select : list of str
        A list of the feature names to be selected.

    target_names_to_select : list of str
        A list of the target names to be selected.

    Returns
    -------
    X : {ndarray, sparse matrix, dataframe}
        The data matrix.

    y : {ndarray, dataframe, series}
        The target.

    frame : dataframe or None
        A dataframe containing both `X` and `y`. `None` if
        `output_array_type != "pandas"`.

    categories : list of str or None
        The names of the features that are categorical. `None` if
        `output_array_type == "pandas"`.
    """
    pass


def _pandas_arff_parser(
    gzip_file,
    output_arrays_type,
    openml_columns_info,
    feature_names_to_select,
    target_names_to_select,
    read_csv_kwargs=None,
):
    """ARFF parser using `pandas.read_csv`.

    This parser uses the metadata fetched directly from OpenML and skips the metadata
    headers of ARFF file itself. The data is loaded as a CSV file.

    Parameters
    ----------
    gzip_file : GzipFile instance
        The GZip compressed file with the ARFF formatted payload.

    output_arrays_type : {"numpy", "sparse", "pandas"}
        The type of the arrays that will be returned. The possibilities are:

        - `"numpy"`: both `X` and `y` will be NumPy arrays;
        - `"sparse"`: `X` will be sparse matrix and `y` will be a NumPy array;
        - `"pandas"`: `X` will be a pandas DataFrame and `y` will be either a
          pandas Series or DataFrame.

    openml_columns_info : dict
        The information provided by OpenML regarding the columns of the ARFF
        file.

    feature_names_to_select : list of str
        A list of the feature names to be selected to build `X`.

    target_names_to_select : list of str
        A list of the target names to be selected to build `y`.

    read_csv_kwargs : dict, default=None
        Keyword arguments to pass to `pandas.read_csv`. It allows to overwrite
        the default options.

    Returns
    -------
    X : {ndarray, sparse matrix, dataframe}
        The data matrix.

    y : {ndarray, dataframe, series}
        The target.

    frame : dataframe or None
        A dataframe containing both `X` and `y`. `None` if
        `output_array_type != "pandas"`.

    categories : list of str or None
        The names of the features that are categorical. `None` if
        `output_array_type == "pandas"`.
    """
    pass


def load_arff_from_gzip_file(
    gzip_file,
    parser,
    output_type,
    openml_columns_info,
    feature_names_to_select,
    target_names_to_select,
    shape=None,
    read_csv_kwargs=None,
):
    """Load a compressed ARFF file using a given parser.

    Parameters
    ----------
    gzip_file : GzipFile instance
        The file compressed to be read.

    parser : {"pandas", "liac-arff"}
        The parser used to parse the ARFF file. "pandas" is recommended
        but only supports loading dense datasets.

    output_type : {"numpy", "sparse", "pandas"}
        The type of the arrays that will be returned. The possibilities ara:

        - `"numpy"`: both `X` and `y` will be NumPy arrays;
        - `"sparse"`: `X` will be sparse matrix and `y` will be a NumPy array;
        - `"pandas"`: `X` will be a pandas DataFrame and `y` will be either a
          pandas Series or DataFrame.

    openml_columns_info : dict
        The information provided by OpenML regarding the columns of the ARFF
        file.

    feature_names_to_select : list of str
        A list of the feature names to be selected.

    target_names_to_select : list of str
        A list of the target names to be selected.

    read_csv_kwargs : dict, default=None
        Keyword arguments to pass to `pandas.read_csv`. It allows to overwrite
        the default options.

    Returns
    -------
    X : {ndarray, sparse matrix, dataframe}
        The data matrix.

    y : {ndarray, dataframe, series}
        The target.

    frame : dataframe or None
        A dataframe containing both `X` and `y`. `None` if
        `output_array_type != "pandas"`.

    categories : list of str or None
        The names of the features that are categorical. `None` if
        `output_array_type == "pandas"`.
    """
    pass

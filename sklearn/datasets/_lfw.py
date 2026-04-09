"""Labeled Faces in the Wild (LFW) dataset

This dataset is a collection of JPEG pictures of famous people collected
over the internet, all details are available on the official website:

    http://vis-www.cs.umass.edu/lfw/
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import logging
from numbers import Integral, Real
from os import PathLike, listdir, makedirs, remove
from os.path import exists, isdir, join

import numpy as np
from joblib import Memory

from sklearn.datasets._base import (
    RemoteFileMetadata,
    _fetch_remote,
    get_data_home,
    load_descr,
)
from sklearn.utils import Bunch
from sklearn.utils._param_validation import (
    Hidden,
    Interval,
    StrOptions,
    validate_params,
)
from sklearn.utils.fixes import tarfile_extractall

logger = logging.getLogger(__name__)

# The original data can be found in:
# http://vis-www.cs.umass.edu/lfw/lfw.tgz
ARCHIVE = RemoteFileMetadata(
    filename="lfw.tgz",
    url="https://ndownloader.figshare.com/files/5976018",
    checksum="055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0",
)

# The original funneled data can be found in:
# http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
FUNNELED_ARCHIVE = RemoteFileMetadata(
    filename="lfw-funneled.tgz",
    url="https://ndownloader.figshare.com/files/5976015",
    checksum="b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a",
)

# The original target data can be found in:
# http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt',
# http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt',
# http://vis-www.cs.umass.edu/lfw/pairs.txt',
TARGETS = (
    RemoteFileMetadata(
        filename="pairsDevTrain.txt",
        url="https://ndownloader.figshare.com/files/5976012",
        checksum="1d454dada7dfeca0e7eab6f65dc4e97a6312d44cf142207be28d688be92aabfa",
    ),
    RemoteFileMetadata(
        filename="pairsDevTest.txt",
        url="https://ndownloader.figshare.com/files/5976009",
        checksum="7cb06600ea8b2814ac26e946201cdb304296262aad67d046a16a7ec85d0ff87c",
    ),
    RemoteFileMetadata(
        filename="pairs.txt",
        url="https://ndownloader.figshare.com/files/5976006",
        checksum="ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592",
    ),
)


#
# Common private utilities for data fetching from the original LFW website
# local disk caching, and image decoding.
#


def _check_fetch_lfw(
    data_home=None, funneled=True, download_if_missing=True, n_retries=3, delay=1.0
):
    """Helper function to download any missing LFW data"""

    data_home = get_data_home(data_home=data_home)
    lfw_home = join(data_home, "lfw_home")

    if not exists(lfw_home):
        makedirs(lfw_home)

    for target in TARGETS:
        target_filepath = join(lfw_home, target.filename)
        if not exists(target_filepath):
            if download_if_missing:
                logger.info("Downloading LFW metadata: %s", target.url)
                _fetch_remote(
                    target, dirname=lfw_home, n_retries=n_retries, delay=delay
                )
            else:
                raise OSError("%s is missing" % target_filepath)

    if funneled:
        data_folder_path = join(lfw_home, "lfw_funneled")
        archive = FUNNELED_ARCHIVE
    else:
        data_folder_path = join(lfw_home, "lfw")
        archive = ARCHIVE

    if not exists(data_folder_path):
        archive_path = join(lfw_home, archive.filename)
        if not exists(archive_path):
            if download_if_missing:
                logger.info("Downloading LFW data (~200MB): %s", archive.url)
                _fetch_remote(
                    archive, dirname=lfw_home, n_retries=n_retries, delay=delay
                )
            else:
                raise OSError("%s is missing" % archive_path)

        import tarfile

        logger.debug("Decompressing the data archive to %s", data_folder_path)
        with tarfile.open(archive_path, "r:gz") as fp:
            tarfile_extractall(fp, path=lfw_home)

        remove(archive_path)

    return lfw_home, data_folder_path


def _load_imgs(file_paths, slice_, color, resize):
    """Internally used to load images"""
    pass


#
# Task #1:  Face Identification on picture with names
#


def _fetch_lfw_people(
    data_folder_path, slice_=None, color=False, resize=None, min_faces_per_person=0
):
    """Perform the actual data loading for the lfw people dataset

    This operation is meant to be cached by a joblib wrapper.
    """
    pass


@validate_params(
    {
        "data_home": [str, PathLike, None],
        "funneled": ["boolean"],
        "resize": [Interval(Real, 0, None, closed="neither"), None],
        "min_faces_per_person": [Interval(Integral, 0, None, closed="left"), None],
        "color": ["boolean"],
        "slice_": [tuple, Hidden(None)],
        "download_if_missing": ["boolean"],
        "return_X_y": ["boolean"],
        "n_retries": [Interval(Integral, 1, None, closed="left")],
        "delay": [Interval(Real, 0.0, None, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
def fetch_lfw_people(
    *,
    data_home=None,
    funneled=True,
    resize=0.5,
    min_faces_per_person=0,
    color=False,
    slice_=(slice(70, 195), slice(78, 172)),
    download_if_missing=True,
    return_X_y=False,
    n_retries=3,
    delay=1.0,
):
    """Load the Labeled Faces in the Wild (LFW) people dataset \
(classification).

    Download it if necessary.

    =================   =======================
    Classes                                5749
    Samples total                         13233
    Dimensionality                         5828
    Features            real, between 0 and 255
    =================   =======================

    For a usage example of this dataset, see
    :ref:`sphx_glr_auto_examples_applications_plot_face_recognition.py`.

    Read more in the :ref:`User Guide <labeled_faces_in_the_wild_dataset>`.

    Parameters
    ----------
    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    funneled : bool, default=True
        Download and use the funneled variant of the dataset.

    resize : float or None, default=0.5
        Ratio used to resize the each face picture. If `None`, no resizing is
        performed.

    min_faces_per_person : int, default=None
        The extracted dataset will only retain pictures of people that have at
        least `min_faces_per_person` different pictures.

    color : bool, default=False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than the shape with color = False.

    slice_ : tuple of slice, default=(slice(70, 195), slice(78, 172))
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : bool, default=False
        If True, returns ``(dataset.data, dataset.target)`` instead of a Bunch
        object. See below for more information about the `dataset.data` and
        `dataset.target` object.

        .. versionadded:: 0.20

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.

        .. versionadded:: 1.5

    delay : float, default=1.0
        Number of seconds between retries.

        .. versionadded:: 1.5

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : numpy array of shape (13233, 2914)
            Each row corresponds to a ravelled face image
            of original size 62 x 47 pixels.
            Changing the ``slice_`` or resize parameters will change the
            shape of the output.
        images : numpy array of shape (13233, 62, 47)
            Each row is a face image corresponding to one of the 5749 people in
            the dataset. Changing the ``slice_``
            or resize parameters will change the shape of the output.
        target : numpy array of shape (13233,)
            Labels associated to each face image.
            Those labels range from 0-5748 and correspond to the person IDs.
        target_names : numpy array of shape (5749,)
            Names of all persons in the dataset.
            Position in array corresponds to the person ID in the target array.
        DESCR : str
            Description of the Labeled Faces in the Wild (LFW) dataset.

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarray. The first containing a 2D array of
        shape (n_samples, n_features) with each row representing one
        sample and each column representing the features. The second
        ndarray of shape (n_samples,) containing the target samples.

        .. versionadded:: 0.20

    Examples
    --------
    >>> from sklearn.datasets import fetch_lfw_people
    >>> lfw_people = fetch_lfw_people()
    >>> lfw_people.data.shape
    (13233, 2914)
    >>> lfw_people.target.shape
    (13233,)
    >>> for name in lfw_people.target_names[:5]:
    ...    print(name)
    AJ Cook
    AJ Lamas
    Aaron Eckhart
    Aaron Guiel
    Aaron Patterson
    """
    lfw_home, data_folder_path = _check_fetch_lfw(
        data_home=data_home,
        funneled=funneled,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )
    logger.debug("Loading LFW people faces from %s", lfw_home)

    # wrap the loader in a memoizing function that will return memmaped data
    # arrays for optimal memory usage
    m = Memory(location=lfw_home, compress=6, verbose=0)
    load_func = m.cache(_fetch_lfw_people)

    # load and memoize the pairs as np arrays
    faces, target, target_names = load_func(
        data_folder_path,
        resize=resize,
        min_faces_per_person=min_faces_per_person,
        color=color,
        slice_=slice_,
    )

    X = faces.reshape(len(faces), -1)

    fdescr = load_descr("lfw.rst")

    if return_X_y:
        return X, target

    # pack the results as a Bunch instance
    return Bunch(
        data=X, images=faces, target=target, target_names=target_names, DESCR=fdescr
    )


#
# Task #2:  Face Verification on pairs of face pictures
#


def _fetch_lfw_pairs(
    index_file_path, data_folder_path, slice_=None, color=False, resize=None
):
    """Perform the actual data loading for the LFW pairs dataset

    This operation is meant to be cached by a joblib wrapper.
    """
    pass


@validate_params(
    {
        "subset": [StrOptions({"train", "test", "10_folds"})],
        "data_home": [str, PathLike, None],
        "funneled": ["boolean"],
        "resize": [Interval(Real, 0, None, closed="neither"), None],
        "color": ["boolean"],
        "slice_": [tuple, Hidden(None)],
        "download_if_missing": ["boolean"],
        "n_retries": [Interval(Integral, 1, None, closed="left")],
        "delay": [Interval(Real, 0.0, None, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
def fetch_lfw_pairs(
    *,
    subset="train",
    data_home=None,
    funneled=True,
    resize=0.5,
    color=False,
    slice_=(slice(70, 195), slice(78, 172)),
    download_if_missing=True,
    n_retries=3,
    delay=1.0,
):
    """Load the Labeled Faces in the Wild (LFW) pairs dataset (classification).

    Download it if necessary.

    =================   =======================
    Classes                                   2
    Samples total                         13233
    Dimensionality                         5828
    Features            real, between 0 and 255
    =================   =======================

    In the `original paper <https://people.cs.umass.edu/~elm/papers/lfw.pdf>`_
    the "pairs" version corresponds to the "restricted task", where
    the experimenter should not use the name of a person to infer
    the equivalence or non-equivalence of two face images that
    are not explicitly given in the training set.

    The original images are 250 x 250 pixels, but the default slice and resize
    arguments reduce them to 62 x 47.

    Read more in the :ref:`User Guide <labeled_faces_in_the_wild_dataset>`.

    Parameters
    ----------
    subset : {'train', 'test', '10_folds'}, default='train'
        Select the dataset to load: 'train' for the development training
        set, 'test' for the development test set, and '10_folds' for the
        official evaluation set that is meant to be used with a 10-folds
        cross validation.

    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By
        default all scikit-learn data is stored in '~/scikit_learn_data'
        subfolders.

    funneled : bool, default=True
        Download and use the funneled variant of the dataset.

    resize : float, default=0.5
        Ratio used to resize the each face picture.

    color : bool, default=False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than the shape with color = False.

    slice_ : tuple of slice, default=(slice(70, 195), slice(78, 172))
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.

        .. versionadded:: 1.5

    delay : float, default=1.0
        Number of seconds between retries.

        .. versionadded:: 1.5

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray of shape (2200, 5828). Shape depends on ``subset``.
            Each row corresponds to 2 ravel'd face images
            of original size 62 x 47 pixels.
            Changing the ``slice_``, ``resize`` or ``subset`` parameters
            will change the shape of the output.
        pairs : ndarray of shape (2200, 2, 62, 47). Shape depends on ``subset``
            Each row has 2 face images corresponding
            to same or different person from the dataset
            containing 5749 people. Changing the ``slice_``,
            ``resize`` or ``subset`` parameters will change the shape of the
            output.
        target : numpy array of shape (2200,). Shape depends on ``subset``.
            Labels associated to each pair of images.
            The two label values being different persons or the same person.
        target_names : numpy array of shape (2,)
            Explains the target values of the target array.
            0 corresponds to "Different person", 1 corresponds to "same person".
        DESCR : str
            Description of the Labeled Faces in the Wild (LFW) dataset.

    Examples
    --------
    >>> from sklearn.datasets import fetch_lfw_pairs
    >>> lfw_pairs_train = fetch_lfw_pairs(subset='train')
    >>> list(lfw_pairs_train.target_names)
    [np.str_('Different persons'), np.str_('Same person')]
    >>> lfw_pairs_train.pairs.shape
    (2200, 2, 62, 47)
    >>> lfw_pairs_train.data.shape
    (2200, 5828)
    >>> lfw_pairs_train.target.shape
    (2200,)
    """
    pass

"""
EDBSCAN: Enforced Density-Based Spatial Clustering of Applications with Noise.

Author: Ruben Broekx <broekxruben@gmail.com>
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import _check_sample_weight

from edbscan._inner import inner

NoneType = type(None)


def edbscan(
    X: NDArray[NDArray[float]],
    y: Optional[NDArray[Optional[int]]] = None,
    eps: float = 0.5,
    *,
    min_samples: int = 5,
    metric: str = "euclidean",
    metric_params: Optional[Dict[str, Any]] = None,
    algorithm: str = "auto",
    leaf_size: int = 30,
    p: int = 2,
    sample_weight: Optional[NDArray[float]] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[NDArray[int], NDArray[int]]:
    """
    Perform EDBSCAN clustering from vector array or distance matrix.

    Parameters
    ----------
    X : {array-like, sparse (CSR) matrix} of shape (n_samples, n_features) or \
            (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.

    y : {array-like} of shape (n_samples, )
        Contains known clusters, which is optional.
        Specified -1 clusters indicate noise.

    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important EDBSCAN parameter to choose appropriately for your data set
        and distance function.

    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", x is assumed to be a distance matrix and
        must be square during fit.
        x may be a :term:`sparse graph <sparse graph>`,
        in which case only "nonzero" elements may be considered neighbors.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

        .. versionadded:: 0.19

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

    leaf_size : int, default=30
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

    p : float, default=2
        The power of the Minkowski metric to be used to calculate distance
        between points. If None, then ``p=2`` (equivalent to the Euclidean
        distance).

    sample_weight : array-like of shape (n_samples,), default=None
        Weight of each sample, such that a sample with a weight of at least
        ``min_samples`` is by itself a core sample; a sample with negative
        weight may inhibit its eps-neighbor from being core.
        Note that weights are absolute, and default to 1.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search. ``None`` means
        1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors. See :term:`Glossary <n_jobs>` for more details.
        If precomputed distance are used, parallel execution is not available
        and thus n_jobs will have no effect.

    Returns
    -------
    core_samples : ndarray of shape (n_core_samples,)
        Indices of core samples.

    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.  Noisy samples are given the label -1.

    See Also
    --------
    DBSCAN : An estimator interface for this clustering algorithm.
    OPTICS : A similar estimator interface clustering at multiple values of
        eps. Our implementation is optimized for memory usage.

    Notes
    -----
    This function is forked from sklearn.cluster.dbscan.

    This implementation bulk-computes all neighborhood queries, which increases
    the memory complexity to O(n.d) where d is the average number of neighbors,
    while original DBSCAN had memory complexity O(n). It may attract a higher
    memory complexity when querying these nearest neighborhoods, depending
    on the ``algorithm``.

    One way to avoid the query complexity is to pre-compute sparse
    neighborhoods in chunks using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
    ``mode='distance'``, then using ``metric='precomputed'`` here.

    Another way to reduce memory and computation time is to remove
    (near-)duplicate points and use ``sample_weight`` instead.

    :func:`cluster.optics <sklearn.cluster.optics>` provides a similar
    clustering with lower memory usage.
    """
    est = EDBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        metric_params=metric_params,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        n_jobs=n_jobs,
    )
    est.fit(X=X, y=y, sample_weight=sample_weight)
    return est.get_core_sample_indices(), est.get_labels()


class EDBSCAN(ClusterMixin, BaseEstimator):
    """
    Perform EDBSCAN clustering from vector array or distance matrix.

    EDBSCAN - Enforced Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them, whilst
    ensuring that the constraints provided by the known (pre-clustered) data points
    are respected. Good for data which contains clusters of similar density.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.

    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    metric : str, or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a :term:`Glossary <sparse graph>`, in which
        case only "nonzero" elements may be considered neighbors for DBSCAN.

        .. versionadded:: 0.17
           metric *precomputed* to accept precomputed sparse matrix.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

        .. versionadded:: 0.19

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

    leaf_size : int, default=30
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

    p : float, default=None
        The power of the Minkowski metric to be used to calculate distance
        between points. If None, then ``p=2`` (equivalent to the Euclidean
        distance).

    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    components_ : ndarray of shape (n_core_samples, n_features)
        Copy of each core sample found by training.
        Access by calling get_components().

    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples.
        Access by calling get_core_sample_indices().

    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.
        Access by calling get_labels().

    See Also
    --------
    DBSCAN : An estimator interface for this clustering algorithm.
    OPTICS : A similar clustering at multiple values of eps. Our implementation
        is optimized for memory usage.

    Notes
    -----
    This class is forked from sklearn.cluster.DBSCAN.

    This implementation bulk-computes all neighborhood queries, which increases
    the memory complexity to O(n.d) where d is the average number of neighbors,
    while original DBSCAN had memory complexity O(n). It may attract a higher
    memory complexity when querying these nearest neighborhoods, depending
    on the ``algorithm``.

    One way to avoid the query complexity is to pre-compute sparse
    neighborhoods in chunks using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
    ``mode='distance'``, then using ``metric='precomputed'`` here.

    Another way to reduce memory and computation time is to remove
    (near-)duplicate points and use ``sample_weight`` instead.

    :class:`cluster.OPTICS` provides a similar clustering with lower memory
    usage.
    """

    _UNLABELED: int = -2

    def __init__(
        self,
        eps: float = 0.5,
        *,
        min_samples: int = 5,
        metric: str = "euclidean",
        metric_params: Optional[Dict[str, Any]] = None,
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: Optional[float] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        """Initialise and configure EDBSCAN."""
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

        # Hidden variables
        self.components_: Optional[NDArray[NDArray[float]]] = None
        self.core_sample_indices_: Optional[NDArray[int]] = None
        self.labels_: Optional[NDArray[int]] = None

    def fit(  # noqa: C901
        self,
        X: NDArray[NDArray[float]],
        y: Optional[NDArray[Optional[int]]] = None,
        sample_weight: Optional[NDArray[float]] = None,
    ) -> EDBSCAN:
        """Perform DBSCAN clustering from features, or distance matrix.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        y : {array-like} of shape (n_samples, )
            Contains known clusters, which is optional.
            Specified -1 clusters indicate noise.

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        # Validate if the given X-values are correct
        X = self._validate_data(X, accept_sparse="csr")

        # Validate if the given y-values are correct
        if y is not None:
            _evaluate_known_clusters(y)
        else:
            # Initialise as None-array
            y = np.full(X.shape[0], None)

        # Validate if the given eps value is correct
        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")

        # Validate if the given sample weights are correct, if provided
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Calculate neighborhood for all samples. This leaves the original
        # point in, which needs to be considered later (i.e. point i is in the
        # neighborhood of point i. While True, its useless information)
        if self.metric == "precomputed" and sparse.issparse(X):
            # set the diagonal to explicit values, as a point is its own
            # neighbor
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)
                X.setdiag(X.diagonal())  # type: ignore  # XXX: modifies X's internals in-place

        # Define a NN model on the given data
        neighbors_model = NearestNeighbors(
            radius=self.eps,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            metric_params=self.metric_params,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        neighbors_model.fit(X)

        # This has worst case O(n^2) memory complexity
        neighborhoods = neighbors_model.radius_neighbors(X, return_distance=False)

        def _is_conflict(neighborhood: NDArray[int]) -> bool:
            """
            Check if the neighborhood is in conflict.

            A neighborhood is in conflict when it contains at least two different annotated classes (-1..N).

            Parameters
            ----------
            neighborhood : {array-like}
                Detected indices being in the same neighborhood.

            Returns
            -------
            conflict : bool
                Whether or not the given cluster is in conflict.
            """
            return len({c for c in y[neighborhood] if c is not None}) > 1  # type: ignore

        def _filter_conflict(label: int, neighborhood: NDArray[int]) -> NDArray[int]:
            """
            Filter out the conflicting indices from the given neighborhood.

            Parameters
            ----------
            label : int
                Target class that should be kept.
            neighborhood : {array-like}
                Detected indices being in the same neighborhood.

            Returns
            -------
            neighborhood : {array-like}
                Neighborhood containing only label and None values.
            """
            return np.asarray(
                [idx for idx in neighborhood if (y[idx] is None) or (y[idx] == label)],  # type: ignore
                dtype=np.intp,
            )

        # Make empty neighborhood if it leads to a conflict
        for i, neighborhood in enumerate(neighborhoods):
            if _is_conflict(neighborhood):
                neighborhoods[i] = _filter_conflict(label=y[i], neighborhood=neighborhood)

        # Find the number of neighbors close by (incl. itself).
        if sample_weight is None:
            n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])
        else:
            n_neighbors = np.array(
                [np.sum(sample_weight[neighbors]) for neighbors in neighborhoods]
            )

        # Initialise labels with the known clusters, the value convention is the following:
        #   0..N are clusters, either pre-defined or discovered
        #  -1 is noise, either pre-defined or discovered
        #  -2 is unlabeled, this notation differs from the original DBSCAN algorithm (where -1 is used)
        labels = deepcopy(y)
        labels[labels == None] = self._UNLABELED  # noqa: E711  (does not work with 'is')
        labels = np.asarray(labels, dtype=np.intp)

        # A list of all core samples found, which are those that stand a chance of being in a cluster.
        core_samples = np.asarray(n_neighbors >= self.min_samples, dtype=np.uint8)
        inner(core_samples, neighborhoods, labels)  # type: ignore

        # Remaining unlabeled labels are considered as noise (-1).
        labels[labels == self._UNLABELED] = -1

        self.core_sample_indices_ = np.where(core_samples)[0]
        self.labels_ = labels  # type: ignore

        if len(self.core_sample_indices_):  # type: ignore
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self

    def fit_predict(
        self,
        X: NDArray[NDArray[float]],
        y: Optional[NDArray[Optional[int]]] = None,
        sample_weight: Optional[NDArray[float]] = None,
    ) -> NDArray[int]:
        """Compute clusters from a data or distance matrix and predict labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        y : {array-like} of shape (n_samples, )
            Contains known clusters, which is optional.
            Specified -1 clusters indicate noise.

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        self.fit(X=X, y=y, sample_weight=sample_weight)
        labels = self.get_labels()
        assert labels is not None
        return labels

    def get_components(self) -> NDArray[NDArray[float]]:
        """Get the components."""
        assert self.components_ is not None
        return self.components_

    def get_core_sample_indices(self) -> NDArray[int]:
        """Get the core sample indices."""
        assert self.core_sample_indices_ is not None
        return self.core_sample_indices_

    def get_labels(self) -> NDArray[int]:
        """Get the predicted labels."""
        assert self.labels_ is not None
        return self.labels_


def _evaluate_known_clusters(known: NDArray[Optional[int]]) -> None:
    """Evaluate the known clusters and throw a suiting exception if necessary."""
    # Known has to be a 1-dimensional vector
    if len(known.shape) != 1:
        raise AttributeError(f"Invalid attribute shape {known.shape}! (has to be one-dimensional)")

    # All values that are not None have to be integers
    if not all(type(v) in (NoneType, int) for v in known):
        raise TypeError("Only NoneType or Integers allowed in known clusters!")

    # All integers are of value -1 or greater
    if not all(v is None or v >= -1 for v in known):
        raise ValueError("All integer values need to be of value -1 or greater!")

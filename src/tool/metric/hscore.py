
import sys

import numpy as np
from ..util import *

__all__ = ['h_score']


def h_score(root_dir):
    r"""
    H-score in `An Information-theoretic Approach to Transferability in Task Transfer Learning (ICIP 2019)
    <http://yangli-feasibility.com/home/media/icip-19.pdf>`_.

    The H-Score :math:`\mathcal{H}` can be described as:

    .. math::
        \mathcal{H}=\operatorname{tr}\left(\operatorname{cov}(f)^{-1} \operatorname{cov}\left(\mathbb{E}[f \mid y]\right)\right)

    where :math:`f` is the features extracted by the model to be ranked, :math:`y` is the groud-truth label vector

    The H-score measures the alignment between the feature covariance and the conditional expectation of the
    features given the labels. A higher H-score indicates better transferability.

    Args:
        root_dir (str): The directory from which to read the features and labels.

    Returns:
        score (float): The computed H-Score, a scalar value representing transferability.
    """
    f, y = read_data(root_dir)

    def covariance(X):
        X_mean = X - np.mean(X, axis=0, keepdims=True)
        cov = np.divide(np.dot(X_mean.T, X_mean), len(X) - 1)
        return cov

    covf = covariance(f)
    C = int(y.max() + 1)
    g = np.zeros_like(f)

    for i in range(C):
        Ef_i = np.mean(f[y == i, :], axis=0)
        g[y == i] = Ef_i

    covg = covariance(g)
    score = np.trace(np.dot(np.linalg.pinv(covf, rcond=1e-15), covg))
    return score


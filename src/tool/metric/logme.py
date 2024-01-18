
import numpy as np
from numba import njit
from ..util import *

__all__ = ['log_maximum_evidence']


def log_maximum_evidence(root_dir, regression=False, return_weights=False):
    r"""
    Log Maximum Evidence in `LogME: Practical Assessment of Pre-trained Models
    for Transfer Learning (ICML 2021) <https://arxiv.org/pdf/2102.11005.pdf>`_.

    This function reads features and targets from the provided directory, and computes the LogME score,
    which can be used to assess the compatibility between the pre-trained features and the target task.
    It can be applied in both classification and regression settings.

    Args:
        root_dir (str): The directory from which to read the features and targets.
        regression (bool, optional): Whether the task is regression. If False, classification is assumed. (Default: False)
        return_weights (bool, optional): If True, the function also returns Bayesian weights in addition to the LogME score. (Default: False)

    Returns:
        score (float): The LogME score, a scalar indicating the transferability.
        weights (np.ndarray, optional): The Bayesian weights matrix, returned only if `return_weights` is True.
    """
    features, targets = read_data(root_dir)
    f = features.astype(np.float64)
    y = targets
    if regression:
        y = targets.astype(np.float64)

    fh = f
    f = f.transpose()
    D, N = f.shape
    v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

    evidences = []
    weights = []
    if regression:
        C = y.shape[1]
        for i in range(C):
            y_ = y[:, i]
            evidence, weight = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            weights.append(weight)
    else:
        C = int(y.max() + 1)
        for i in range(C):
            y_ = (y == i).astype(np.float64)
            evidence, weight = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            weights.append(weight)

    score = np.mean(evidences)
    weights = np.vstack(weights)

    if return_weights:
        return score, weights
    else:
        return score


@njit
def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    Compute the maximum evidence for each class or regression target.

    This is a helper function called within log_maximum_evidence to calculate the evidence
    for a given set of targets y_, using a precomputed singular value decomposition.

    Args:
        y_ (np.ndarray): The target vector for a specific class or regression target.
        f (np.ndarray): The transposed feature matrix.
        fh (np.ndarray): The original feature matrix.
        v (np.ndarray): The left singular vectors from the SVD of f @ fh.
        s (np.ndarray): The singular values from the SVD of f @ fh.
        vh (np.ndarray): The right singular vectors from the SVD of f @ fh.
        N (int): The number of samples.
        D (int): The number of features.

    Returns:
        evidence (float): The evidence for the given target vector.
        m (np.ndarray): The Bayesian weight vector for the given target vector.
    """
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ y_))

    for _ in range(11):
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + lam)).sum()
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / alpha_de
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / beta_de
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam

    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * beta_de \
               - alpha / 2.0 * alpha_de \
               - N / 2.0 * np.log(2 * np.pi)

    return evidence / N, m


import numpy as np
from ..util import *
__all__ = ['log_expected_empirical_prediction']


def log_expected_empirical_prediction(root_dir):
    r"""
    Log Expected Empirical Prediction in `LEEP: A New Measure to
    Evaluate Transferability of Learned Representations (ICML 2020)
    <http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf>`_.

    The LEEP :math:`\mathcal{T}` can be described as:

    .. math::
        \mathcal{T}=\mathbb{E}\log \left(\sum_{z \in \mathcal{C}_s} \hat{P}\left(y \mid z\right) \theta\left(y \right)_{z}\right)

    where :math:`\theta\left(y\right)_{z}` is the predictions of pre-trained model on source category, :math:`\hat{P}\left(y \mid z\right)` is the empirical conditional distribution estimated by prediction and ground-truth label.

    The LEEP score quantifies the expected log-likelihood of the empirical conditional distribution of the
    target labels given the source labels, as predicted by the pre-trained model.

    Args:
        root_dir (str): The directory from which to read the predictions and labels.

    Returns:
        score (float): The computed LEEP score, a scalar value representing transferability.
    """
    predictions, labels = read_data(root_dir)
    N, C_s = predictions.shape
    labels = labels.reshape(-1)
    C_t = int(np.max(labels) + 1)

    normalized_prob = predictions / float(N)
    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for joint distribution over (y, z)

    for i in range(C_t):
        this_class = normalized_prob[labels == i]
        row = np.sum(this_class, axis=0)
        joint[i] = row

    p_target_given_source = (joint / joint.sum(axis=0, keepdims=True)).T  # P(y | z)
    empirical_prediction = predictions @ p_target_given_source
    empirical_prob = np.array([predict[label] for predict, label in zip(empirical_prediction, labels)])
    score = np.mean(np.log(empirical_prob))

    return score
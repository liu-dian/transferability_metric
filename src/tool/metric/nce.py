import numpy as np
from ..util import *

__all__ = ['negative_conditional_entropy']


def negative_conditional_entropy(src_root_dir, tar_root_dir):
    r"""
    Negative Conditional Entropy in `Transferability and Hardness of Supervised
    Classification Tasks (ICCV 2019) <https://arxiv.org/pdf/1908.08142v1.pdf>`_.

    The NCE :math:`\mathcal{H}` can be described as:

    .. math::
        \mathcal{H}=-\sum_{y \in \mathcal{C}_t} \sum_{z \in \mathcal{C}_s} \hat{P}(y, z) \log \frac{\hat{P}(y, z)}{\hat{P}(z)}

    where :math:`\hat{P}(z)` is the empirical distribution and :math:`\hat{P}\left(y \mid z\right)` is the empirical
    conditional distribution estimated by source and target label.

    This implementation calculates NCE using the empirical distributions of source
    and target labels obtained from the provided directories.

    Args:
        src_root_dir (str): Directory containing the source dataset with labels.
        tar_root_dir (str): Directory containing the target dataset with labels.

    Returns:
        A scalar value representing the negative conditional entropy.

    The function reads the source and target data using the `read_data` utility,
    which should return the features and labels for the datasets.
    """
    _, source_labels = read_data(src_root_dir)
    _, target_labels = read_data(tar_root_dir)
    C_t = int(np.max(target_labels) + 1)
    C_s = int(np.max(source_labels) + 1)
    N = len(source_labels)

    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for the joint distribution, shape [C_t, C_s]
    for s, t in zip(source_labels, target_labels):
        s = int(s)
        t = int(t)
        joint[t, s] += 1.0 / N
    p_z = joint.sum(axis=0, keepdims=True)

    p_target_given_source = (joint / p_z).T  # P(y | z), shape [C_s, C_t]
    mask = p_z.reshape(-1) != 0  # valid Z, shape [C_s]
    p_target_given_source = p_target_given_source[mask] + 1e-20  # remove NaN where p(z) = 0, add 1e-20 to avoid log (0)
    entropy_y_given_z = np.sum(- p_target_given_source * np.log(p_target_given_source), axis=1, keepdims=True)
    conditional_entropy = np.sum(entropy_y_given_z * p_z.reshape((-1, 1))[mask])

    return -conditional_entropy
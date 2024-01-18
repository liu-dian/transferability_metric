import numpy as np
import ot
import geomloss
import torch
import math
from ..util import *

def compute_coupling(X_src, X_tar, Y_src, Y_tar):
    """
    Compute the optimal transport plan (coupling) and the Wasserstein distance between
    source and target distributions using the Earth Mover's Distance (EMD).

    Args:
        X_src (torch.Tensor): Feature matrix for the source domain samples.
        X_tar (torch.Tensor): Feature matrix for the target domain samples.
        Y_src (np.ndarray): Labels for the source domain samples.
        Y_tar (np.ndarray): Labels for the target domain samples.

    Returns:
        P (np.ndarray): The optimal transport plan matrix.
        W (float): The Wasserstein distance given by the optimal transport plan.
    """
    # Define the cost function as squared Euclidean distance
    cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)

    # Compute the cost matrix
    C = cost_function(X_src, X_tar)

    # Compute the optimal transport plan using EMD
    P = ot.emd(ot.unif(X_src.shape[0]), ot.unif(X_tar.shape[0]), C.numpy(), numItermax=100000)
    # Calculate the Wasserstein distance
    W = np.sum(P * np.array(C.numpy()))

    return P, W

def compute_CE(P, Y_src, Y_tar):
    """
    Compute the Conditional Entropy (CE) of the target labels given the source labels
    under the optimal transport plan.

    Args:
        P (np.ndarray): The optimal transport plan matrix.
        Y_src (np.ndarray): Labels for the source domain samples.
        Y_tar (np.ndarray): Labels for the target domain samples.

    Returns:
        ce (float): The conditional entropy.
    """
    # Create sets of unique labels in source and target domains
    src_label_set = set(sorted(list(Y_src.flatten())))
    tar_label_set = set(sorted(list(Y_tar.flatten())))

    # Initialize the joint distribution matrix
    P_src_tar = np.zeros((np.max(Y_src) + 1, np.max(Y_tar) + 1))

    # Populate the joint distribution matrix
    for y1 in src_label_set:
        y1_idx = np.where(Y_src == y1)
        for y2 in tar_label_set:
            y2_idx = np.where(Y_tar == y2)
            RR = y1_idx[0].repeat(y2_idx[0].shape[0])
            CC = np.tile(y2_idx[0], y1_idx[0].shape[0])
            P_src_tar[y1, y2] = np.sum(P[RR, CC])

    # Compute the marginal distribution for the source labels
    P_src = np.sum(P_src_tar, axis=1)

    # Compute the conditional entropy
    ce = 0.0
    for y1 in src_label_set:
        P_y1 = P_src[y1]
        for y2 in tar_label_set:
            if P_src_tar[y1, y2] != 0:
                ce += -(P_src_tar[y1, y2] * math.log(P_src_tar[y1, y2] / P_y1))

    return ce

def optimal_transport(src_root_dir, tar_root_dir):
    """
    Calculate the Conditional Entropy (CE) using optimal transport between the source
    and target domain datasets.

    Args:
        src_root_dir (str): The directory path of the source domain data.
        tar_root_dir (str): The directory path of the target domain data.

    Returns:
        ce (float): The conditional entropy indicating the difficulty of transferring
                    labels from the source to the target domain.
    """
    # Read the source and target domain data
    src_x, src_y = read_data(src_root_dir)
    tar_x, tar_y = read_data(tar_root_dir)

    # Convert arrays to torch tensors
    src_x = torch.tensor(src_x, dtype=torch.float)
    tar_x = torch.tensor(tar_x, dtype=torch.float)

    # Compute the optimal transport plan and Wasserstein distance
    P, W = compute_coupling(src_x, tar_x, src_y, tar_y)

    # Compute the conditional entropy
    ce = compute_CE(P, src_y, tar_y)

    return ce

import numpy as np
from scipy import ndimage


def get_k_largest_connected_components(
    mask: np.ndarray,
    structure: np.ndarray,
    k: int = 1,
) -> np.ndarray:
    """
    Extract the k largest connected components from a binary mask.

    Args:
        mask (np.ndarray): Binary input mask.
        structure (np.ndarray): Structuring element for connectivity.
        k (int): Number of largest components to retain.

    Returns:
        np.ndarray: Binary mask containing only the k largest components.
    """
    if k <= 0:
        raise ValueError("`k` must be a positive integer.")

    labeled_mask, num_features = ndimage.label(mask, structure=structure)
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)

    labels, sizes = np.unique(labeled_mask[labeled_mask > 0], return_counts=True)
    largest_labels = labels[np.argsort(sizes)[-k:]]

    return np.isin(labeled_mask, largest_labels)

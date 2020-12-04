from typing import List

import numpy as np
from skimage import measure


def build_texture_vec(
    feat_vol: np.ndarray, thresholds: List[float], method: str = "prop"
) -> np.ndarray:
    """Compute the textural component of the RADISTAT metric.

    Parameters
    ----------
    feat_vol : array-like
        A 3D volume containing expression levels in the range 0-1.
        All values in this volume should be elements of the `thresholds` parameter.
        Empty or null values are assumed to be outside the ROI.

    thresholds : list[float]
        List of thresholds (upper bounds) used to classify values into expression levels.
        For example, for 3 bins, pass ``[1/3, 2/3, 1]``.

    method : {'prop', 'propratio', 'wghtprop'}
        Method used to compute texture metric. One of ['prop', 'propratio', 'wghtprop'].

    Returns
    -------
    array-like
        A vector containing the intensities of each expression level.
    """
    intensity_vec = np.zeros(len(thresholds))
    feat_roi = feat_vol[feat_vol > 0]

    # Proportion of each expression level
    if method == "prop":
        for i, t in enumerate(thresholds):
            intensity_vec[i] = np.count_nonzero(feat_vol == t) / np.size(feat_roi)

    # Proportion of clusters of each expression level
    elif method == "wghtprop":
        # Build up proportion vector
        proportion_vec = np.zeros(len(thresholds))
        for i, t in enumerate(thresholds):
            intensity_vec[i] = np.count_nonzero(feat_vol == t) / np.size(feat_roi)

        # Build up cluster proportion vector
        prop_clusters_vec = np.zeros(len(thresholds))
        for i, t in enumerate(thresholds):
            threshold_clusters = feat_vol == t
            num_clusters = np.max(measure.label(threshold_clusters))
            prop_clusters_vec[i] = num_clusters
        prop_clusters_vec /= sum(prop_clusters_vec)

        intensity_vec = proportion_vec * prop_clusters_vec

    # Ratio of expression levels to one another
    elif method == "propratio":
        proportion_vec = np.zeros(len(thresholds))
        for i, t in enumerate(thresholds):
            proportion_vec[i] = np.count_nonzero(feat_vol == t) / np.size(feat_roi)

        # TODO check that this works properly
        for i in range(len(proportion_vec)):
            for k in range(i + 1, len(proportion_vec)):
                intensity_vec[i] = proportion_vec[i] / proportion_vec[k]

        intensity_vec = np.clip(intensity_vec, 0, 100)

    else:
        raise ValueError(
            "Invalid texture metric, must be one of ['prop', 'wghtprop', 'propratio']"
        )

    return intensity_vec

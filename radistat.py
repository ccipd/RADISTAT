from dataclasses import dataclass
from sys import stderr

import numpy as np

from build_metrics import build_texture_vec, build_spatial_vec
from slic_supervoxels import slic


@dataclass
class RadistatResult:
    """Data class holding the result of a call to the RADISTAT function."""

    supervoxel_labels: np.ndarray
    cluster_values: np.ndarray
    expression_values: np.ndarray
    texture_vec: list
    spatial_vec: list


def radistat(
    img: np.ndarray,
    mask: np.ndarray,
    feat_map: np.ndarray,
    window_size: int = 5,
    num_min_voxels: int = 5,
    texture_metric: str = "prop",
    spatial_metric: str = "adjacency",
    threshold_pctiles: list = [33, 67],
) -> RadistatResult:
    """Compute the RADISTAT feature descriptor for a given image and feature.

    The current implementation only allows for 3 bins,
    thresholding at the 33rd and 67th percentiles.

    Parameters
    ----------

    img : array-like
        M x N x S array containing the image from which feature values were extracted

    mask : array-like
        M x N x S array corresponding to the label on `img` within with feature values were extracted.
        0 everywhere except the region of interest.

    feat_map : array-like
        M x N x S matrix containing representative feature values for each nonzero pixel in the mask ROI.

    window_size : int
        Distance between supervoxels.

    num_min_voxels : int
        Minimum number of voxels in a supervoxel cluster.

    texture_metric : {'prop', 'propratio', 'wghtprop'}
        Which texture metric to be computed. One of ['prop', 'propratio', 'wghtprop'].

    spatial_metric : {'adjacency', 'mst', 'delauney'}
        Which spatial_metric to be used.

    threshold_pctiles : list[str]
        List of percentiles, between 0 and 100 inclusive, at which to threshold each bin.
        For example, [33, 67] will produce 3 bins of expression values,
        separated at the 33rd and 67th percentiles.
    """

    # Validate inputs
    if img.shape != mask.shape or mask.shape != feat_map.shape:
        raise ValueError(
            f"img (shape {img.shape}), mask (shape {mask.shape}),\
                         and feat_map (shape {feat_map.shape}) must be the same shape."
        )
    if not np.any(mask):
        raise ValueError("Mask is empty!")

    feat_vals = feat_map[mask > 0]

    # Get superpixels
    print("Computing superpixel clusters...\n", file=stderr)
    # If image is 2D, tile the array so it's 3D
    vol_supervoxel: np.ndarray
    if len(img.shape) == 2:
        step = (window_size, window_size, 1)
        _, _, vol_supervoxel = slic(
            np.tile(feat_vals, (1, 2)).transpose(),
            np.tile(mask, (2, 1, 1)).transpose((1, 2, 0)),
            step,
            num_min_voxels,
        )
        vol_supervoxel = vol_supervoxel[:, :, 0]
    else:
        step = (window_size, window_size, window_size)
        _, _, vol_supervoxel = slic(feat_map, mask, step, num_min_voxels)

    labeled_voxels = vol_supervoxel[mask > 0]
    # Fill each cluster with mean of all voxels in that cluster
    cluster_vals = np.zeros(np.shape(feat_vals))
    all_labels = np.unique(labeled_voxels)
    for label in all_labels:
        cluster_vals[labeled_voxels == label] = np.mean(
            feat_vals[labeled_voxels == label]
        )

    # Bin clusters based on RADISTAT expression value thresholds
    print("Partitioning clusters into expression levels...\n", file=stderr)
    expression_vals = np.zeros(np.shape(cluster_vals))
    threshold_vals = [np.percentile(feat_vals, p) for p in threshold_pctiles]
    for idx, thresh in enumerate(threshold_vals):
        # Indices of unassigned clusters below current thresh
        cluster_idxs = [
            i
            for i in range(cluster_vals.shape[0])
            if cluster_vals[i] <= thresh and expression_vals[i] == 0
        ]
        expression_vals[cluster_idxs] = (idx + 1) / (len(threshold_vals) + 1)
    # At this point, the remaining zeros in expresion_vals should go in the last bin
    expression_vals[expression_vals == 0] = 1

    # Create feature volume out of expression levels
    feat_vol = np.zeros(np.shape(mask))
    feat_vol[mask == 1] = expression_vals

    thresholds = np.unique(expression_vals)

    # Calculate the textural and spatial metrics
    print("Calculating RADISTAT metrics...", file=stderr)
    texture_vec = build_texture_vec(feat_vol, thresholds, texture_metric)

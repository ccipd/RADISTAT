from slic_supervoxels import slic

import numpy as np


class RadistatResult:
    """Data class holding the result of a call to the RADISTAT function.
    """

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
    view: bool = False,
) -> RadistatResult:
    """
    Compute the RADISTAT feature descriptor for a given image and feature.

    The current implementation only allows for 3 bins,
    thresholding at the 33rd and 67th percentiles.

    Parameters
    ----------

    img :
    M x N x S array containing the image from which feature values were extracted

    mask :
    M x N x S array corresponding to the label on `img` within with feature values were extracted.
    0 everywhere except the region of interest.

    feat_map :
    M x N x S matrix containing representative feature values for each nonzero pixel in the mask ROI.

    opts :
    A `RadistatOptions` struct containing options. See the `RadistatOptions` class for available options.

    window_size:
    Distance between supervoxels.

    num_min_voxels:
    Minimum number of voxels in a supervoxel cluster.

    texture_metric:
    Which texture metric to be computed. One of ['prop', 'propratio', 'wghtprop'].

    spatial_metric:
    Which spatial_metric to be used. One of ['adjacency'].

    threshold_pctiles:
    List of percentiles, between 0 and 100 inclusive, at which to threshold each bin.
    For example, [33, 67] will produce 3 bins of expression values,
    separated at the 33rd and 67th percentiles.

    view:
    Whether or not to plot results.
    """

    ## Validate inputs
    if img.shape != mask.shape or mask.shape != feat_map.shape:
        raise ValueError(
            f"img (shape {img.shape}), mask (shape {mask.shape}),\
                         and feat_map (shape {feat_map.shape}) must be the same shape."
        )
    if not np.any(mask):
        raise ValueError("Mask is empty!")

    feat_vals = feat_map[mask > 0]

    ## Get superpixels
    # If image is 2D, tile the array so it's 3D
    vol_supervoxel: np.ndarray
    if len(img.shape) == 2:
        step = [window_size, window_size, 1]
        _, _, vol_supervoxel = slic(
            np.tile(feat_vals, (1, 2)).transpose(), np.tile(mask, (2, 1, 1)).transpose(1, 2, 0), step, num_min_voxels
        )
    else:
        step = [window_size, window_size, window_size]
        _, _, vol_supervoxel = slic(feat_map, mask, step, num_min_voxels)

    labeled_voxels = vol_supervoxel[mask > 0]
    # Fill each cluster with mean of all voxels in that cluster
    cluster_vals = np.zeros(np.shape(feat_vals))
    supervoxel_labels = np.unique(labeled_voxels)
    for label in supervoxel_labels:
        cluster_vals[labeled_voxels == label] = np.mean(
            feat_vals[labeled_voxels == label]
        )

    # Bin clusters based on RADISTAT expression value thresholds
    threshold_vals = map(lambda p: np.percentile(feat_vals, p), threshold_pctiles)

from itertools import combinations
from typing import List, Dict, Tuple

import numpy as np
from skimage import measure
from skimage.future import graph


def build_texture_vec(
        feat_vol: np.ndarray, thresholds: List[float], method: str = "prop"
) -> Dict[Tuple[int, int], int]:
    """Compute the textural component of the RADISTAT metric.

    Parameters
    ----------
    feat_vol : array-like
        A 3D volume containing expression levels in the range 0-1.
        All values in this volume should be elements of the `thresholds` parameter.
        Empty or null values are assumed to be outside the ROI.

    thresholds : list[float]
        List of thresholds (upper bounds) used to classify values into expression levels.
        For example, for 3 bins, pass ``[1, 2, 3]``.

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


def build_spatial_vec(
        feat_vol: np.ndarray, thresholds: List[int], graphstruct: str
) -> np.ndarray:
    """Compute the spatial component of the RADISTAT metric.

    Parameters
    ----------
    img : array-like
        The image of

    feat_vol : array-like
        A 2D image or 3D volume containing expression levels in the range 0-1.
        Should be the same shape as vol_supervoxel.
        All values in this volume should be elements of the `thresholds` parameter.
        Empty or null values are assumed to be outside the ROI.

    thresholds : list[float]
        List of thresholds (upper bounds) used to classify values into expression levels.
        For example, for 3 bins, pass ``[1, 2, 3]``.

    graphstruct : {'adjacency', 'mst', 'delauney'}
        Which method to use when calculating the spatial metric.

    Returns
    -------
    dict
        Proportion of adjacencies between clusters of different expression levels.
    """
    # Compute the region-adjacency graph of the labeled image.
    # The "colors" will be the feature bin numbers of each pixel.
    rag = graph.rag_mean_color(feat_vol, measure.label(feat_vol))
    rag.remove_node(0)  # Remove background node

    # Each node in the RAG represents a RADISTAT cluster (i.e. a group of supervoxels that were
    # binned together via thresholding). It has a "mean color" attribute equal to the cluster's
    # bin number.
    # We get the adjacency dictionary from this RAG structure (which is just a graph).
    # Adjacencies will be duplicated, but proportions will remain the same.

    # Create a dictionary for the spatial metric.
    # Keys are each possible pair of bin numbers
    spatial_dict = {k: 0 for k in combinations(thresholds, 2)}

    # Define a quick lambda to get the expression level of a node
    get_level = lambda n: int(rag.nodes[n]['mean color'][0])

    # Iterate through and count adjacencies between expression levels
    for node, adj in rag.adjacency():
        level = get_level(node)
        for adj_node in adj.keys():
            adj_level = get_level(adj_node)
            # Sort the levels
            key = tuple(sorted((level, adj_level)))
            spatial_dict[key] += 1

    # Calculate proportions of adjacencies
    num_adj = sum(spatial_dict.values())
    spatial_prop = {k: v / num_adj for (k, v) in spatial_dict.items()}

    return spatial_prop

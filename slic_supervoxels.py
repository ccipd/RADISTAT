import math
import sys
from typing import Tuple

import numpy as np
from numpy import ndarray
from scipy import stats
from skimage import measure

# --- PARAMETERS ---
# Threshold of displacement of centroids
# -- k-means will stop of mean displacement is below this value
EPSILON = 1e-3

LAMBDA = np.ones(4)  # Defines how to weight each vector

# Maximum number of iterations used in k-means clustering
MAX_ITER = 100


def slic(
        feats: ndarray, vol_mask: ndarray, seed_dist: Tuple, min_voxels: int
) -> Tuple[ndarray, ndarray, ndarray]:
    """Generate superpixel clusters based on feature vectors.

    Parameters
    ----------

    `feats` : An M x P matrix, where M is the number of voxels and P is the
    number of features

    `vol_mask` : An R x C feats D volume matrix filled with zeroes
    everywhere except the region of interest, which is filled with ones

    `seed_dist` : A list of size 3, indicating distances between the initial
    centroids of supervoxels.
    As an example, `(3,2,5)` indicates that the centroids should be 3 row,
    2 column, and 5 depth units away from each other.

    `min_voxels` : The minimum number of voxels in a supervoxel

    Returns
    -------

    A tuple, containing:

    * An M x 1 column vector, where the mth vector indicates the label of
      supervoxel m, a number from 1 to K.
    * A K x (P + |S|) matrix, where the kth row is the feature vector of
      the kth centroid (including spatial features).
    * An R x C x D volume matrix, where each element indicates the ID
      of the supervoxel that voxel belongs to.
        ( TODO: This can be reconstructed from the other two. )
    """

    # Extract row, col, and depth from the volume mask
    r, c, d = vol_mask.shape
    # TODO why are c and r switched in the MATLAB codebase?
    rg, cg, dg = np.meshgrid(range(r), range(c), range(d))
    # Filter against region of interest
    rg = rg[vol_mask[:] > 0]
    cg = cg[vol_mask[:] > 0]
    dg = dg[vol_mask[:] > 0]

    # Don't need to z-score spatial coords
    # s is 3-column matrix with (row, col, depth) points in each row
    s = np.array([rg, cg, dg]).transpose()
    # Two corners of "cube" surrounding mask
    smax = s.max(axis=0)
    smin = s.min(axis=0)

    # Z-score radiomic features
    m, _ = feats.shape
    feats = stats.zscore(feats, axis=0, ddof=1)

    # Used to determine which seed a pixel is assigned to
    # Each row is a voxel's radiomic and spatial features
    feats_total = np.concatenate((feats, s), axis=1)

    w = np.array([1, 1 / r, 1 / c, 1 / d]) * LAMBDA

    # Create initial centroids
    sys.stderr.write("Computing superpixel clusters...\n")
    r0, c0, d0 = np.meshgrid(
        np.arange(0, r, seed_dist[0]),
        np.arange(0, c, seed_dist[1]),
        np.arange(0, d, seed_dist[2]),
    )
    grid_shape = r0.shape
    # Initial list of pixels
    s0 = np.array(
        [
            [r0[x, y, z], c0[x, y, z], d0[x, y, z]]
            for x in range(grid_shape[0])
            for y in range(grid_shape[1])
            for z in range(grid_shape[2])
        ]
    )
    # Trim intitial pixels to masked volume
    for i in range(len(r0.shape)):
        s0 = s0[s0[:, i] >= smin[i]]
        s0 = s0[s0[:, i] <= smax[i]]

    if s0.size == 0:
        raise Exception("Bad seed step size. No centroids to initialize.")

    # Initialize clusters: Each centroid gets its own cluster initially
    # Stores values for each cluster
    centers = np.zeros((s0.shape[0], feats_total.shape[1]))
    centers_isolated = [False for _ in range(len(centers))]

    for k, center in enumerate(s0):
        # Find neighboring points for each centroid
        neighbor_idxs = nearest_neighbors(center, s, seed_dist)

        # If cluster has no neighbor, mark as isolated
        if len(neighbor_idxs) == 0:
            centers_isolated[k] = True
        else:
            # Assign mean feature value to each cluster
            nh_feats = feats_total[neighbor_idxs, :]
            centers[k, :] = np.mean(nh_feats, axis=0)
    # Remove all isolated clusters
    centers = np.array([c for i, c in enumerate(centers) if not centers_isolated[i]])

    # ---- ITERATIVE K-MEANS CLUSTERING ----
    ctr_displacement = np.infty
    for _ in range(MAX_ITER):
        # Stop if centroid displacement is small enough
        if ctr_displacement < EPSILON:
            break

        # Distance matrix from each voxel to each centroid
        dist_ik = np.full((len(feats), len(centers)), np.infty)

        # Compute feature distances from each neighbor to its centroid
        for k, center in enumerate(centers):
            neighbor_idxs = nearest_neighbors(center[-3:], s, seed_dist)
            nh_feats = feats_total[neighbor_idxs, :]
            # Calculate feature distance between center and neighbors
            delta = nh_feats - np.tile(center, [len(nh_feats), 1])
            delta = delta * np.tile(w, [len(delta), 1])

            # Fill distance matrix with squared-distances
            dist_ik[neighbor_idxs, k] = np.sum(delta ** 2, axis=1)

        # Assign center indices to each voxel
        # ctr_idxs[i] = X means that voxel i belongs to center X
        # with a minimum feature distance of dist_min[i]
        dist_min, ctr_idxs = np.min(dist_ik, axis=1), np.argmin(dist_ik, axis=1)

        # There may still be isolated voxels at this point, e.g. if centers
        # move away from other voxels.
        isolated_idxs = np.where(dist_min == np.infty)[0]
        if isolated_idxs.size > 0:
            for idx in isolated_idxs:
                # Calculate distance to closest centroid, using only spatial coords
                delta = np.tile(feats_total[idx, -3:], (centers.shape[0], 1)) - centers[:, -3:]
                new_center = np.argmin(np.sum(delta ** 2, axis=1), axis=0)
                ctr_idxs[idx] = new_center

        # Check for isolated/unused/duplicate center indices and clean up
        uniq_ctr_idx = np.unique(ctr_idxs)
        centers = centers[uniq_ctr_idx, :]
        dist_ik = dist_ik[:, uniq_ctr_idx]

        # Sanity check: No isolated centers should remain
        dist_min, ctr_idxs = np.min(dist_ik, axis=1), np.argmin(dist_ik, axis=1)
        uniq_ctr_idx = np.unique(ctr_idxs)
        assert np.setdiff1d(uniq_ctr_idx, range(len(centers))).size == 0

        centers_old = np.copy(centers)

        # Update candidate centers to mean feature values
        ctr_idxs, _ = rankindex_array(ctr_idxs)
        for k in range(len(centers)):
            centers[k, :] = np.mean(feats_total[ctr_idxs == k, :], axis=0)

        # Displace & calculate displacements
        # Total displacement is the mean Euclidean distance between old and new centers
        cluster_displacement = centers[:, -3:] - centers_old[:, -3:]
        ctr_displacement = np.mean(np.sqrt(np.sum(cluster_displacement ** 2, axis=1)))

    # At this point, there may be some voxels that are still isolated
    # Split non-contiguous regions into new clusters
    # skimage.measure.label should do this by relabeling contiguous regions
    vol_supx = np.zeros(vol_mask.shape)
    vol_supx[vol_mask > 0] = ctr_idxs
    labels, num_labels = measure.label(vol_supx, return_num=True)
    # labels is the input volume clustered (via value) into contiguous regions

    # Check for regions that are too small, i.e. under num_min_voxels
    small_regions = [
        i for i in range(num_labels) if (labels == i).sum() < min_voxels
    ]
    # Each element in small_regions is an index of a small region
    # We need to reassign these to neighboring clusters
    for region_index in small_regions:
        isol_voxels = np.argwhere(labels == region_index)

        # Get the union of all neighbors of the small regions
        isol_neighbors = np.array([])
        for voxel in isol_voxels:
            isol_neighbors = np.append(
                isol_neighbors, nearest_neighbors(voxel, s, (1, 1, 1)), axis=0
            )

        isol_neighbors = np.setdiff1d(
            isol_neighbors, np.nonzero(labels == region_index)
        )
        if not isol_neighbors:
            continue
        nh_candidates = labels[isol_neighbors]
        # Filter on neighborhoods that are big enough
        idx_valid_nh = [
            i
            for i, n in enumerate(nh_candidates)
            if (labels == n).sum() >= min_voxels - len(isol_voxels)
        ]
        new_nh = 0
        if idx_valid_nh:
            # Assign to the most frequently occurring neighbor
            new_nh = np.argmax(np.bincount(nh_candidates[idx_valid_nh]))
        else:
            new_nh = np.argmax(np.bincount(nh_candidates))

    supervoxel_ids, supervoxel_vals = rankindex_array(labels)
    vol_supervoxel = np.zeros(np.shape(vol_mask))
    vol_supervoxel[vol_mask > 0] = supervoxel_ids[vol_mask > 0]
    supervoxel_id_list = np.unique(supervoxel_ids[:])
    # Calculate centers in feature space
    vk = np.zeros((len(supervoxel_id_list), np.shape(feats_total)[1]))
    supervoxel_size = np.zeros(len(supervoxel_id_list))  # no. of voxels in supervoxel
    for i in range(len(supervoxel_id_list)):
        vk[i] = np.mean(feats_total[supervoxel_ids[vol_mask > 0] == i, :], axis=0)
        supervoxel_size[i] = (supervoxel_ids == i).sum()

    return supervoxel_ids, vk, vol_supervoxel


def nearest_neighbors(center: np.ndarray, points: np.ndarray, step: Tuple) -> list:
    """
    Compute the spatially nearest neighbors to a given center from candidate points.

    Parameters
    ----------
    center : The point for which to find neighbors

    points : Candidate points over which to search for neighbors to `center`

    step : Maximum distance between candidate neighbor and `center`, in the form of
    a tuple containing (row, column, depth) distances

    Returns
    -------
    A list of indices of `points` that are spatially within `step` of `center`.
    """

    # Check that dimensions match
    if np.shape(points)[1] != np.shape(center)[0]:
        sys.stderr.write(
            "nearest_neighbors: warning: Dimension mismatch. Returning empty list."
        )
        return []

    # Return all points within step size of center
    return [
        idx
        for idx, pt in enumerate(points)
        if all([abs(center[i] - pt[i]) <= step[i] for i in range(len(step))])
    ]


def rankindex_array(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce any-dimension array entries to their 0-indexed ascending ranking in the array.

    For example,
    [ [1.0, 0.5, -1.0],
      [0.5, 1.0, -1.0] ]
    becomes
    [ [2, 1, 0],
      [1, 2, 0] ]

    Additionally, a list is returned that maps ranking (index in list) to the actual value.
    """
    vals = np.unique(a)
    ranked = np.copy(a)
    for i, v in enumerate(vals):
        ranked[a == v] = i

    return ranked, vals

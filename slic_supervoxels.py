import math
import sys
from typing import Tuple

import numpy as np
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
    feats: np.ndarray, vol_mask: np.ndarray, seed_dist: Tuple, min_voxels: int
) -> Tuple[list, list, np.ndarray]:
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
    r, c, d = vol_mask.shape[:3]
    rg, cg, dg = np.meshgrid(
        range(1, r, seed_dist[0]), range(1, c, seed_dist[1]), range(1, d, seed_dist[2])
    )
    # Filter against region of interest
    rg = rg[vol_mask[:] > 0]
    cg = cg[vol_mask[:] > 0]
    dg = dg[vol_mask[:] > 0]

    # Don't need to z-score spatial coords
    # s is 3-column matrix with (row, col, depth) points in each row
    s = np.array([rg, cg, dg])
    smax = np.ndarray.max(s, axis=0)
    smin = np.ndarray.min(s, axis=0)

    # Z-score radiomic features
    m, p = feats.shape
    feats = stats.zscore(feats, axis=0, ddof=1)

    # Used to determine which seed a pixel is assigned to
    feats_total = np.concatenate((np.transpose(s), feats), axis=1)

    w = np.array([1, 1 / r, 1 / c, 1 / d]) * LAMBDA

    # Create initial centroids
    sys.stderr.write("Creating initial seeds...\n")
    c0, r0, d0 = np.meshgrid(
        np.arange(1, c, seed_dist[1]),
        np.arange(1, r, seed_dist[0]),
        np.arange(1, d, seed_dist[2]),
    )
    centroids = np.array([r0[:], c0[:], d0[:]])
    if centroids.size == 0:
        raise Exception("Bad seed step size. No centroids to initialize.")

    # Initialize clusters: Each centroid gets its own cluster initially
    # Stores values for each cluster
    clusters = np.zeros(len(centroids), np.shape(feats_total)[1])
    cluster_isolated = [False for _ in range(len(clusters))]

    for k, centroid in enumerate(centroids):
        # Find neighboring points for each centroid
        neighbor_idxs = nearest_neighbors(centroid, centroids, seed_dist)

        # If cluster has no neighbor, mark as isolated
        if len(neighbor_idxs) == 0:
            cluster_isolated[k] = True
        else:
            # Assign value to each cluster
            nh_feats = feats_total[neighbor_idxs, :]
            clusters[k, :] = np.mean(nh_feats, axis=0)
    # Remove all isolated clusters
    clusters = [c for i, c in enumerate(clusters) if not cluster_isolated[i]]

    # ---- ITERATIVE K-MEANS CLUSTERING ----
    ctr_displacement = np.infty
    for _ in range(MAX_ITER):
        # Stop if centroid displacement is small enough
        if ctr_displacement < EPSILON:
            break

        # Distance matrix
        dist_ik = np.full((len(feats), len(centroids)), np.infty)

        # Compute feature distances from each neighbor to its centroid
        for k, centroid in enumerate(centroids):
            neighbor_idxs = nearest_neighbors(centroid, centroids, seed_dist)
            nh_feats = feats_total[neighbor_idxs, :]
            # Calculate feature distance between center and neighbors
            delta = nh_feats - np.tile(centroid, [len(nh_feats), 1])
            delta = delta * np.tile(w, [len(delta), 1])

            # Fill distance matrix with squared values
            dist_ik[neighbor_idxs, k] = np.sum(delta ** 2, axis=1)

        # Assign center indices to each voxel
        # ctr_idxs[i] = X means that voxel i belongs to center X
        dist_min, ctr_idxs = np.min(dist_ik, axis=1), np.argmin(dist_ik, axis=1)

        # There may still be isolated voxels at this point, e.g. if centers
        # move away from other voxels.
        isolated_idxs = np.transpose(np.nonzero(dist_min[dist_min == np.infty]))
        if np.any(isolated_idxs):
            for idx in isolated_idxs:
                # Calculate distance to closest centroid, using only spatial coords
                delta = np.tile(feats_total[idx, -3:], len(clusters)) - clusters[:, -3:]
                new_center = np.min(np.sum(delta**2, axis=1), axis=0)
                ctr_idxs[idx] = new_center

        # Check for isolated/unused/duplicate center indices and clean up
        uniq_ctr_idx = np.unique(ctr_idxs)
        clusters = clusters[uniq_ctr_idx,:]
        dist_ik = dist_ik[:, uniq_ctr_idx]

        # Sanity check: No isolated centers should remain
        assert np.setdiff1d(uniq_ctr_idx, list(range(len(clusters)))) == []

        clusters_old = clusters

        # Update alive centers
        ctr_idxs = rankindex_array(ctr_idxs)
        for k in range(len(clusters)):
            clusters[k, :] = np.mean(feats_total[ctr_idxs == k, :], axis=0)

        # Displace & calculate displacements
        cluster_displacement = clusters[:, -3:-1] - clusters_old[:, -3:-1]
        ctr_displacement = np.mean(math.sqrt(np.sum(cluster_displacement ** 2)), axis=1)

    # At this point, there may be some voxels that are still isolated
    # Split non-contiguous regions into new clusters
    # skimage.measure.label should do this by relabeling contiguous regions
    vol_supx = np.zeros(vol_mask.shape)
    vol_supx[vol_mask > 0] = ctr_idxs
    labels = measure.label(vol_supx)

def nearest_neighbors(center: np.ndarray, points: np.ndarray, step: Tuple) -> list:
    """
    Compute the spatially nearest neighbors to a given center from candidate points.

    Parameters
    ----------
    center : The point for which to find neighbors

    points : Candidate points over which to search for neighbors to `center`

    step : Maximum distance between candidate neighbor and `center`, in the form of
    a tuple containing (row, column, depth) distances
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
        if all([center[i] - pt[i] <= step[i] for i in range(len(step))])
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

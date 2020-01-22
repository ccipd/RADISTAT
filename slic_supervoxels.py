import sys
from typing import Tuple

import numpy as np
from scipy import stats

# --- PARAMETERS ---
# Threshold of displacement of centroids -- k-means will stop of mean displacement is below this value
EPSILON = 1e-3

LAMBDA = np.ones(4)

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

    `seed_dist` : A list of size 3, indicating distances between the initial centroids
    of supervoxels.
    As an example, `(3,2,5)` indicates that the centroids should be 3 row, 2 column,
    and 5 depth units away from each other.

    `min_voxels` : The minimum number of voxels in a supervoxel

    Returns
    -------

    A tuple, containing:

    * An M x 1 column vector, where the mth vector indicates the label of
      supervoxel m, a number from 1 to K.
    * A K x (P + |S|) matrix, where the kth row is the feature vector of
      the kth centroid.
    * An R x C x D volume matrix, where each element indicates the ID
      of the supervoxel that voxel belongs to.
    """

    # Extract row, col, and depth from the volume mask
    r, c, d = vol_mask.shape[:3]
    rg, cg, dg = np.meshgrid(
        range(1, r, seed_dist[0]), range(1, c, seed_dist[1]), range(1, d, seed_dist[2]))
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
    x_hat = np.concatenate((np.transpose(s), feats), axis=1)

    w = np.array([1, 1 / r, 1 / c, 1 / d]) * LAMBDA

    # Create initial centroids
    print("Creating initial seeds...")
    c0, r0, d0 = np.meshgrid(
        np.arange(1, c, seed_dist[1]),
        np.arange(1, r, seed_dist[0]),
        np.arange(1, d, seed_dist[2])
    )
    centroids = np.array([r0[:], c0[:], d0[:]])
    if centroids.size == 0:
        raise Exception("Bad seed step size. No centroids to initialize.")

    # Initialize clusters: Each centroid gets its own cluster initially
    clusters = np.zeros(np.shape(centroids)[0], np.shape(x_hat)[1]) # Stores values for each cluster
    num_clusters = np.shape(clusters)[0]
    cluster_isolated = np.full(np.shape(centroids)[0], False)

    for k, centroid in enumerate(centroids):
        # Find neighboring points for each centroid
        neighbor_idxs = nearest_neighbors(centroid, centroids, seed_dist)

        # If cluster has no neighbor, mark as isolated
        if len(neighbor_idxs) == 0:
            cluster_isolated[k] = True
        else:
            # Assign value to each cluster
            nh_feats = x_hat[neighbor_idxs,:]
            clusters[k,:] = np.mean(nh_feats, axis=0)
    clusters = [c for i, c in enumerate(clusters) if not cluster_isolated[i]]

    # TODO Write k-means
    # TODO Implement nearest neighbor


def kmeans():
    """
    Compute k-means over a set of initial clusters
    """


def nearest_neighbors(center: np.ndarray, points: np.ndarray, step: Tuple) -> list:
    """
    Compute the nearest neighbors to a given center from candidate points.

    Parameters
    ----------
    `center` : The point for which to find neighbors

    `points` : Candidate points over which to search for neighbors to `center`

    `step` : Maximum distance between candidate neighbor and `center`, in the form of a tuple
    containing (row, column, depth) distances
    """

    # Check that dimensions match
    if np.shape(points)[1] != np.shape(center)[0]:
        sys.stderr.write("nearestneighbors: warning: Dimension mismatch. Returning empty list.")
        return []

    # TODO write lambda for if point is within step of center
    point_near_center = lambda p: all([center[i] - p[i] <= step[i] for i in range(len(step))])
    
    # Return all points within step size of center
    return [idx for idx, point in enumerate(points) if point_near_center(point)]
    

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

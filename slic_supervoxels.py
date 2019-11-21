from typing import Tuple

import numpy as np
from scipy import stats

# --- PARAMETERS ---
# Threshold of displacement of centroids
EPSILON = 1e-3

LAMBDA = np.ones(4)
MAX_ITER = 100


def slic(
    x: np.ndarray, vol_mask: np.ndarray, seed_dist: list, min_voxels: int
) -> Tuple[list, list, np.ndarray]:
    """Generate superpixel clusters based on feature vectors.

    Parameters
    ----------

    `x` : An M x P matrix, where M is the number of voxels and P is the number of features
    
    `vol_mask` : An R x C x D volume matrix filled with zeroes everywhere except the ROI, which is filled with ones

    `seed_dist` : A list of size 3, indicating distances between the centroids of supervoxels.
    As an example, `[3,2,5]` indicates that the centroids are 3 row, 2 column, and 5 depth units away from each other.

    `min_voxels` : The minimum number of voxels in a supervoxel

    Returns 
    -------

    A tuple, containing:

    * An M x 1 column vector, where the mth vector indicates the label of supervoxel m,
      a number from 1 to K.
    * A K x (P + |S|) matrix, where the kth row is the feature vector of the kth centroid.
    * An R x C x D volume matrix, where each element indicates the ID of the supervoxel that voxel belongs to.
    """

    # Extract row, col, and depth from the volume mask
    r, c, d = vol_mask.shape[:3]
    rg, cg, dg, = np.meshgrid(range(1, r), range(1, c), range(1, d))
    rg = rg[vol_mask[:] > 0]
    cg = cg[vol_mask[:] > 0]
    dg = dg[vol_mask[:] > 0]

    # Don't need to z-score spatial coords
    s = np.array([rg, cg, dg])
    smax = np.ndarray.max(s, axis=0, initial=None)
    smin = np.ndarray.min(s, axis=0, initial=None)

    # Z-score radiomic features
    m, p = x.shape
    x = stats.zscore(x, axis=0, ddof=1)

    # Used to determine which seed a pixel is assigned to
    x_hat = np.concatenate((np.transpose(s), x), axis=1)

    w = np.array([1, 1 / r, 1 / c, 1 / d]) * LAMBDA

    # Create initial centroids
    print("Creating initial seeds...")
    c0, r0, d0 = np.meshgrid(
        np.arange(1, c, seed_dist[1]), np.arange(1, r, seed_dist[0]), np.arange(1, d, seed_dist[2])
    )
    centroids = [r0[:], c0[:], d0[:]]

    # Initialize centroids

    # TODO Write k-means
    # TODO Implement nearest neighbor


def kmeans():
    """Compute k-means
    """


def nearestneighbors():
    """
    """

if __name__ == "__main__":
    # TODO parse incoming arguments and pass to slic()
    pass

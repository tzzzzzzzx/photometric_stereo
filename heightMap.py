import numpy as np
import scipy
import cv2


def compute_depth(mask, N):
    """
    compute the depth picture
    """
    M = scipy.sparse.lil_matrix((mask.size * 2, mask.size))
    v = np.zeros(M.shape[0], dtype=np.float32)

    nx = N[:, :, 0].ravel()
    ny = N[:, :, 1].ravel()
    nz = N[:, :, 2].ravel()

    v[0:nx.shape[0]] = -nx / (nz + 1e-8)
    v[nx.shape[0]:v.shape[0]] = -ny / (nz + 1e-8)

    obj_h, obj_w = np.where(mask != 0)
    # 得到非零元素的数量
    no_pix = np.size(obj_h)

    numPixels = mask.size
    width = mask.shape[1]
    height = mask.shape[0]

    for i in range(height):
        for j in range(width):
            pixel = width * i + j
            if j != width - 1:
                M[pixel, pixel] = -1
                M[pixel, pixel + 1] = 1
            if i != height - 1:
                M[pixel + numPixels, pixel] = -1
                M[pixel + numPixels, pixel + width] = 1

    MtM = M.T @ M
    Mtv = M.T @ v
    z, _ = scipy.sparse.linalg.cg(MtM, Mtv)
    std_z = np.std(z, ddof=1)
    mean_z = np.mean(z)
    z_zscore = (z - mean_z) / std_z

    # 因奇异值造成的异常
    outlier_ind = np.abs(z_zscore) > 10
    z_min = np.min(z[~outlier_ind])
    z_max = np.max(z[~outlier_ind])

    Z = mask.astype('float')
    for idx in range(no_pix):
        t_height = obj_h[idx]
        t_width = obj_w[idx]
        Z[t_height, t_width] = (z[t_height * width + t_width] -
                                z_min) / (z_max - z_min) * 255

    return Z


def save_depthmap(depth, filename=None):
    if filename is None:
        raise ValueError("filename is None")
    np.save(filename, depth)


def display_depthmap(depth=None, delay=0, name=None):
    depth = np.uint8(depth)
    if name is None:
        name = 'depth map'
    cv2.imshow(name, depth)
    cv2.waitKey()
    cv2.destroyAllWindows()

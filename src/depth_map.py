import os
import psutil
from matplotlib import pyplot as plt
import math
from scipy import linalg
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import numpy as np
import scipy
import cv2
def read_img(path):
    files = os.listdir(path)
    image_list = []
    for item in files:
        if item.find('Image') != -1:
            image_list.append(item)
    return image_list


def compute_depth_map(mask, N):
    """
    compute the depth picture
    """
    im_h, im_w = mask.shape
    N = np.reshape(N, (im_h, im_w, 3))

    # =================get the non-zero index of mask=================
    obj_h, obj_w = np.where(mask != 0)
    no_pix = np.size(obj_h)  # 37244
    # print(no_pix)
    full2obj = np.zeros((im_h, im_w))
    for idx in range(np.size(obj_h)):
        full2obj[obj_h[idx], obj_w[idx]] = idx
    full2obj = np.round(full2obj).astype(int)
    M = scipy.sparse.lil_matrix((2 * no_pix, no_pix))
    v = np.zeros((2 * no_pix, 1))

    # ================= fill the M&V =================
    for idx in range(no_pix):
        # obtain the 2D coordinate
        h = obj_h[idx]
        w = obj_w[idx]
        # obtian the surface normal vector
        if 0 < h < im_h - 1 and 0 < w < im_w - 1:
            n_x = N[h, w, 0]
            n_y = N[h, w, 1]
            n_z = N[h, w, 2]

            row_idx = idx * 2
            if mask[h, w + 1]:
                idx_horiz = full2obj[h, w + 1]
                M[row_idx, idx] = -1
                M[row_idx, idx_horiz] = 1
                if n_z == 0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_x / n_z
            elif mask[h, w - 1]:
                idx_horiz = full2obj[h, w - 1]
                M[row_idx, idx_horiz] = -1
                M[row_idx, idx] = 1
                if n_z == 0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_x / n_z

            row_idx = idx * 2 + 1
            if mask[h + 1, w]:
                idx_vert = full2obj[h + 1, w]
                M[row_idx, idx] = 1
                M[row_idx, idx_vert] = -1
                if n_z == 0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_y / n_z
            elif mask[h - 1, w]:
                idx_vert = full2obj[h - 1, w]
                M[row_idx, idx_vert] = 1
                M[row_idx, idx] = -1
                if n_z == 0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_y / n_z

    # =================sloving the linear equations Mz = v=================

    MtM = M.T @ M
    Mtv = M.T @ v
    for i in range(MtM.shape[0]):
        MtM[i, i] = MtM[i, i] + 1e-6
    z = scipy.sparse.linalg.spsolve(MtM, Mtv)
    # print(z)
    std_z = np.std(z, ddof=1)
    mean_z = np.mean(z)
    z_zscore = (z - mean_z) / std_z
    outlier_ind = np.abs(z_zscore) > 10
    z_min = np.min(z[~outlier_ind])
    z_max = np.max(z[~outlier_ind])

    Z = mask.astype('float')
    for idx in range(no_pix):
        # obtain the position in 2D picture
        h = obj_h[idx]
        w = obj_w[idx]
        Z[h, w] = (z[idx] - z_min) / (z_max - z_min) * 255

    depth = Z
    return depth


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


def save_depth_map(filename,depth):
    """
    将深度图保存为npy格式
    :param filename: filename of a depth map
    :retur: None
    """
    psutil.save_depthmap_as_npy(filename=filename,depth=depth)

def save_depthmap_as_npy(filename=None, depth=None):
    """
    将深度图保存为npy
    :param filename: filename of the depth array
    :param normal: surface depth array
    :return: None
    """
    if filename is None:
        raise ValueError("filename is None")
    np.save(filename, depth)


def disp_depth_map(depth=None, mask=None, delay=0, name=None):
    """
    显示深度图
    :param depth: array of surface depth
    :param delay: duration (ms) for visualizing normal map. 0 for displaying infinitely until a key is pressed.
    :param name: display name
    :return: None
    """
    if depth is None:
        raise ValueError("Surface depth `depth` is None")
    if mask is not None:
        depth = depth * mask

    depth = np.uint8(depth)

    if name is None:
        name = 'depth map'

    cv2.imshow(name, depth)
    cv2.waitKey(delay)
    cv2.destroyAllWindows(name)
    cv2.waitKey(1)

def main(path,animal):
    mask_image_file = path + '/Objects' + '/' + str(animal) + '_mask.png'
    normal_map_file = path + '/Objects/normal.png'
    mask = cv2.imread(mask_image_file)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = np.array(mask)
    normal_map = cv2.imread(normal_map_file)
    normal_map = np.array(normal_map)

    depth_map = compute_depth_map(mask,normal_map)
    #scaled_depth_map = (depth_map / np.max(depth_map) * 255).clip(0, 255).astype(np.uint8)
    disp_depth_map(depth_map,mask)
    #print(depth_map.shape)




if __name__ == '__main__':
    path = '../data/frog'
    main(path,'frog')

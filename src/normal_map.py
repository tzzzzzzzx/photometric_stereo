import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import numpy
import math
from scipy import linalg


def compute_normals(light_matrix, mask_array, images_array, threshold=100):
    shap = mask_array.shape
    shaper = (shap[0], shap[1], 3)

    normal_map = np.zeros(shaper)
    ivec = np.zeros(len(images_array))

    for (xT, value) in np.ndenumerate(mask_array):
        if(value > threshold):
            for (pos, image) in enumerate(images_array):
                ivec[pos] = image[xT[0], xT[1]]

            (normal, res, rank, s) = linalg.lstsq(light_matrix, ivec)

            normal = normal/linalg.norm(normal)

            if not np.isnan(np.sum(normal)):
                normal_map[xT] = normal

    return normal_map

def compute_albedo(light_matrix, mask_array, images_array, normal_map, threshold=100):
    shap = mask_array.shape
    shaper = (shap[0], shap[1], 3)

    albedo_map = np.zeros(shaper)
    ivec = np.zeros((len(images_array), 3))

    for (xT, value) in np.ndenumerate(mask_array):
        if(value > threshold):
            for (pos, image) in enumerate(images_array):
                ivec[pos] = image[xT[0], xT[1]]

            i_t = np.dot(light_matrix, normal_map[xT])

            k = np.dot(np.transpose(ivec), i_t)/(np.dot(i_t, i_t))

            if not np.isnan(np.sum(k)):
                albedo_map[xT] = k

    return albedo_map

def photometric_stereo(images_files, mask_image_file, lights_file, threshold=25):
    f = open(lights_file, encoding='utf-8')
    txt = []
    for line in f:
        txt.append(line.strip())
    lights = []
    tot = len(txt)
    for i in range(tot):
        w = txt[i].split()
        lights.append((float(w[0])/tot,float(w[1])/tot,float(w[2])/tot))
    mask_image = cv2.imread(mask_image_file)
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    mask_image_array = np.array(mask_image_gray)

    images = []
    images_array = []
    images_gray = []
    images_gray_array = []

    for image_file in images_files:
        image = cv2.imread(path +'/Objects/'+image_file)
        image_array = np.array(image)

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray_array = np.array(image_gray)

        images.append(image)
        images_array.append(image_array)
        images_gray.append(image_gray)
        images_gray_array.append(image_gray_array)

    normal_map = compute_normals(np.array(lights), mask_image_array, images_gray_array)
    albedo_map = compute_albedo(np.array(lights), mask_image_array, images_array, normal_map)

    return normal_map, albedo_map

def read_img(path):
    files = os.listdir(path)
    image_list = []
    for item in files:
        if item.find('Image') != -1:
            image_list.append(item)
    return image_list

def main(path,animal):
    images_files = read_img(path+'/Objects')
    mask_image_file = path +'/Objects'+ '/'+str(animal)+'_mask.png'
    lights_file = path+'/light.txt'
    normal_map,albedo_map = photometric_stereo(images_files,mask_image_file,lights_file)
    normal_map = np.array(normal_map)
    albedo_map = np.array(albedo_map)
    scaled_normal_map = (normal_map * 255 / np.max(normal_map)).astype(np.uint8)
    scaled_albedo_map = (albedo_map * 255 / np.max(albedo_map)).astype(np.uint8)


    plt.imshow(cv2.cvtColor(np.float32(scaled_normal_map / 255), cv2.COLOR_BGR2RGB))
    plt.show()

    plt.imshow(cv2.cvtColor(np.float32(scaled_albedo_map / 255), cv2.COLOR_BGR2RGB))
    plt.show()



if __name__ == '__main__':
    path = '../data/cat'
    main(path,'cat')
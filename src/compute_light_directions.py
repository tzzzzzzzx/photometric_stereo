import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import argparse
import numpy
import scipy.misc
import math
def init_circle_data(path):
    f = open(path+'/circle_data.txt', encoding='utf-8')
    txt = []
    for line in f:
        txt.append(line.strip())
    cx = float(txt[0])
    cy = float(txt[1])
    r = float(txt[2])
    return cx,cy,r

def read_circle(path):
    file = os.listdir(path)
    image_list = []
    for item in file:
        if item.find('Image') != -1:
            image_list.append(item)
    return image_list

def generate_mask(path):
    image_list = read_circle(path)
    # 生成金属球的mask
    file = path + '/' + image_list[0]
    img = cv2.imread(file)
    shape = img.shape
    h, w = shape[0], shape[1]
    shape = np.array(shape)
    mask = np.zeros(shape, np.uint8)
    cx, cy, r = init_circle_data(path)
    for y in range(h):
        for x in range(w):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                mask[y][x] = (255, 255, 255)
            else:
                mask[y][x] = (0, 0, 0)
    plt.imshow(cv2.cvtColor(np.float32(mask / 255), cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite(path + '/' + 'mask.jpg', mask)

def filterF(a, threshold):
    if a > threshold:
        return (255, 0, 0, 0)
    else:
        return (0, 0, 0, 0)

def compute_centroid(im_array, threshold=100):
    tot = 0.0
    xS = 0
    yS = 0
    for (x,value) in numpy.ndenumerate(im_array):
        if value > threshold:
            xS = xS + x[0]
            yS = yS + x[1]
            tot = tot+1

    return (xS/tot, yS/tot)

def compute_light_directions(path):
    image_list = read_circle(path)
    cx,cy,r = init_circle_data(path)
    R = np.array([0, 0, 1])
    mask_image = cv2.imread(path+'/'+'mask.jpg')
    mask_image_gray =  cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    mask_image_array = Image.fromarray(mask_image_gray)
    light_directions = []
    for image_file in image_list:
        image = cv2.imread(path+'/'+image_file)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_array = Image.fromarray(image_gray)
        centroid = compute_centroid(image_array)
        dx = centroid[1] - cy
        dy = centroid[0] - cx
        dy = -dy
        N = np.array([dx / r,
                         dy / r,
                         math.sqrt(r * r - dx * dx - dy * dy) / r])
        L = 2 * np.dot(N, R) * N - R
        light_directions.append(L)
    return light_directions
def avg_light_direction(path):
    light_directions_1 = compute_light_directions(path+'/'+'LightProbe-1')
    light_directions_2 = compute_light_directions(path+'/'+'LightProbe-2')
    light_directions = []
    for i in range(len(light_directions_1)):
        d = (light_directions_1[i] + light_directions_2[i]) / 2
        light_directions.append(d)
    with open(path+'/light.txt','w') as output_file:
        for light in light_directions:
            output_file.write('%lf %lf %lf\n' % (light[0], light[1], light[2]))

path = '../data/scholar'
avg_light_direction(path)



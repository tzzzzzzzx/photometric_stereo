import cv2
import numpy as np
import os
from sklearn.preprocessing import normalize

from matplotlib import pyplot as plt
from heightMap import compute_depth, save_depthmap,display_depthmap
from norm_vector import compute_surfNorm
from point_cloud import visualize


#显示图片
def display(img):
    plt.imshow(cv2.cvtColor(np.float32(img/255), cv2.COLOR_BGR2RGB))
    plt.show()

def main(Image_name):

    # =================read the information in MASK=================
    mask = cv2.imread('mask\\'+Image_name+'_mask.png')
    mask2 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    height,width,_=mask.shape
    dst=np.zeros((height,width,3),np.uint8)
    for k in range(3):
        for i in range(height):
            for j in range(width):
                dst[i,j][k]=255-mask[i,j][k]

    # ================obtain the light vector=================
    file_path = 'light\\'+Image_name+'_light.txt'
    file = open(file_path,'r')
    L=[]
    i=0
    while 1:
        line = file.readline()
        if not line:
            break
        if(i!=0):
            line = line.split("\t")
            #print(line)
            line[2] = line[2].replace("\n",'')
            for l in range(3):
                line[l] = float(line[l])
            L.append(tuple(line))
        i+=1
    file.close()
    L = np.array(L)
    

    # =================obtain picture infor=================
    dir = 'data\\'+Image_name+'\\Objects'
    imgList = os.listdir(dir)
    I = []
    for i in range(len(imgList)):
        picture = cv2.imread(dir+'\\'+imgList[i])
        picture = cv2.cvtColor(picture,cv2.COLOR_RGB2GRAY)
        height, width = picture.shape #(340, 512)
        picture = picture.reshape((-1,1)).squeeze(1)
        I.append(picture)
    I = np.array(I)
    

    # =================compute surface normal vector=================
    normal = compute_surfNorm(I, L)
    normal = normalize(normal, axis=1) 
    
    
    N = np.reshape(normal.copy(),(height, width, 3))
    # RGB to BGR
    N[:,:,0], N[:,:,2] = N[:,:,2], N[:,:,0].copy()
    N = (N + 1.0) / 2.0
    result = N + dst
    result = result * 255
    display(result)
    cv2.imwrite('result//'+Image_name+'//norm_'+Image_name+'.jpg',result)
    
    f = open('result//'+Image_name+'//norm_'+Image_name+'.txt','w')
    for nor in normal:
        f.write(str(nor)+'\n')
    f.close
 
    # =================compute depth map=================
    
    Z = compute_depth(mask=mask2.copy(),N=normal.copy())
    depth_path = str('result//'+Image_name+'//depth_'+Image_name+'.npy')
    save_depthmap(Z,filename=depth_path)
    display_depthmap(depth=Z,name="height")

    # =================generate the obj file to visualize=================
    visualize(depth_path,Image_name)
    

if __name__ == "__main__":
    main('cat')
    main('frog')
    main('scholar')
    
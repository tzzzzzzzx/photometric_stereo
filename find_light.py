import numpy as np
import cv2
import os

def find_light(img):

    img = cv2.GaussianBlur(img, (9, 9), 2.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    cv2.circle(img, maxLoc, 5, (255, 0, 0), 2)
    return list(maxLoc)

def light(P,C,R):
    L = []
    for i in range(len(P)):
        Nx = P[i][0] - C[0]
        Ny = P[i][1] - C[1]
        Ny = -Ny
        Nz = (C[2] ** 2. - Nx ** 2. - Ny ** 2.)
        if Nz < 0:
            Nz = -np.sqrt(Nz)
        else:
            Nz = np.sqrt(Nz)
        N = np.array([Nx, Ny, Nz])
        # N = np.array([Ny, Nx, Nz])
        l = 2 * (N @ R) * N - R
        l = l / np.linalg.norm(l)
        L.append(l)
    return L

def Avge(L1,L2):
    L = []
    for i in range(len(L1)):
        l = []
        for j in range(len(L1[0])):
            l.append(L1[i][j]/2+L2[i][j]/2)
        L.append(l)
    return L

    
S = ['cat','frog','scholar']
for stem in S:
    P = []
    dir = './data/'+stem+'/LightProbe-1'
    imgList = os.listdir(dir)
    print(imgList)
    P1 = []
    P2 = []
    for i in range(1,21):

        path1 = './data/'+stem+'/LightProbe-1/'+imgList[i]
        #print(path1)
        img = cv2.imread(path1)
        loc1 = find_light(img)
        
        path2 = './data/'+stem+'/LightProbe-2/'+imgList[i]
        img = cv2.imread(path2)
        loc2 = find_light(img)
        
        P1.append(loc1)
        P2.append(loc2)
        
        loc = [int(loc1[0]/2+loc2[0]/2),int(loc1[1]/2+loc2[1]/2)]
        
        '''
        cv2.circle(img, tuple(loc2), 5, (255, 0, 0), 2)
        cv2.imshow('mask', img)
        cv2.waitKey()
        cv2.destroyAllWindows() 
        '''
    path1 = './data/'+stem+'/LightProbe-1/'+imgList[0]
    path2 = './data/'+stem+'/LightProbe-2/'+imgList[0]
    R = np.array([0,0,1])
    f = open(path1)
    print(path1)
    C1 = []
    for lines in f.readlines():
        C1.append(float(lines.strip('\n')))
    f.close()
    
    f = open(path2)
    C2 = []
    for lines in f.readlines():
        C2.append(float(lines.strip('\n')))
        
    f.close()
    L1 = light(P1,C1,R)
    L2 = light(P2,C2,R)
    L = Avge(L1,L2)
    
    f = open('light/'+stem+'_light.txt','w+')
    f.write(str(len(L))+'\n')
    for i in range(len(L)):
        f.write(str(L[i][0])+'\t'+str(L[i][1])+'\t'+str(L[i][2])+'\n')
    f.close()
    
            





    
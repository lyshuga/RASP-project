import numpy as np
from numpy import linalg
from scipy import signal

def optical_flow(I1g, I2g, window_size, tau=1e-2):
 
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    w = int(window_size/2) # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255. # normalize pixels
    I2g = I2g / 255. # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            #b = ... # get b here
            #A = ... # get A here
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            b = np.reshape(It, (It.shape[0],1)) # get b here
            A = np.vstack((Ix, Iy)).T # get A here

            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b) # get velocity here
                u[i,j]=nu[0]
                v[i,j]=nu[1]
 
    return (u,v)

import cv2

from dataLoader import DataLoader

loader = DataLoader("dataset-opticalflow-eval/Backyard")
pictures = []

for _ in range(loader.maxIndex):

    images = loader.getPair(False)

    img1 = images[0]
    img2 = images[0]

    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    vector = optical_flow(img1, img2, window_size=35, tau=1e-2)

    img = images[0]


    for i in range(0,vector[0].shape[0],10):
        y_temp = vector[0].shape[0] - i
        for j in range(0,vector[0].shape[1],10):
            u = vector[0][i,j]
            v = vector[1][i,j]

            cv2.arrowedLine(img, (j,i), (int(j+v), int(i+u)), (1,0,0))
    
    pictures.append(img)

import glob
show = True

while show:
    for img in pictures:
        cv2.imshow('mahMovie', img)
        k = cv2.waitKey(120)
        if k == 27:
            cv2.destroyAllWindows()
            show=False
            break

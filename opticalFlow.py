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
        
            A = np.stack((np.array(Ix).T, np.array(Iy).T))
            #A = np.array(A)
            b = np.array(-It)

        # if threshold τ is larger than the smallest eigenvalue of A'A:
            if tau < min(linalg.eig(A.T.dot(A))[0]):
        #nu = ... # get velocity here
                nu = linalg.pinv(A).T.dot(b)
                u[i,j]=nu[0]
                v[i,j]=nu[1]
    return (u,v)

import cv2

img1 = cv2.imread('C:\\Users\\Tulek\\Desktop\\RASP\RASP-Projekt\\dataset-opticalflow-eval\\Basketball\\frame07.png')
img2 = cv2.imread('C:\\Users\\Tulek\\Desktop\\RASP\RASP-Projekt\\dataset-opticalflow-eval\\Basketball\\frame08.png')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

vector = optical_flow(img1, img2, window_size=2, tau=0.1)
print(vector)
print(vector[0].shape)
print(vector[1].shape)

cv2.imshow('frame1',vector[0])
cv2.imshow('frame2',vector[1])
cv2.waitKey(0)
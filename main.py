import numpy as np
import cv2
import sys

from dataLoader import DataLoader
from opticalFlow import OpticalFlow
from sklearn.cluster import DBSCAN

data_path = sys.argv[1]
loader = DataLoader(data_path, video=True)

# Parameters for lucas kanade optical flow
window_size = 65
lk_params = dict( winSize  = (window_size, window_size),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.02))

_, gray_frame = loader.getFrame()
flow = OpticalFlow(lk_params, first_frame=gray_frame, grid_space=20)

class_colors = {}

while True:
    frame, gray = loader.getFrame()
    data = flow.calculateOpticalFlow(gray, treshold=1)

    for n in data:
        frame = cv2.arrowedLine(frame, (n[0], n[1]), (n[2] + n[0], n[3] + n[1]), (0, 0, 255)) # drawing arrows


    cluster_data = [d for d in data if d[2] != 0 and d[3] != 0]
    if cluster_data != []:
        clustering = DBSCAN(eps=60, min_samples=10).fit(cluster_data) 
        for i,d in enumerate(cluster_data):
            if clustering.labels_[i] in class_colors.keys():
                color = class_colors[clustering.labels_[i]]
            else:
                color = np.random.randint(0,255,(1,3))[0]
                color = tuple(map(int, color))
                class_colors[clustering.labels_[i]] = color

            frame = cv2.circle(frame, (d[0], d[1]), 4, color, -1)

    cv2.imshow("Showcase",frame) 
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
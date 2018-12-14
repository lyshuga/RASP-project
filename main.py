import numpy as np
import cv2
import sys
# import keras


from dataLoader import DataLoader
from opticalFlow import OpticalFlow
from DBSCAN import DBSCAN
import crowd_tracking as ct

data_path = sys.argv[1]
loader = DataLoader(data_path, video=True)

# Parameters for lucas kanade optical flow
window_size = 55
lk_params = dict( winSize  = (window_size, window_size),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.02))

_, gray_frame = loader.getFrame()
flow = OpticalFlow(lk_params, first_frame=gray_frame, grid_space=20)

class_colors = {}

prev_cluster_info = None

while True:
    frame, gray = loader.getFrame()
    data = flow.calculateOpticalFlow(gray, treshold=1)

    for n in data:
        frame = cv2.arrowedLine(frame, (n[0], n[1]), (n[2] + n[0], n[3] + n[1]), (0, 0, 255)) # drawing arrows


    cluster_data = [d for d in data if d[2] != 0 and d[3] != 0]
    if cluster_data != []:
        clustering = DBSCAN(cluster_data,70, 5)
        
        #Do crowd_tracking, so that same clusters in two different frames get same id (color)
        cluster_info = ct.label_specific_properties(cluster_data, clustering)
        
        #TODO:
        #if prev_cluster_info is not None:
            
            
        for i,d in enumerate(cluster_data):
            if clustering[i]!=-1: #filer noise
                if clustering[i] in class_colors.keys():
                    color = class_colors[clustering[i]]
                else:
                    color = tuple(map(int, np.random.randint(0,255,(1,3))[0]))
                    class_colors[clustering[i]] = color

                frame = cv2.circle(frame, (d[0], d[1]), 4, color, -1)
        
        prev_cluster_info = cluster_info
        
    cv2.imshow("Showcase",frame) 
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
import numpy as np
import cv2
import sys
import os
import pprint as pp
import time
time.time

from dataLoader import DataLoader
from opticalFlow import OpticalFlow
from DBSCAN import DBSCAN
from crowd_tracker import CrowdTracker

data_path = sys.argv[1]

# To save results
if not os.path.exists('out'):
   os.makedirs('out')
out_dir_name = 'out\\'+data_path.split('\\')[-1]+'_'+str(time.time())
os.makedirs(out_dir_name)

loader = DataLoader(data_path, video=True)

#########################
# Parameters

# Optical Flow
window_size = 13 #for 
grid_space = 10

# Crowd Tracking
T = 0.6
a_1 = 0.7
a_2 = 0.2
a_3 = 0.1
norm_a_1 = 500
norm_a_2 = 180
norm_a_3 = 800

# DBSCAN
eps = 25 
min_points = 5

###########################
lk_params = dict( winSize  = (window_size, window_size),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.02))

_, gray_frame = loader.getFrame()
flow = OpticalFlow(lk_params, first_frame=gray_frame, grid_space=grid_space)




crowd_tracker = CrowdTracker(T=T, a_1=a_1, a_2=a_2, a_3=a_3, norm_a_1=norm_a_1, norm_a_2=norm_a_2, norm_a_3=norm_a_3)
cp_prev = None

class_colors = {}
frame_counter = 0
while True:
    frame_counter+=1
    frame, gray = loader.getFrame()
    data = flow.calculateOpticalFlow(gray, treshold=1)
   
    if data != []:
        # TODO remove noise from clustering, because for high min_pts sometimes only noise is returned
        clustering = DBSCAN(data, eps, min_points)
        
        # Unify class IDs from previous and current frame
        clustering, cp = crowd_tracker.map_cluster_IDs(data, clustering)

        for i,d in enumerate(data):
            if clustering[i]!=-1: #filter noise
                if clustering[i] in class_colors.keys():
                    color = class_colors[clustering[i]]
                else:
                    color = tuple(map(int, np.random.randint(0,255,(1,3))[0]))
                    class_colors[clustering[i]] = color

                frame = cv2.arrowedLine(frame, (d[0], d[1]), (d[2], d[3]), (0, 0, 255)) # drawing arrows
                frame = cv2.circle(frame, (d[0], d[1]), 4, color, -1)
        
        # Draw center of each cluster
        if cp_prev is not None:
            for i in cp:
                coord = (int(cp[i]['center'][0]), int(cp[i]['center'][1]))
                frame = cv2.circle(frame, coord, 14, (0,255,0), -1 )
                frame = cv2.putText(frame, str(i),(coord[0]-10,coord[1]+8), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,0), thickness= 3)
            
        cp_prev = cp
    cv2.imshow("Showcase",frame) 
    cv2.imwrite(out_dir_name+f'\\frame_{frame_counter}.jpg', frame)
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break

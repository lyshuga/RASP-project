import cv2
import math
import numpy as np

class OpticalFlow:
    def __init__(self, lk_params, first_frame, grid_space=20):
        self.lk_params = lk_params
        self.prev_frame = first_frame.copy()

        self._setPointsToTrack(grid_space)

    def _setPointsToTrack(self, grid_space):
        self.points_to_track = []
        for i in range(0, self.prev_frame.shape[0], grid_space):
            for j in range(0, self.prev_frame.shape[1], grid_space):
                self.points_to_track.append([[np.float32(j), np.float32(i)]])
        
        self.points_to_track = np.array(self.points_to_track)

    
    def calculateOpticalFlow(self, frame, treshold=1):
        def euclidianDistance(x1, x2):
            x = np.square(x1[0] - x2[0]) + np.square(x1[1] - x2[1])
            x = np.sqrt(x)
            return x
        
        new_points, _, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, frame, self.points_to_track, None, **self.lk_params)

        flow_data = []

        for (new, old) in zip(new_points, self.points_to_track):
                x_old, y_old = old.ravel()
                x_new, y_new = new.ravel()
                m = euclidianDistance([x_old,x_new],[y_old,y_new])
                angle_vector = (x_new-x_old,y_new-y_old)
                if angle_vector[0] == 0:
                    alpha = math.degrees(math.atan(0))
                else:
                    alpha = math.degrees(math.atan(angle_vector[1]/angle_vector[0]))
                
                #we throw out points that have not moved far enough
                if m >= treshold:
                    flow_data.append([int(x_old),
                                    int(y_old),
                                    int(x_new),
                                    int(y_new),
                                    int(m), 
                                    int(alpha)])
                                    
        self.prev_frame = frame.copy()
        return flow_data
        
    
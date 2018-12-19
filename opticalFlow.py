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
                a, b = old.ravel()
                c, d = new.ravel()
                m = euclidianDistance([a,b],[c,d])
                vector = (c-a,d-b)
                #flow_data_matrix = np.full((width, height), None)
               
                if vector[0] == 0:
                    angle = math.degrees(math.atan(0))
                else:
                    angle = math.degrees(math.atan(vector[1]/vector[0]))
                flow_data.append([int(a), int(b), int(c), int(d), int(c-a), int(d-b), int(m), int(angle)])
                # if math.sqrt(pow(c-a , 2.0) + pow(d-b , 2.0)) >= treshold:
                #     flow_data.append([int(a), int(b) , int(c-a), int(d-b)]) # storing vector components
                # else:
                #     flow_data.append([int(a), int(b) , 0, 0])

        self.prev_frame = frame.copy()
        return flow_data
        
    
import numpy as np
import cv2

from dataLoader import DataLoader

loader = DataLoader("dataset-opticalflow-eval/Backyard")
images = loader.getPair(True,(3,3))

cv2.imshow("Integral Image",images[0]) 
cv2.waitKey()
cv2.destroyAllWindows()
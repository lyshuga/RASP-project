import glob
import cv2
import numpy as np

cap = cv2.VideoCapture('./dataset-opticalflow-eval/video.mp4')

# Parameters for lucas kanade optical flow
window_size = 45
lk_params = dict( winSize  = (window_size, window_size),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.02))

_, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = []
points = 20
for i in range(0, old_frame.shape[0], points):
    for j in range(0, old_frame.shape[1], points):
        p0.append([[np.float32(j), np.float32(i)]])

p0 = np.array(p0)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

show = True
treshold = 1

while show:
    _, frame = cap.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1
    good_old = p0
    #img = cv.add(frame,mask)
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = old.ravel()
        c,d = new.ravel()
        if abs(a-c) > treshold and abs(b-d) > treshold: 
            frame = cv2.arrowedLine(frame, (a,b),(c,d), (0, 0, 255))
        #frame = cv2.circle(frame,(a,b),2,12,-1)
    cv2.imshow('mahMovie', frame)
    k = cv2.waitKey(33)
    if k == 27:
        cv2.destroyAllWindows()
        show=False
        break
    old_gray = frame_gray.copy()

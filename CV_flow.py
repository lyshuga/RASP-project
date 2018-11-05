import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
_, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

p0 = []
for i in range(0, old_frame.shape[0], 50):
    for j in range(0, old_frame.shape[1], 50):
        p0.append([[np.float32(j), np.float32(i)]])

p0 = np.array(p0)
print(p0)
print(type(p0))
#p0 = np.array([1,2])
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    _, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1
    good_old = p0
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = old.ravel()
        c,d = new.ravel()
        frame = cv.arrowedLine(frame, (a,b),(c,d), 12)
        frame = cv.circle(frame,(a,b),2,12,-1)
    #img = cv.add(frame,mask)
    cv.imshow('frame',frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    #p0 = good_new.reshape(-1,1,2)
cv.destroyAllWindows()
cap.release()
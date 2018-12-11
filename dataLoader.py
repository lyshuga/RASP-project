import glob
import os 
import cv2

class DataLoader:
    def __init__(self, path, video=False):
        self.path = path
        self.index = 0

        if not video:
            self.files = glob.glob(self.path + "/*.*")
            self.maxIndex = len(self.files)
        else:
            self.videoData = cv2.VideoCapture(self.path)

    def getFrame(self):
        _, frame = self.videoData.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame, gray

    def getPair(self, grayscale=True, blur=True, blurSize=(3,3)):
        if(self.index + 1 >= self.maxIndex):
            self.index = 0
        img1 = cv2.imread(self.files[self.index])
        img2 = cv2.imread(self.files[self.index + 1])
        if(grayscale):
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        if(blur):
            img1 = cv2.blur(img1,blurSize)
            img2 = cv2.blur(img2,blurSize)
        pair = (img1, img2)
        self.index = self.index + 1
        return pair

import glob
import os 
import cv2

class DataLoader:
    def __init__(self, path):
        self.path = path
        self.index = 0
        #load data
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.files = glob.glob(dir_path+"/"+self.path+"/*.*")
        self.maxIndex = len(self.files)

    def getPair(self):
        if(self.index + 1 >= self.maxIndex):
            self.index = 0;
        img1 = cv2.cvtColor(cv2.imread(self.files[self.index]), cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(cv2.imread(self.files[self.index + 1]), cv2.COLOR_BGR2GRAY)
        pair = (img1, img2)
        self.index = self.index + 1
        return pair

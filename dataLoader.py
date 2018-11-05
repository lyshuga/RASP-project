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

    def getPair(self, grayscale=True, blur=True, blurSize=(3,3)):
        if(self.index + 1 >= self.maxIndex):
            self.index = 0;
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

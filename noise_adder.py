import cv2
import numpy as np

class noise:
    def add_noise(self,img):
        w,h,c=img.shape
        noise=np.random.randn(w,h,c)
        img=img+noise*np.random.randint(0,255)
        return img

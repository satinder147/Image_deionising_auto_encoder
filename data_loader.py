import os
import cv2
from noise_adder import noise
from keras.preprocessing.image import img_to_array



class loader:
    def __init__(self):
        self.x_data=[]
        self.y_data=[]
        self.noise=noise()

    def load(self,path):
        files=os.listdir(path)
        j=0
        for i in files:
            img=cv2.imread(path+'/'+i,1)
            self.x_data.append(img_to_array(img))
            img_noise=self.noise.add_noise(img)
            self.y_data.append(img_to_array(img_noise))

        return self.x_data,self.y_data

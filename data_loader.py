import os
import cv2
from noise_adder import noise
from keras.preprocessing.image import img_to_array



class loader:
    def __init__(self):
        self.x_data=[]
        self.y_data=[]
        self.noise=noise()

    def load(self,path1,path2,path3):
        files=os.listdir(path1)

        for i in files:
            img=cv2.imread(path1+'/'+i,1)
            img=cv2.resize(img,(240,240))
            self.y_data.append(img_to_array(img))
            img_noise=self.noise.add_noise(img)
            self.x_data.append(img_to_array(img_noise))
        files=os.listdir(path2)

        for i in files:
            img=cv2.imread(path2+'/'+i,1)
            img=cv2.resize(img,(240,240))
            self.y_data.append(img_to_array(img))
            img_noise=self.noise.add_noise(img)
            self.x_data.append(img_to_array(img_noise))
        files=os.listdir(path3)

        for i in files:
            img=cv2.imread(path3+'/'+i,1)
            img=cv2.resize(img,(240,240))
            self.y_data.append(img_to_array(img))
            img_noise=self.noise.add_noise(img)
            self.x_data.append(img_to_array(img_noise))
        return self.x_data,self.y_data

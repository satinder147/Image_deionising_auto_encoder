import cv2
import os
from keras.models import load_model
from model import Models
from noise_adder import noise
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
n=noise()
m=Models(240,240,3)
img=cv2.imread('6.jpg',1)
img=cv2.resize(img,(240,240))
noise=n.add_noise(img)
img2=noise
img2=img2.astype('float')/255.0
img2=img_to_array(img2)
img2=np.expand_dims(img2,axis=0)
print(img2.shape)
mod=m.Arch1()
mod.load_weights('auto.MODEL')
print('hello')
img2=mod.predict(img2)[0]
img2=img2.astype('float')

img2=img2*255.0
img2=img2.astype('uint8')
print(np.max(img2))

cv2.imshow('predicted',img2)
plt.show()
cv2.imshow('d',img)
cv2.imshow('noise',noise)
cv2.waitKey(0)

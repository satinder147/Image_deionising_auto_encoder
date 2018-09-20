import cv2
import numpy as np
from data_loader import loader
from model import Models
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

w=240
h=240
c=3

load_img=loader()
mod=Models(w,h,c)
auto_encoder=mod.Arch1()
auto_encoder.summary()
x_data,y_data=load_img.load('new')

x_data=np.array(x_data,dtype='float')/255.0
y_data=np.array(y_data,dtype='float')/255.0

train_x,test_x,train_y,test_y=train_test_split(x_data,y_data,test_size=0.1,random_state=30)
aug=ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2)
auto_encoder.compile(optimizer='adadelta',loss='binary_crossentropy')
auto_encoder.fit_generator(aug.flow(train_x,train_y,batch_size=64),epochs=50,validation_data=(test_x,test_y),verbose=1)
auto_encoder.save('auto.MODEL')

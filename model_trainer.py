import cv2
import numpy as np
from data_loader import loader
from model import Models
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
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
opt=Adam(lr=0.01,decay=0.01/50)
train_x,test_x,train_y,test_y=train_test_split(x_data,y_data,test_size=0.1,random_state=30)
auto_encoder.compile(optimizer=opt,loss='binary_crossentropy')
auto_encoder.fit(train_x,train_y,batch_size=32,shuffle='true',epochs=50,validation_data=(test_x,test_y),verbose=1)
auto_encoder.save('auto.MODEL')

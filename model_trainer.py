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


mod=Models(w,h,c)
auto_encoder=mod.Arch2()
load_img=loader()
auto_encoder.summary()
x_data,y_data=load_img.load('stone','paper','scissor')
x_data=np.array(x_data,dtype='float')/255.0
y_data=np.array(y_data,dtype='float')/255.0
opt=Adam(lr=0.001,decay=0.001/50)
train_x,test_x,train_y,test_y=train_test_split(x_data,y_data,test_size=0.1,random_state=30)
auto_encoder.compile(optimizer='adadelta',loss='binary_crossentropy')
auto_encoder.fit(train_x,train_y,batch_size=32,shuffle='true',epochs=15,validation_data=(test_x,test_y),verbose=1)
auto_encoder.save('noise.MODEL')

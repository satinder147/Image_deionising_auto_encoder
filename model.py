from keras.models import Model
from keras.layers import Conv2D,UpSampling2D,MaxPooling2D
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras import regularizers

class Models:
    def __init__(self,w,h,c):
        self.w=w
        self.h=h
        self.c=c
    def Arch1(self):
        inp=Input(shape=(self.w,self.h,self.c))
        enc=Conv2D(64,(3,3),padding='same')(inp)
        enc=BatchNormalization()(enc)
        enc=LeakyReLU(alpha=0.1)(enc)
        enc=MaxPooling2D(pool_size=(2,2))(enc)
        enc=Conv2D(32,(3,3),padding='same')(enc)
        enc=LeakyReLU(alpha=0.1)(enc)
        enc=BatchNormalization()(enc)
        enc=MaxPooling2D(pool_size=(2,2))(enc)
        enc=Conv2D(16,(3,3),padding='same')(enc)
        enc=LeakyReLU(alpha=0.1)(enc)
        enc=BatchNormalization()(enc)
        enc=MaxPooling2D(pool_size=(2,2))(enc)
        enc=Conv2D(8,(3,3),padding='same')(enc)
        enc=LeakyReLU(alpha=0.1)(enc)
        enc=MaxPooling2D(pool_size=(2,2))(enc)


        dec=Conv2D(8,(3,3),padding='same')(enc)
        dec=LeakyReLU(alpha=0.1)(dec)
        dec=UpSampling2D((2,2))(dec)
        dec=Conv2D(16,(3,3),padding='same')(dec)
        dec=LeakyReLU(alpha=0.1)(dec)
        dec=UpSampling2D((2,2))(dec)
        dec=Conv2D(32,(3,3),padding='same')(dec)
        dec=LeakyReLU(alpha=0.1)(dec)
        dec=UpSampling2D((2,2))(dec)
        dec=Conv2D(64,(3,3),padding='same')(dec)
        dec=LeakyReLU(alpha=0.1)(dec)
        dec=UpSampling2D((2,2))(dec)
        final=Conv2D(3,(3,3),padding='same',activation='sigmoid')(dec)
        auto=Model(inp,final)
        return auto

    def Arch2(self):
        inp=Input(shape=(self.w,self.h,self.c))
        enc=Conv2D(64,(3,3),padding='same',activity_regularizer=regularizers.l1(10e-5))(inp)
        enc=BatchNormalization()(enc)
        enc=LeakyReLU(alpha=0.1)(enc)
        enc=MaxPooling2D(pool_size=(2,2))(enc)
        enc=Conv2D(32,(3,3),padding='same',activity_regularizer=regularizers.l1(10e-5))(enc)
        enc=LeakyReLU(alpha=0.1)(enc)
        enc=BatchNormalization()(enc)
        enc=MaxPooling2D(pool_size=(2,2))(enc)
        enc=Conv2D(16,(3,3),padding='same',activity_regularizer=regularizers.l1(10e-5))(enc)
        enc=LeakyReLU(alpha=0.1)(enc)
        enc=BatchNormalization()(enc)
        enc=MaxPooling2D(pool_size=(2,2),activity_regularizer=regularizers.l1(10e-5))(enc)
        enc=Conv2D(8,(3,3),padding='same')(enc)
        enc=LeakyReLU(alpha=0.1)(enc)
        enc=MaxPooling2D(pool_size=(2,2))(enc)


        dec=Conv2D(8,(3,3),padding='same',activity_regularizer=regularizers.l1(10e-5))(enc)
        dec=LeakyReLU(alpha=0.1)(dec)
        dec=UpSampling2D((2,2))(dec)
        dec=Conv2D(16,(3,3),padding='same',activity_regularizer=regularizers.l1(10e-5))(dec)
        dec=LeakyReLU(alpha=0.1)(dec)
        dec=UpSampling2D((2,2))(dec)
        dec=Conv2D(32,(3,3),padding='same',activity_regularizer=regularizers.l1(10e-5))(dec)
        dec=LeakyReLU(alpha=0.1)(dec)
        dec=UpSampling2D((2,2))(dec)
        dec=Conv2D(64,(3,3),padding='same',activity_regularizer=regularizers.l1(10e-5))(dec)
        dec=LeakyReLU(alpha=0.1)(dec)
        dec=UpSampling2D((2,2))(dec)
        final=Conv2D(3,(3,3),padding='same',activation='sigmoid')(dec)
        auto=Model(inp,final)
        return auto

import cv2
import os
import numpy
path='data'
os.mkdir('neww')
paths=os.listdir(path)

count=1
for i in paths:
    img=cv2.imread(path+'/'+i,1)
    img=cv2.resize(img,(240,240))
    cv2.imwrite('neww/'+str(count)+'.jpg',img)
    count=count+1

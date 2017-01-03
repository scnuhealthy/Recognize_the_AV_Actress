#coding:utf-8
import os
from PIL import Image
import numpy as np
import random
import cv2


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import h5py
from keras.models import model_from_json


IMAGE_SIZE = 50

def face_detect(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) >0:
        fc = faces[0]
        x = fc[0]
        y = fc[1]
        w = fc[2]
        h = fc[3]
        return img[y:y+h, x:x+w]
    else:
        return None
    
def get_max(array):
    max_num = max(array)
    for i in range(len(array)):
        if array[i] == max_num:
            return i

photo_path = '../predict_photo/photo'
face_path = '../predict_photo/face'


# load the test data
imgs_path = os.listdir(photo_path)
tol_num = len(imgs_path)

data = np.empty((tol_num,3,50,50),dtype="float32")
label = []

i = 0
for img_path in imgs_path:
    img = cv2.imread(photo_path+'/'+img_path)
    face = face_detect(img)
    if face != None:
        face= cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
        img_path = face_path+'/'+img_path
        cv2.imwrite(img_path,face)
        img = Image.open(img_path)
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
        label.append(imgs_path[i].split('/')[-1].split('.')[0])
        i += 1
    else:
        print 'cannot detect face'
        exit(0)

# load the trained model       
model = Sequential()

model = model_from_json(open('my_model.json').read())  
model.load_weights('my_model_weights.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# predict
predict = model.predict(data,batch_size = 64,verbose = 0)

name_list = ['shangyuanyayi','seguguobu','zuozuomumingxi','xiaotianbumei','aika']
name_class_dict={}
x = 0
for name in name_list:
    name_class_dict[x] = name
    x += 1
print predict
for i in range(len(predict)):
    print 'predict: ',name_class_dict[get_max(predict[i])],' true: ',label[i]

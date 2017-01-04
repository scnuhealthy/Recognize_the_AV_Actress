#Recognize the AV actress by CNN

##Summary
I want my computer to be able to recognize the actress by deep learning, even if there are Mosaics on the face.
![zuozuomumingxi](https://github.com/scnuhealthy/Recognize_the_AV_Actress/blob/master/predict_photo/photo/zuozuomumingxi.jpg)    

For example, it's difficult for human to regonize the women on the picture above. But my program tells me, she is zuozuomumingxi(佐々木あき). The whole process can be divided into 3 steps.
> * crawl the photoes of the AV actress
> * detect the the faces in the photo as the network's input
> * build a CNN depp learing model  

My program is applied in Python programming language. With its package keras and OpenCV, coding is not hard. Let us see the detail.

##Environment
My python's version is 2.7.11. I recommend to install [Anaconda](https://www.continuum.io/downloads), a leading open data science platform powered by python, including almost python packages you need. And two additional packages, [OpenCV](http://opencv.org/) and [keras](https://github.com/fchollet/keras), should be installed.

##Crawling
To get enough faces of a AV actress, a good way is to crawl the covers of her films. These faces are full of expression. Then I found an available website(http://www.nh87.cn/), we can crawl the covers on it easily. If you cannot crawl them or there is something wrong on the website, you can ask me. My email is 2506596454@qq.com. 
```python
import re
import os
import urllib2
import urllib

def url_open(url):
    req = urllib2.Request(url)
    req.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36')
    response=urllib2.urlopen(req)
    html = response.read().decode('utf-8')
    return html

# get the image url
def get_image(html):
    s = "data-original='([^']+)'" #too dog ' "
    img_list = re.findall(s,html)
    return img_list

name_list2 = ['boduoyejieyi','jizemingbu','tianhaiyi','jingxiangjulia','daqiaoweijiu','mrhql','baishimolinai']
name_list = ['shangyuanyayi','seguguobu','zuozuomumingxi','xiaotianbumei','aika']
name = name_list[4]
url = 'http://www.nh87.cn/'+name+'/'
html = url_open(url)
img_list =  get_image(html)
path = '../AV_photo/'+name

if os.path.exists(path) == False:
    os.mkdir(path)
for i in range(len(img_list)):
    imgurl = 'http://www.nh87.cn'+img_list[i]
    img_path = path+'/' + name + '_' + str(i)+'.jpg'    
    urllib.urlretrieve(imgurl,img_path) # download the image
```
##Detect the faces with OpenCV
```python
import cv2
IMAGE_SIZE = 50

# detect the face with opencv
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
```

If you have dowmloaded OpenCV successfully, the path in the file 'haarcascade_frontalface_default.xml' is oepncv/build/etc. The function **face_detect** will return the face image in the format of numpy array. If the original image has more than one face, it just returns the first face, and this face may not be the Actress's. And the result by OpenCV is not correct always. In my experience, there are about 20% faces are wrong.  To make sure the input data's accurary of  deep learning, you need to delete the wrong faces by hand.

##Build a CNN depp learing model
The network includes two convolution layers and two full-connected layers, looking like the mnist's.
```python
from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import h5py
from keras.models import model_from_json
from load_data import *

batch_size = 64
nb_classes = len(name_list)
nb_epoch = 128

# input image dimensions
img_rows, img_cols = 50, 50
# number of convolutional filters to use
nb_filters1 = 32
nb_filters2 = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters1, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))

model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))



model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
```
##The training result
![result](https://github.com/scnuhealthy/Recognize_the_AV_Actress/blob/master/result.png)
> * Data: the faces of shangyuanyayi, seguguobu, zuozuomumingyi, xiaotianbumei
> * Epoch: 128 times
> * Accuaracy on test set: over 80%

Dropout is very important. If you do this not well, you will get a good result on training set but much worse on test set. And the actresses you choose will have a great impact on the result. I try another dataset of boduoyejieyi, jizemingbu, tianhaiyi , jingxiangjulia, daqiaoweijiu, mrhql, baishimolinai, the accuaracy fells to 67%. I think the faces of those actresses are too similar to recognzie. 

##Reference resources
http://qiita.com/tmnck/items/af82deb04d432f1f4f6e

# Author:kemo
#coding:utf-8
import os
from PIL import Image
import numpy as np
import random

np.random.seed(133)  # for reproducibility

name_list2 = ['boduoyejieyi','jizemingbu','tianhaiyi','jingxiangjulia','daqiaoweijiu','mrhql']
name_list = ['shangyuanyayi','seguguobu','zuozuomumingxi','xiaotianbumei','aika']

#name to class
name_class_dict={}
x = 0
for name in name_list:
    name_class_dict[name] = x
    x += 1
    
def load_data():

 
    # data dir
    dir_path = "../Actress_face/"
    imgs_path = []
    for name in name_list:
        imgs_path_part = (os.listdir(dir_path+name))
        for i in range(len(imgs_path_part)):
                imgs_path_part[i] = dir_path+name+'/'+imgs_path_part[i]
        imgs_path += imgs_path_part

    tol_num = len(imgs_path)
    train_num = int(tol_num*0.8) # training samples accounts for 80%
    
    data = np.empty((tol_num,3,50,50),dtype="float32")
    label = np.empty((tol_num,),dtype="uint8")
    
    for i in range(tol_num):
        # load the images 
        img = Image.open(imgs_path[i])
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
        name_text = imgs_path[i].split('/')[-1].split('_')[0]
        label[i]= name_class_dict[name_text]
    # the data, shuffled and split between train and test sets
    rr = [i for i in range(tol_num)] 
    random.shuffle(rr)
    X_train = data[rr][:train_num]
    y_train = label[rr][:train_num]
    X_test = data[rr][train_num:]
    y_test = label[rr][train_num:]
    return (X_train,y_train),(X_test,y_test)

load_data()

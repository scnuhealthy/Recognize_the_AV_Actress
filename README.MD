#Recognize the AV actress by CNN

##Summary
I want my computer to be able to recognize the actress by deep learning, even if there are Mosaics on the face.
![zuozuomumingxi](https://github.com/scnuhealthy/Recognize_the_AV_Actress/blob/master/predict_photo/photo/zuozuomumingxi.jpg)    

For example, it's difficult for human to regonize the women on the picture above. But my program tells me, she is zuozuomumingxi(佐々木あき). 
My program is applied in Python programming language. With its package keras and OpenCV, coding is not hard. Let us see the detail.

##Environment
My python's version is 2.7.11. I recommend to install [Anaconda](https://www.continuum.io/downloads), a leading open data science platform powered by python, including almost python packages you need. And two additional packages, [OpenCV](http://opencv.org/) and [keras](https://github.com/fchollet/keras), should be installed.

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

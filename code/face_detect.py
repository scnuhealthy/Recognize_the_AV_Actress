import cv2
import os


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

name_list2 = ['boduoyejieyi','jizemingbu','tianhaiyi','jingxiangjulia','daqiaoweijiu','mrhql','baishimolinai']
name_list = ['shangyuanyayi','seguguobu','zuozuomumingxi','xiaotianbumei','aika']

name = name_list[4]

# read_path
photo_path = '../AV_photo/'+name

# write_path
face_path = '../Actress_face/'+name
if os.path.exists(face_path) == False:
    os.mkdir(face_path)

imgs_path = os.listdir(photo_path)

for img_path in imgs_path:
    img = cv2.imread(photo_path+'/'+img_path)
    face = face_detect(img)
    if face != None:
        face= cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
        cv2.imwrite(face_path+'/'+img_path,face)

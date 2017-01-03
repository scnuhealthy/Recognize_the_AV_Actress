#author:kemo
import re
import os
import urllib2
import urllib
import random


def url_open(url):

    req = urllib2.Request(url)
    req.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36')
    response=urllib2.urlopen(req)
    html = response.read().decode('utf-8')
    return html

def get_year(html,name):
    s = '<a href="/'+name+'/[0-9]{4}/">([0-9]{4})</a>'
    year_list = re.findall(s,html)
    return year_list

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

import numpy as np
from bs4 import BeautifulSoup
import os
import cv2
import pandas as pd
import shutil
import random

data_dir = r'path to xmls and images'
save_yolo_format = r'path where to make folder of yolo data'
validation_ratio = 0.2 # valudation split ratio

try:
    os.mkdir(save_yolo_format)
except:
    print(f"\n\n>>>>[WARNING]: directory : '{save_yolo_format}' :> Already Exists. values will be duplicated [remove folder manually]")
    
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x,y,w,h]


def make_yolo_format(data_dir):
    data_dir_list = os.listdir(data_dir)
    anotations = [i for i in data_dir_list if i[-3:]=='xml'][:]
    bbs = {}
    label_map = {}
    c = 0
    for xml_file_name in anotations:
        print('processing_file_no: ',c)
       
        xml_file_name = os.path.join(data_dir+xml_file_name)
        infile = open(xml_file_name,"r")
        contents = infile.read()
        soup = BeautifulSoup(contents,'xml')
        real_image_name = soup.find('filename').text
        image_name = os.path.join(data_dir+real_image_name)
        h,w = cv2.imread(image_name).shape[:2]
        classes = soup.find_all('name')
        xmin = soup.find_all('xmin')
        ymin = soup.find_all('ymin')
        xmax = soup.find_all('xmax')
        ymax = soup.find_all('ymax')
        bb = []
        classes__ = 0
        for class_,xmi,xma,ymi,yma in zip(classes,xmin,xmax,ymin,ymax):
            class_ = class_.text
            classes__+=1
            xmi,xma,ymi,yma = [int(i) for i in [xmi.text,xma.text,ymi.text,yma.text]]
            try:
                label = label_map[class_]
            except:
                label_map[class_] = len(label_map.keys())
                label = label_map[class_]
            bb.append([label]+convert((w,h), (xmi,xma,ymi,yma)))
        bbs[real_image_name] = bb
        c+=1
    return bbs,label_map

bbs,label_map = make_yolo_format(data_dir)
pd.DataFrame([str(label_map)]).to_csv(save_yolo_format+'labelmap.txt',header=False,index=False)
mk_tr_flag = 1



train_rat = int(len(bbs)*(1-validation_ratio))
items = list(bbs.items())
random.shuffle(items)
for cc,i in enumerate(items):
    df = pd.DataFrame(i[1])
    if mk_tr_flag:
        try:
            os.mkdir(os.path.join(save_yolo_format, 'data/'))
            os.mkdir(os.path.join(save_yolo_format, 'data/images/'))
            os.mkdir(os.path.join(save_yolo_format, 'data/images/train'))
            os.mkdir(os.path.join(save_yolo_format, 'data/images/val'))
            os.mkdir(os.path.join(save_yolo_format, 'data/labels/'))
            os.mkdir(os.path.join(save_yolo_format, 'data/labels/train'))
            os.mkdir(os.path.join(save_yolo_format, 'data/labels/val'))
        except:
            print(f"\n\n>>>>[WARNING]: directory : '{os.path.join(save_yolo_format, 'train')}' :> Already Exists")
        mk_tr_flag=0
    if cc<train_rat:
        df.to_csv(os.path.join(save_yolo_format,'data/labels/train','.'.join(i[0].split('.')[:-1])+'.txt'),header=False,index=False,sep = ' ')
        shutil.copy(os.path.join(data_dir,i[0]),os.path.join(save_yolo_format,'data/images/train',i[0]))
    else:
        df.to_csv(os.path.join(save_yolo_format,'data/labels/val','.'.join(i[0].split('.')[:-1])+'.txt'),header=False,index=False,sep = ' ')
        shutil.copy(os.path.join(data_dir,i[0]),os.path.join(save_yolo_format,'data/images/val',i[0]))
print('Done!!')
classes_ = list(label_map.keys())
nc = len(classes_)
yamlstr = f'''train: data/images/train/  \nval: data/images/val  \nnc: {nc}  \nnames: {classes_} '''
file1 = open(save_yolo_format+"cc.yaml","a")#append mode
file1.write(yamlstr)
file1.close()

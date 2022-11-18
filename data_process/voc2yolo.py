#该脚本文件需要修改第10行（classes）即可
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
from os import getcwd
from pathlib import Path
 
sets = ['train', 'test','val']
#这里使用要改成自己的类别
classes = ['round_pavilion', 'car', 'hydrological_station','tank', 'helicopter', 'umbralla', 'pavilion', 'rubber_boat', 'life_buoy']
# 车：car  圆亭子：round_pavilion 亭子：pavilion 水文站房：hydrological_station 坦克：tank 
# 直升机：helicopter  雨伞：umbralla 橡皮艇：rubber_boat  救生圈：life_buoy

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    x = round(x,6)
    w = round(w,6)
    y = round(y,6)
    h = round(h,6)
    return x, y, w, h
 
#后面只用修改各个文件夹的位置
def convert_annotation(image_id):
     #try:
    in_file = open('D:\Data\jiujiang_train_data\\train\\temp\labels\\%s.xml' % (image_id), encoding='utf-8')
    out_file = open('D:\Data\jiujiang_train_data\\train\\temp\yolo_labels\\%s.txt' % (image_id), 'w', encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes or int(difficult) == 1:
        #     continue
        if cls not in classes:
            print(cls)
        # 由于标签起始值为1，cls_id从0开始，相当于将原来标签中的id值全部减去1，提前一个
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        # return cls_id, bb
        out_file.write(str(cls_id) + " " +
                        " ".join([str(a) for a in bb]) + '\n')
     #except Exception as e:
         #print(e, image_id)
 
xml_label_path = Path('D:\Data\jiujiang_train_data\\train\\temp\\labels')
# yolo_label_dst = Path('D:\Data\训练数据\\train\dynamic_guanfang\yolo_labels')
xml_file_list = [f for f in xml_label_path.iterdir() if f.is_file]

for xml_file in tqdm(xml_file_list):
    xml_file_stem = xml_file.stem
    convert_annotation(str(xml_file_stem))

 
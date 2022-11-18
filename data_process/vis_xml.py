# import os
# import os.path
# import xml.etree.cElementTree as ET
# import cv2
# def draw(image_path, xml_path, root_saved_path):
#     """
#     图片根据标注画框
#     """
#     src_img_path = image_path
#     src_ann_path = xml_path
#     for file in os.listdir(src_ann_path):
#         # print(file)
#         file_name, suffix = os.path.splitext(file)
#         # import pdb
#         # pdb.set_trace()
#         if suffix == '.xml':
#             # print(file)
#             xml_path = os.path.join(src_ann_path, file)
#             image_path = os.path.join(src_img_path, file_name+'.jpg')
#             img = cv2.imread(image_path)
#             tree = ET.parse(xml_path)
#             root = tree.getroot()
#             # import pdb
#             # pdb.set_trace()
#             for obj in root.iter('object'):
#                 name = obj.find('name').text
#                 xml_box = obj.find('bndbox')
#                 x1 = int(xml_box.find('xmin').text)
#                 x2 = int(xml_box.find('xmax').text)
#                 y1 = int(xml_box.find('ymin').text)
#                 y2 = int(xml_box.find('ymax').text)
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
#                 # 字为绿色
#                 # cv2.putText(img, name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), thickness=2)
#             cv2.imwrite(os.path.join(root_saved_path, file_name+'.jpg'), img)
 
 
# if __name__ == '__main__':
#     image_path = "F:/bling/data/VisDrone2019-DET-train/images"
#     xml_path = "F:/bling/data/VisDrone2019-DET-train/Annotations_XML"
#     root_saved_path = "F:/bling/data/VisDrone2019-DET-train/result"
#     draw(image_path, xml_path, root_saved_path)



from logging import exception
import os
import os.path
import numpy as np
import xml.etree.ElementTree as xmlET
from PIL import Image, ImageDraw

#'1': 'people', '2': 'people','3': 'bicycle', '4': 'car', '5': 'car',
# 6':'others','7':'others','8':'others','9':'others','10': 'motor','11':'others'

import cv2 as cv
import os
import colorsys
import random

def get_n_hls_colors(num):
  hls_colors = []
  i = 0
  step = 360.0 / num
  while i < 360:
    h = i
    s = 90 + random.random() * 10
    l = 50 + random.random() * 10
    _hlsc = [h / 360.0, l / 100.0, s / 100.0]
    hls_colors.append(_hlsc)
    i += step

  return hls_colors

def ncolors(num):
  rgb_colors = []
  if num < 1:
    return rgb_colors
  hls_colors = get_n_hls_colors(num)
  for hlsc in hls_colors:
    _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
    r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
    rgb_colors.append([r, g, b])

  return rgb_colors


# classes = ('pedestrian', 'people','bicycle','car','van','truck','tricycle','awning-tricycle',
#            'bus','motor')
'''
车：car  圆亭子：round_pavilion 亭子：pavilion 水文站房：hydrological_station 坦克：tank 
直升机：helicopter  雨伞：umbralla 橡皮艇：rubber_boat  救生圈：life_buoy
'''
classes = ['round_pavilion', 'car', 'hydrological_station', 'tank', 'helicopter', 'umbralla', 'pavilion', 'rubber_boat', 'life_buoy']
colors = ncolors(len(classes))

#把下面的路径改为自己的路径即可
file_path_img = 'D:\Data\训练数据\圆亭子\label\images'
file_path_xml = 'D:\Data\训练数据\圆亭子\label\labels'
save_file_path = 'D:\Data\训练数据\圆亭子\label\\vis'

pathDir = os.listdir(file_path_xml)
for idx in range(len(pathDir)):
    filename = pathDir[idx]#xml文件名
    tree = xmlET.parse(os.path.join(file_path_xml, filename))#解析xml
    objs = tree.findall('object')

    num_objs = len(objs)
    boxes = np.zeros((num_objs, 5), dtype=np.uint16)

    for ix, obj in enumerate(objs):
        print(obj)
        
        bbox = obj.find('bndbox')
        if bbox == 'None':
          
          continue
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)

        cla = obj.find('name').text
        label = classes.index(cla)
        boxes[ix, 0:4] = [x1, y1, x2, y2]
        boxes[ix, 4] = label
        # boxes[ix, 4] = cla

    image_name = os.path.splitext(filename)[0]
    image_suffix = os.path.splitext(filename)[1]
    # print(image_suffix)
    print(filename)
    if os.path.exists(os.path.join(file_path_img, image_name + '.JPG')):
      img = Image.open(os.path.join(file_path_img, image_name + '.JPG'))
      draw = ImageDraw.Draw(img)
      for ix in range(len(boxes)):
          xmin = int(boxes[ix, 0])
          ymin = int(boxes[ix, 1])
          xmax = int(boxes[ix, 2])
          ymax = int(boxes[ix, 3])
          draw.rectangle([xmin, ymin, xmax, ymax], outline=(colors[boxes[ix, 4]][2], colors[boxes[ix, 4]][1], colors[boxes[ix, 4]][0]), width=3)
          draw.text([xmin, ymin-8], classes[boxes[ix, 4]], (255, 0, 0), stroke_width=3)
      img.save(os.path.join(save_file_path, image_name + '.png'))
    else:
      img = Image.open(os.path.join(file_path_img, image_name + '.jpg'))
      draw = ImageDraw.Draw(img)
      for ix in range(len(boxes)):
          xmin = int(boxes[ix, 0])
          ymin = int(boxes[ix, 1])
          xmax = int(boxes[ix, 2])
          ymax = int(boxes[ix, 3])
          draw.rectangle([xmin, ymin, xmax, ymax], outline=(colors[boxes[ix, 4]][2], colors[boxes[ix, 4]][1], colors[boxes[ix, 4]][0]), width=3)
          draw.text([xmin, ymin-8], classes[boxes[ix, 4]], (255, 0, 0), stroke_width=3)
      img.save(os.path.join(save_file_path, image_name + '.png'))




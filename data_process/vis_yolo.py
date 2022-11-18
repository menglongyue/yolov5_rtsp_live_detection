# from pathlib import Path
# from alphabet import alphabet
# import pandas as pd
# import cv2
# import matplotlib.pyplot as plt

# alphabet = alphabet.split("\n")

# label_root = Path("labels/test")
# image_root = Path("images/test")


# def paint(label_file,image_file):
#     #读取标签
#     df = pd.read_csv(label_file,sep=" ",names=['id','center-x','center-y','w','h'])
#     df['id'] = df['id'].apply(lambda x:alphabet[x])
#     df = df.sort_values(by='center-x')
#     #读取图片
#     img = cv2.imread(str(image_root/image_file))
#     h,w = img.shape[:2]
    
#     df[['center-x','w']] = df[['center-x','w']].apply(lambda x:x*w)
#     df[['center-y','h']] = df[['center-y','h']].apply(lambda x:x*h)
    
#     df['x1'] = df['center-x']-df['w']/2
#     df['x2'] = df['center-x']+df['w']/2
#     df['y1'] = df['center-y']-df['h']/2
#     df['y2'] = df['center-y']+df['h']/2

#     df[['x1','x2','y1','y2']] = df[['x1','x2','y1','y2']].astype('int')
    
#     points = zip(df['x1'],df['y1'],df['x2'],df['y2'])
#     for point in points:
#         img = cv2.rectangle(img,point[:2],point[2:],color=(0, 255, 0),thickness=1)
#     plt.imshow(img)
#     plt.show()

# for label_file in label_root.iterdir():
#     image_file = label_file.name.replace(".txt",".jpg")
#     paint(label_file,image_file)
#     break


import cv2 as cv
import os
import colorsys
import random
from tqdm import tqdm

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

def convert(bbox,shape):
    x1 = int((bbox[0] - bbox[2] / 2.0) * shape[1])
    y1 = int((bbox[1] - bbox[3] / 2.0) * shape[0])
    x2 = int((bbox[0] + bbox[2] / 2.0) * shape[1])
    y2 = int((bbox[1] + bbox[3] / 2.0) * shape[0])
    return (x1,y1,x2,y2)




n = 9 # 类别数
# 获取n种区分度较大的rgb值
colors = ncolors(n)
root = 'D:\Data\jiujiang_train_data\\train\\temp'
classes = ['round_pavilion', 'car', 'hydrological_station','tank', 'helicopter', 'umbralla', 'pavilion', 'rubber_boat', 'life_buoy']
images_dir = 'images' # 图片目录
labels_dir = 'yolo_labels' # label目录
output_dir = 'vis_yolo' # 输出图片目录
images_list = os.listdir(root + os.sep + images_dir) # 获取图片名列表
if not os.path.exists(root + os.sep + output_dir):
    os.makedirs(root + os.sep + output_dir)
for img_id in tqdm(images_list):
    img = cv.imread(root + os.sep + images_dir + os.sep + img_id) 
    # print(img.shape)

    # 判断后缀是为了排除隐藏文件.ipynb_checkpoints
    if img_id.endswith('jpg') or img_id.endswith('JPG'):
      shape = img.shape[0:2]
      # print(shape)
      if img_id.endswith('.jpg'):
        txt_id = img_id.replace('jpg', 'txt') 
      elif img_id.endswith('.JPG'):
        txt_id = img_id.replace('JPG', 'txt')
      else:
        print(txt_id)
      with open(root + os.sep + labels_dir + os.sep + txt_id) as r:
          lines = r.readlines()
          for line in lines:
              line = [float(i) for i in line.split(' ')] # 按空格划分并转换float类型
              label = int(line[0]) #获取类别信息
              bbox = line[1:] # 获取box信息
              (x1,y1,x2,y2) = convert(bbox,shape)
              cv.rectangle(img, (x1, y1), (x2, y2), (colors[label][2], colors[label][1], colors[label][0]), 1)
              cv.putText(img, classes[label], (x1, y1-6), cv.FONT_HERSHEY_SIMPLEX, 0.75, color=(colors[label][2], colors[label][1], colors[label][0]), thickness=1)
              # cv.waitKey(0)
      cv.imwrite(root + os.sep + output_dir + os.sep + img_id, img)




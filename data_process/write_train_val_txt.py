from pathlib import Path
import os
import random

img_data_path1 = Path('D:\Data\jiujiang_train_data\\train_dataset\car\images')
img_data_path2 = Path('D:\Data\jiujiang_train_data\\train_dataset\car_life_buoy_umbralla\images')
img_data_path3 = Path('D:\Data\jiujiang_train_data\\train_dataset\dynamic_guanfang\images')
img_data_path4 = Path('D:\Data\jiujiang_train_data\\train_dataset\hydrological_station\images')
img_data_path5 = Path('D:\Data\jiujiang_train_data\\train_dataset\life_buoy\images')
img_data_path6 = Path('D:\Data\jiujiang_train_data\\train_dataset\pavilion\images')
img_data_path7 = Path('D:\Data\jiujiang_train_data\\train_dataset\\robber_boat\images')
img_data_path8 = Path('D:\Data\jiujiang_train_data\\train_dataset\\round_pavilion\images')
img_data_path9 = Path('D:\Data\jiujiang_train_data\\train_dataset\\tanks_helicopters\images')
img_data_path10 = Path('D:\Data\jiujiang_train_data\\train_dataset\\umbralla\images')

img_path_list = [img_data_path1, img_data_path2, img_data_path3, img_data_path4, img_data_path5,
                 img_data_path6, img_data_path7, img_data_path8, img_data_path9, img_data_path10]
img_file_list = []
for img_path in img_path_list:
    img_file_list += [f for f in img_path.iterdir() if f.is_file]

# print(img_file_list)
num_images = len(img_file_list)

random.shuffle(img_file_list)
random.shuffle(img_file_list)
random.shuffle(img_file_list)
random.shuffle(img_file_list)
random.shuffle(img_file_list)

num_train = int(num_images * 0.8)

with open('train.txt', 'a', encoding='utf-8') as f1:
    for img in img_file_list[:num_train]:
        f1.write(str(img) + '\n')

with open('val.txt', 'a', encoding='utf-8') as f2:
    for img in img_file_list[num_train:]:
        f2.write(str(img) + '\n')



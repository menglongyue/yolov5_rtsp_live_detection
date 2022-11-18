from pathlib import Path
import os
import shutil

img_root = Path('D:\Data\训练数据\雨伞\san_crop')
img_path_dst_root = 'D:\Data\训练数据\雨伞\san_crop2\\'

img_list = [f for f in img_root.iterdir() if f.is_file]

for img_path in img_list:
    if str(img_path).split('.')[0][-1] == 'W':
        img_name = img_path.name
        img_ori = str(img_path)
        # img_name_= str(img_name).split('.')[0] + '_yuantingzi2' + '.jpg'
        img_path_dst = img_path_dst_root + str(img_name)
        shutil.copy(str(img_path), img_path_dst)
    elif str(img_path).split('.')[0][-1] == 'Z':
        img_name = img_path.name
        img_ori = str(img_path)
        # img_name_= str(img_name).split('.')[0] + '_yuantingzi2' + '.jpg'
        img_path_dst = img_path_dst_root + str(img_name)
        shutil.copy(str(img_path), img_path_dst)

    # if str(img_path).split('.')[0][-1] == 'Z':
    #     img_name = img_path.name
    #     img_ori = str(img_path)
    #     # img_name_= str(img_name).split('.')[0] + '_yuantingzi2' + '.jpg'
    #     img_path_dst = img_path_dst_root + str(img_name)
    #     shutil.copy(str(img_path), img_path_dst)
    

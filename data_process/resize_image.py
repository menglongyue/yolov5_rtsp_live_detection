from pathlib import Path
import os
import shutil
import cv2

img_root = Path('D:\Data\jiujiang_train_data\\train\\temp\images')
img_path_dst_root = 'D:\Data\jiujiang_train_data\\train\\temp\images_resize'

img_list = [f for f in img_root.iterdir() if f.is_file]

for img_path in img_list:
    img_name = img_path.name
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (960, 720))
    cv2.imwrite(img_path_dst_root + os.sep + str(img_name), img)

    # img_name = img_path.name
    # img_ori = str(img_path)
    # img_name_= str(img_name).split('.')[0] + '_yuantingzi2' + '.jpg'
    # img_path_dst = img_path_dst_root + img_name_
    # shutil.copy(str(img_path), img_path_dst)

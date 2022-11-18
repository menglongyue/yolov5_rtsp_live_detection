from pathlib import Path
import os
import shutil

# img_root = Path('D:\Data\训练数据\亭子3\标记')
# img_path_dst_root = 'D:\Data\训练数据\亭子_2_img\\'

# img_list = [f for f in img_root.iterdir() if f.is_file]

# for img_path in img_list:
#     img_name = img_path.name
#     img_ori = str(img_path)
#     img_name_= str(img_name).split('.')[0] + 'tingzi3' + '.jpg'
#     img_path_dst = img_path_dst_root + img_name_
#     shutil.copy(str(img_path), img_path_dst)


# file_root = Path('D:\Data\训练数据\object_jiujiang\输出')
# file_list = Path.rglob(file_root, pattern='*')                                                                
# num = 0

# for i in file_list:
#     if i.is_file:

#         if str(i.parent).split(os.sep)[-1] == 'Image':
#             if str(i.parent.parent.parent).split(os.sep)[-1] == '救生圈':
#                 if str(i).endswith('.jpg'):
#                     new_name = str(i).split('.')[0] + '_jiushengquan.jpg'
#                     os.rename(str(i), new_name)
#             elif str(i.parent.parent.parent).split(os.sep)[-1] == '特殊车辆1':
#                 if str(i).endswith('.jpg'):
#                     new_name = str(i).split('.')[0] + '_teshucheliang1.jpg'
#                     os.rename(str(i), new_name)
#             elif str(i.parent.parent.parent).split(os.sep)[-1] == '特殊车辆2':
#                 if str(i).endswith('.jpg'):
#                     new_name = str(i).split('.')[0] + '_teshucheliang2.jpg'
#                     os.rename(str(i), new_name)
#             elif str(i.parent.parent.parent).split(os.sep)[-1] == '橡皮艇':
#                 if str(i).endswith('.jpg'):
#                     new_name = str(i).split('.')[0] + '_xiangpiting.jpg'
#                     os.rename(str(i), new_name)
#             elif str(i.parent.parent.parent).split(os.sep)[-1] == '雨伞':
#                 if str(i).endswith('.jpg'):
#                     new_name = str(i).split('.')[0] + '_yusan.jpg'
#                     os.rename(str(i), new_name)
#             else:
#                 print(str(i))
#         elif str(i.parent).split(os.sep)[-1] == 'Label':
#             if str(i.parent.parent.parent).split(os.sep)[-1] == '救生圈':
#                 if str(i).endswith('.xml'):
#                     new_name = str(i).split('.')[0] + '_jiushengquan.xml'
#                     os.rename(str(i), new_name)
#             elif str(i.parent.parent.parent).split(os.sep)[-1] == '特殊车辆1':
#                 if str(i).endswith('.xml'):
#                     new_name = str(i).split('.')[0] + '_teshucheliang1.xml'
#                     os.rename(str(i), new_name)
#             elif str(i.parent.parent.parent).split(os.sep)[-1] == '特殊车辆2':
#                 if str(i).endswith('.xml'):
#                     new_name = str(i).split('.')[0] + '_teshucheliang2.xml'
#                     os.rename(str(i), new_name)
#             elif str(i.parent.parent.parent).split(os.sep)[-1] == '橡皮艇':
#                 if str(i).endswith('.xml'):
#                     new_name = str(i).split('.')[0] + '_xiangpiting.xml'
#                     os.rename(str(i), new_name)
#             elif str(i.parent.parent.parent).split(os.sep)[-1] == '雨伞':
#                 if str(i).endswith('.xml'):
#                     new_name = str(i).split('.')[0] + '_yusan.xml'
#                     os.rename(str(i), new_name)
#             else:
#                 print(str(i))


import os
 
class BatchRename():
 
    def rename(self):
        path = "D:\Data\jiujiang_train_data\\train\car"
        filelist = os.listdir(path)
        total_num = len(filelist)
        i = 0
        for item in filelist:
            if item.endswith('.png'):
                src = os.path.join(os.path.abspath(path), item)
                dst = os.path.join(os.path.abspath(path), ''+str(i)+'.png')
                try:
                    os.rename(src, dst)
                    i += 1
                except:
                    continue
        print('total %d to rename & converted %d png'%(total_num, i))
 
if __name__=='__main__':
    demo = BatchRename()
    demo.rename()

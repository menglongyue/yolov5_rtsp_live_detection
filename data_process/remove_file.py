import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as xmlET

''' 
Path模块中的glob： 可以用 img_path.rglob(pattern='*') 或者 Path.rglob(img_path, pattern='*)
rglob('*')  返回：当前目录，及所有子目录中的 所有文件和文件夹
rglob('**)  返回：当前目录，及其下所有子目录中的 所有文件夹
glob(r'*')  返回：当前目录中的所有文件和文件夹
glob(r'**') 返回：当前目录，及其下所有子目录中的 所有文件夹

glob模块中的glob：
# import glob2 as gb
# 1、gb.glob("*") 与 gb.iglob("*")
gb.glob("*")  # 返回：当前目录下的所有文件、文件夹；列表
gb.iglob("*")  # 返回：当前目录下的所有文件、文件夹；map


# 2、gb.glob("**") 与 gb.iglob("**")
gb.glob("**")    # 返回：当前目录、及子目录下的所有文件、文件夹；列表
gb.iglob("**")  # 返回：当前目录、及子目录下的所有文件、文件夹；map

总结：
1、Path模块，返回的是 generator
   glob模块，返回的是 文件/文件夹名

2、1) glob.glob(*) = glob.iglob(*)：返回当前目录下的文件和文件夹
   2) glob.glob(**) = glob.iglob(**)：返回当前目录、及子目录下的文件和文件夹
   3) Path.glob(**) = Path.rglob(**) ：返回当前目录、及子目录下的文件夹
   4) Path.glob(*)：返回当前目录下的文件和文件夹
   5)  Path.rglob(*)：返回当前目录、及子目录下的文件和文件夹
'''


# file_root = Path('D:\Data\训练数据\object_jiujiang\输出')
# file_root = Path('D:\Data\jiujiang_train_data\\train\car')
# file_list = Path.rglob(file_root, pattern='*')    
# file_list_ = [f for f in file_list if f.is_file]                                                            
# num = 0

# #删除Image文件下的xml
# # for i in file_list:
# #     if i.is_file:
# #         if str(i.parent).split(os.sep)[-1] == 'Image':
# #             if str(i).endswith('.xml'):
# #                 os.remove(str(i))

# dst_path = 'D:\Data\jiujiang2\\traing_data2\W\\to_delete'
# #根据xml中的bbox，删除没有目标的xml文件及对应的image
# for i in file_list:
#     if i.is_file:
#         if str(i.parent).split(os.sep)[-1] == 'labels':
#             if str(i).endswith('.xml'):
#                 xml_name = i.name
#                 img_stem = i.stem
#                 img_name = str(img_stem) + '.JPG'
#                 dst_images_root = 'D:\Data\jiujiang2\\traing_data2\W\\to_delete' + os.sep + str(xml_name)
#                 dst_labels_root = 'D:\Data\jiujiang2\\traing_data2\W\\to_delete' + os.sep + str(img_name)
#                 img_path = 'D:\Data\jiujiang2\\traing_data2\W\images' + os.sep + img_name
#                 tree = xmlET.parse(str(i))#解析xml
#                 objs = tree.findall('object')
#                 class_name = []
#                 for obj in objs:  #找出person节点
#                     child = obj.getchildren()[0]
#                     # print(child.text)
#                     class_name.append(child.text)
#                     # if child.text == 'rubber_boat': # 将VerticalCrack改为verticalcrack
                
#                 if 'life_buoy' not in class_name and 'umbralla' not in class_name:
#                 # if 'rubber_boat' in class_name:
#                     shutil.move(str(i), dst_path)
#                     shutil.move(img_path, dst_path)
                # if len(objs) == 0:
                #     img_name = str(i.stem) + '.JPG'
                #     img_path = str(i.parent.parent) + os.sep + 'images' + os.sep + img_name 
                #     os.remove(str(i))
                #     os.remove(img_path)


# for i in file_list:
#     if i.is_file:
#         if str(i).endswith('.xml'):
#             tree = xmlET.parse(str(i))#解析xml
#             objs = tree.findall('object')
#             if len(objs) == 0:
#                 img_name = str(i.stem) + '.jpg'
#                 if os.path.exists(str(i.parent.parent) + os.sep + 'images' + os.sep + img_name):
#                     img_path = str(i.parent.parent) + os.sep + 'images' + os.sep + img_name
#                     os.remove(str(i))
#                     os.remove(img_path)
#                 else:
#                     img_name = str(i.stem) + '.JPG'
#                     img_path = str(i.parent.parent) + os.sep + 'images' + os.sep + img_name
#                     os.remove(str(i))
#                     os.remove(img_path)

# 将不同文件夹下的images，labels移动到同一个images，labels下：
# dst_images = 'D:\Data\训练数据\水文站房\label\images'
# dst_labels = 'D:\Data\训练数据\水文站房\label\labels'
# for i in file_list:
#     if i.is_file:
#         if str(i.parent).split(os.sep)[-1] == 'Label':
#             if str(i).endswith('.xml'):
#                 shutil.copy(str(i), dst_labels)
#         if str(i.parent).split(os.sep)[-1] == 'Image':
#             if str(i).endswith('.jpg'):
#                 shutil.copy(str(i), dst_images)


# dst_images = 'D:\Data\jiujiang2\\traing_data2\W\images'
# dst_labels = 'D:\Data\jiujiang2\\traing_data2\W\labels'
# for i in file_list:
#     if i.is_file:
#         if str(i).endswith('.xml'):
#             shutil.move(str(i), dst_labels)
#         elif str(i).endswith('.jpg') or str(i).endswith('.JPG'):
#             shutil.move(str(i), dst_images)
#         else:
#             print(i)
file_root = Path('D:\Data\jiujiang_train_data\\train\\umbralla')
# file_list = Path.rglob(file_root, pattern='*')    
file_list = [f for f in file_root.iterdir() if f.is_file]    
dst_images = 'D:\Data\jiujiang_train_data\\train\\umbralla\images'
dst_labels = 'D:\Data\jiujiang_train_data\\train\\umbralla\labels'
for i in file_list:
    if str(i).endswith('.txt'):
        shutil.move(str(i), dst_labels)
    elif str(i).endswith('.jpg') or str(i).endswith('.JPG'):
        shutil.move(str(i), dst_images)
    else:
        print(i)

# dst_images = 'D:\Data\jiujiang2\myself2\labels'
# # dst_labels = 'D:\Data\训练数据\圆亭子\label\labels'
# for i in file_list:
#     if i.is_file:
        
#         img_stem = i.stem
#         xml_name = str(img_stem) + '.xml'
#         xml_path = 'D:\Data\jiujiang2\myself\labels' + os.sep + xml_name
#         if os.path.exists(xml_path):
        
#             shutil.move(xml_path, dst_images)
        # elif str(i).endswith('.jpg') or str(i).endswith('.JPG'):
        #     shutil.move(str(i), dst_images)
        # else:
        #     print(i)

#根据xml中的bbox，删除没有目标的xml文件及对应的image
# dst_images_path = 'D:\Data\jiujiang_train_data\\train\dynamic_guanfang\\robber_boat\images'
# dst_labels_path = 'D:\Data\jiujiang_train_data\\train\dynamic_guanfang\\robber_boat\labels'
# for i in file_list:
#     if i.is_file:
#         if str(i).endswith('.xml'):

#             xml_name = i.name
#             img_stem = i.stem
#             img_name = str(img_stem) + '.jpg'
#             dst_images_root = 'D:\Data\jiujiang_train_data\\train\dynamic_guanfang\\robber_boat\images' + os.sep + str(xml_name)
#             dst_labels_root = 'D:\Data\jiujiang_train_data\\train\dynamic_guanfang\\robber_boat\labels' + os.sep + str(img_name)
            
#             img_path = 'D:\Data\jiujiang_train_data\\train\dynamic_guanfang\images' + os.sep + img_name
#             tree = xmlET.parse(str(i))#解析xml
#             objs = tree.findall('object')
#             class_name = []
#             for obj in objs:  #找出person节点
#                 child = obj.getchildren()[0]
#                 print(child.text)
#                 class_name.append(child.text)
#                 # if child.text == 'rubber_boat': # 将VerticalCrack改为verticalcrack
#             if 'rubber_boat' in class_name:
#                 shutil.copy(str(i), dst_labels_path)
#                 shutil.copy(img_path, dst_images_path)
                    

# dst_images = 'D:\Data\jiujiang_train_data\\train\dynamic_guanfang\\robber_boat\\robber_aug\labels_'
# # dst_labels = 'D:\Data\训练数据\圆亭子\label\labels'
# for i in file_list:
#     if i.is_file:
        
#         img_stem = i.stem
#         xml_name = str(img_stem) + '.txt'
#         xml_path = 'D:\Data\jiujiang_train_data\\train\dynamic_guanfang\\robber_boat\\robber_aug\\new_label' + os.sep + xml_name
#         if os.path.exists(xml_path):
#             print(i)
        
#             shutil.move(xml_path, dst_images)
#         # elif str(i).endswith('.jpg') or str(i).endswith('.JPG'):
#         #     shutil.move(str(i), dst_images)
#         else:
#             # print(i)
#             pass
# dst_path_root = 'D:\Data\jiujiang_train_data\\all_data'
# for i in file_list_:
#     # print(str(i))
#     # print(str(i))
#     # print(i.is_file)
#     if str(i).endswith('.jpg') or str(i).endswith('.txt'):
#         print(str(i))
#         img_stem = i.stem
#         img_suffix = i.suffix
#         img_plus = str(i).split(os.sep)[-2]
#         img_new_name = str(img_stem) + '_' + str(img_plus) + img_suffix
#         dst_path = dst_path_root + os.sep + img_new_name
#         print(dst_path)
#         shutil.copy(str(i), dst_path)
#     else:
#         pass
#         # elif str(i).endswith('.jpg') or str(i).endswith('.JPG'):
        #     shutil.move(str(i), dst_images)
        # else:
        #     # print(i)
        #     pass
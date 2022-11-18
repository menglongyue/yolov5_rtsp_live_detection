import xml.etree.ElementTree as xmlET
from pathlib import Path 
import os 

'''类别名：
车：car  圆亭子：round_pavilion 亭子：pavilion 水文站房：hydrological_station 坦克：tank 
直升机：helicopter  雨伞：umbralla 橡皮艇：rubber_boat  救生圈：life_buoy
'''

# file_root = Path('D:\Data\训练数据\object_jiujiang\输出')
# file_list = Path.rglob(file_root, pattern='*')                                                                
# num = 0
# for i in file_list:
#     if i.is_file:
#         if str(i.parent).split(os.sep)[-1] == 'Label':
#             if str(i).endswith('.xml'):
#                 print(str(i))
#                 tree = xmlET.parse(str(i))#解析xml
#                 objs = tree.findall('object')
#                 # p= per.findall('/object')
#                 for obj in objs:  #找出person节点
#                     child = obj.getchildren()[0] #找出person节点的子节点
#                     if child.text == '车': # 将VerticalCrack改为verticalcrack
#                         child.text = 'car'
#                     elif child.text == '救生圈':
#                         child.text = 'life_buoy'
#                     elif child.text == '橡皮艇':
#                         child.text = 'rubber_boat'
#                     elif child.text == '雨伞':
#                         child.text = 'umbralla'
#                     elif child.text == '亭子2':
#                         child.text = 'pavilion'
#                     else:
#                         print(child.text)
#                 tree.write(str(i))


# file_root = Path('D:\Data\训练数据\水文站房\label\labels')
# file_list = Path.rglob(file_root, pattern='*')                                                                
# num = 0
# for i in file_list:
#     if i.is_file:
#         if str(i.parent).split(os.sep)[-1] == 'Label':
#             if str(i).endswith('.xml'):
#                 print(str(i))
#                 tree = xmlET.parse(str(i))#解析xml
#                 objs = tree.findall('object')
#                 # p= per.findall('/object')
#                 for obj in objs:  #找出person节点
#                     # child = obj.getchildren()[0] #找出person节点的子节点
#                     child = obj.getchildren()[1]
#                     print(child.text)
#                     if child.text == '车': # 将VerticalCrack改为verticalcrack
#                         child.text = 'car'
#                     elif child.text == '救生圈':
#                         child.text = 'life_buoy'
#                     elif child.text == '橡皮艇':
#                         child.text = 'rubber_boat'
#                     elif child.text == '雨伞':
#                         child.text = 'umbralla'
#                     elif child.text == '亭子2':
#                         child.text = 'pavilion'
#                     elif child.text == '3':
#                         child.text = 'hydrological_station'
#                     else: 
#                         print(child.text)
#                 tree.write(str(i))
                
file_root = Path('D:\Data\jiujiang2\\traing_data2\\umbralla\labels')
file_list = Path.rglob(file_root, pattern='*')                                                                
num = 0
for i in file_list:
    if i.is_file:
        if str(i).endswith('.xml'):
            # print(str(i))

            tree = xmlET.parse(str(i))#解析xml
            objs = tree.findall('object')
            if len(objs) == 0:
                print('No Object found!!!')
            # p= per.findall('/object')
            for obj in objs:  #找出person节点
                # child = obj.getchildren()[0] #找出person节点的子节点
                if str(i.name).startswith('D'):
                    child = obj.getchildren()[0]
                elif str(i.name).startswith('i'):
                    print(str(i))
                    child = obj.getchildren()[1]
                else:
                    print('第3个name:', child.text)
                print(child.text)
                if child.text == '车': # 将VerticalCrack改为verticalcrack
                    child.text = 'car'
                elif child.text == '救生圈':
                    child.text = 'life_buoy'
                elif child.text == '橡皮艇':
                    child.text = 'rubber_boat'
                elif child.text == '雨伞':
                    child.text = 'umbralla'
                elif child.text == '亭子2':
                    child.text = 'pavilion'
                elif child.text == '3':
                    child.text = 'hydrological_station'
                elif child.text == '4':
                    child.text = 'tank'
                elif child.text == '5':
                    child.text = 'helicopter'
                elif child.text == '7':
                    child.text = 'pavilion'
                elif child.text == '2':
                    child.text = 'car'
                elif child.text == '6':
                    child.text = 'umbralla'
                elif child.text == '1':
                    child.text = 'round_pavilion'
                elif child.text == '8':
                    child.text = 'rubber_boat'
                elif child.text == '9':
                    child.text = 'life_buoy'
                elif child.text == '亭子':
                    child.text = 'round_pavilion'
                elif child.text == 'tingzi1':
                    child.text = 'round_pavilion'
                elif child.text == 'umbrella':
                    child.text = 'umbralla'
                
                # elif child.text == '&#20141;&#23376;':
                #     child.text = 'round_pavilion'
                else:
                    print(child.text)
            tree.write(str(i))
            

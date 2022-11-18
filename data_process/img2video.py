import cv2
import os
from pathlib import Path

def createVideo(filePath):
    ## 创建视频合成格式
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')  # 指定输出视频的格式
    ## VideoWriter(fileName,fourcc,fps,frameSize[:,iscolor])创建VideoWriter对象

    ## 此处的图像大小必须与原始图像的大小保持一致，否则会报错
    out = cv2.VideoWriter('D:\Code\yolov5\\test_video\\test_car.mp4', fourcc, 15, (960,720)) # 此处的frame.shape = (480, 220, 3)

    ## filePath即为图片保存路径
    fileNames = os.listdir(filePath)
    file_list = [f for f in Path(filePath).iterdir() if f.is_file]
    # fileNames = sorted(fileNames)
    file_list = sorted(file_list, key=lambda x:str(x.stem)[-4:])
    # print(fileNames)
    # print(fileNames)
    for file in file_list:
        ## 错误：cv2.error: OpenCV(4.5.5) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\color.cpp:182: error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'
        ## 代码中出现任何问题基本上都会报此错误，检查代码即可，不必过分纠结于错误提示
        # frame = cv2.imread('./' + str(file))
        frame = cv2.imread(str(file))
        # print(frame.shape)
        out.write(frame)
        cv2.imshow('video',frame)
        c = cv2.waitKey(1)
        if c == 27:
            break

    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    file_path = 'D:\Data\Video\car'
    createVideo(file_path)

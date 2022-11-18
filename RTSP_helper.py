import threading
import cv2
import sys


class RTSCapture(cv2.VideoCapture):
    '''Real time streaming capture
    这个类必须使用 RTSCapture.create 方法创建， 请不要直接实例化
    '''
    _cur_frame = None
    _reading = False
    # 用于识别实时流
    schemes = ['rtsp://', 'rtmp://', '*.mp4'] 
    
    @staticmethod
    def create(url, *schemes):
        '''实例化和初始化
        rtscap = RTSCapture.create('rtsp://example.com/live/1')
        or
        rtscap = RTSCapture.create('http://example.com/live/1.d3k1', 'http://')
        '''
        rtscap = RTSCapture(url)
        rtscap.frame_receiver = threading.Thread(target=rtscap.recv_frame, daemon=True) # daemon：守护进程，主程序结束时，子程序自动结束，不需要干涉
        rtscap.schemes.extend(schemes)
        if isinstance(url, str) and url.startswith(tuple(rtscap.schemes)):
            rtscap._reading = True
        elif isinstance(url, int):
            # 可能是本机器设备
            print('may be local camera!')
        return rtscap
    
    
    def isStarted(self):
        '''
        该方法替代VideoCapture.isOpened()
        '''
        ok = self.isOpened()
        if ok and self._reading:
            ok = self.frame_receiver.is_alive()
        return ok
    
    def recv_frame(self):
        '''
        子线程读取最新视频帧方法
        '''
        while self._reading and self.isOpened():
            ok, frame = self.read()
            if not ok:
                break
            self._cur_frame = frame
        self._reading = False
        
    def read2(self):
        '''
        读取最新视频帧，返回结果格式与VideoCapture.read()一样
        '''
        frame  = self._cur_frame
        self._cur_frame = None
        return frame is not None, frame
    
    def start_read(self):
        '''
        启动子线程读取视频帧
        '''
        self.frame_receiver.start()
        self.read_latest_frame = self.read2 if self._reading else self.read
        
    def stop_read(self):
        '''
        退出子线程方法
        '''
        self._reading = False
        if self.frame_receiver.is_alive():
            self.frame_receiver.join() # 主程序运行完等待子线程运行完再一起结束
            

if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print('usage:')
    #     print('python3 opencv_read_utils.py "rtsp://xxx"')
    #     sys.exit()
        
    # rtscap = RTSCapture.create(sys.argv[1])
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v") # XVID
    out = cv2.VideoWriter('output_shuiwenzhan.mp4',fourcc, 30.0, (1280,720))
    rtscap = RTSCapture.create('rtsp://18770748650:19990728c@172.20.10.4:8554/streaming/live/1')
    # rtscap = RTSCapture.create('D:\Code\yolov5\data\\test_video.mp4')
    rtscap.start_read() # 启动子线程，改变read_latest_frame的指向
    # fps = 30
    # save_path = 'test.mp4'
    # width = 
    # vid_writer = cv2.VideoWriter(
    #         save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    #     )
    i = 0
    while rtscap.isStarted():
        i += 1
        ok, frame = rtscap.read_latest_frame() # read_latest_frame替代read()
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        if not ok:
            continue
        # if i % 4 == 0:
        #     cv2.imwrite('D:\Data\yuantingzi2\img_{}.jpg'.format(str(i)), frame)
        cv2.imwrite('D:\Data\jiujiang2\\2\img_{}.jpg'.format(str(i)), frame)
        
        # 帧处理代码逻辑在此处
        print(frame.shape)
        out.write(frame)
        # cv2.imwrite('D:\Data\jiujiang\\test\\{i}_img.jpg'.format(str(i)), frame)
        cv2.imshow('cam', frame)

        
    
    rtscap.stop_read()
    rtscap.release()
    out.release()
    cv2.destroyAllWindows()
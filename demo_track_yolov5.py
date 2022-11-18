from loguru import logger
import numpy as np
from RTSP_helper import *
import cv2
import torch

from utils.visualize import plot_tracking
from tracker.byte_tracker import BYTETracker
from tracking_utils.timer import Timer
import argparse
import os
import time
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, check_img_size, scale_coords
from utils.torch_utils import select_device
from utils.boxes import postprocess

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
# device = "cpu"


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "--demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    # rtsp://15270095695:Zjb15270095695@192.168.185.72:8554/streaming/live/1
    parser.add_argument(
        "--path", default="data\\test_video.mp4", help="path to images or video"
        # "--path", default="./videos/16h-17h.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        type=bool,
        default=True,
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="test",
        help="save name for results txt/video",
    )

    parser.add_argument("-c", "--ckpt", default="weights/yolov5s6_visdrone.pt", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="0",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--num_classes", type=int, default=11, help="number of classes")
    parser.add_argument("--conf_thres", default=0.3, type=float, help="test conf")
    parser.add_argument("--iou_thres", default=0.5, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=(1280, 1280), type=tuple, help="test image size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=40, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    return parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

class Predictor(object):
    def __init__(
        self,
        model,
        num_classes, 
        conf_thresh, 
        iou_thresh, 
        test_size,
        device="cpu",
        fp16=False
    ):
        self.model = model
        self.num_classes = num_classes # 1
        self.conf_thre = conf_thresh
        self.iou_thre = iou_thresh
        self.test_size = test_size
        self.device = device
        self.fp16 = fp16
        

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img.copy()

        stride= self.model.stride
    # # imgsz = img0.shape[0]
        imgsz = check_img_size(self.test_size, s=stride)  # check image size

        # stride用于提前处理图片，使之可以被图片下采样倍数整除，如32，  auto决定是否保持宽高比缩放，如果为False，则填充0至imgsz
        auto = True
        img = letterbox(img, imgsz, stride=stride, auto=auto)[0]
        print('img shape: ', img.shape)
        # img = cv2.resize(img, (1280, 1280), interpolation=cv2.INTER_LINEAR)
        # print('letterbox:', img.shape)
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
       
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        # print(img)
        
        
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        img_info['deal_img'] = img

        with torch.no_grad():
            timer.tic()
            # print('image shape:', img.size())
            # print(img)
            outputs = self.model(img)
            # if self.decoder is not None:
            #     outputs = self.decoder(outputs, dtype=outputs.type())
            print('before nms:', outputs.size())
            print('model outputs: ', outputs)
            outputs = non_max_suppression(outputs, self.conf_thre, self.iou_thre, agnostic=True, max_det=1000)
            
            # outputs = postprocess(outputs, self.num_classes, self.conf_thre, self.iou_thre)
            # outputs = non_max_suppression(outputs, 0.1, 0.45, classes=[0], max_det=1000)
            # print(outputs[0].shape)
            # print('after nms:', outputs[0].size())
            timer.toc()
           
        return outputs, img_info

def save_outputs(outputs, folder, save_name):
    sn = save_name.split('/')[-1].replace('.jpg', '.txt')
    # if not os.path.exists('yolov5_outputs'):
    #     os.mkdir('yolov5_outputs')

    sn = os.path.join('runs', folder, sn)
    # if not os.path.exists(os.path.join('yolov5_outputs', folder)):
    #     os.mkdir(os.path.join('yolov5_outputs', folder))
    # open or creat new file if dont exist
    with open(sn, 'w') as f:
        if outputs[0] is not None:
                for i in range(len(outputs[0])):
                    op = outputs[0][i].tolist()
                    for j in op:
                        f.write(str(j) + ' ')
                    f.write('\n')
            


def image_demo(predictor, vis_folder, path, current_time, save_result, save_name, test_size):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    results = []
    for frame_id, image_name in enumerate(files, 1):
        
        outputs, img_info = predictor.inference(image_name, timer)
        save_outputs(outputs, save_name, image_name)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], test_size)
            # print('height:', img_info['height'], 'width:', img_info['width'])
            # print('test size:', exp.test_size)
            print('online_targets:', len(online_targets))
            # print(online_targets)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        else:
            timer.toc()
            online_im = img_info['raw_img']

        #result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, save_name
            )
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            print("Save tracked image to {}".format(save_file_name))
            cv2.imwrite(save_file_name, online_im)
        
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    
    if save_result:
        res_file = os.path.join(vis_folder, os.path.basename(save_name + '.txt'))
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
    

def imageflow_demo(predictor, vis_folder, current_time, args):
    
    rtscap = RTSCapture.create(args.path)
    rtscap.start_read() # 启动子线程，改变read_latest_frame的指向
    while rtscap.isStarted():
        ok, frame = rtscap.read_latest_frame() # read_latest_frame替代read()
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        if not ok:
            continue
        # fps = 30
        # width = 960
        # height = 720
        save_name = args.save_name
        save_folder = os.path.join(
            vis_folder, save_name
        )
        os.makedirs(save_folder, exist_ok=True)
        print(frame.shape)
        if args.demo == "video":
            save_path = os.path.join(save_folder, args.path.split(os.sep)[-1])
        else:
            save_path = os.path.join(save_folder, save_name + ".mp4")
        logger.info(f"video save_path is {save_path}")
        # vid_writer = cv2.VideoWriter(
        #     save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        # )
        tracker = BYTETracker(args, frame_rate=30)
        timer = Timer()
        frame_id = 0
        results = []
    
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        
        outputs, img_info = predictor.inference(frame, timer)
        im = img_info['deal_img']
        im0 = img_info['raw_img']

        if outputs[0] is not None:
            print('有目标！！！！！！！！！！！！！！！！！！')
            print('outputs:', outputs)
            # print(det.shape)
            outputs[0][:, :4] = scale_coords(im.shape[2:], outputs[0][:, :4], im0.shape).round()
            class_name = outputs[0][:, 5]
            # online_targets = tracker.update(det, [img_info['height'], img_info['width']], args.tsize)
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], im0.shape[:2])
            # online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], args.tsize)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_classes = []
            for t in online_targets:
                # print(t.classes)
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_classes.append(t.classes)
                    results.append(
                                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, scores=online_scores, frame_id=frame_id + 1, fps=1. / timer.average_time, ids2=online_classes)
        else:
            timer.toc()
            online_im = img_info['raw_img']
        # if args.save_result:
        #     vid_writer.write(online_im)
        cv2.imshow("online_im", online_im)
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

        frame_id += 1
    

    rtscap.stop_read()
    rtscap.release()
    cv2.destroyAllWindows()


def main(args):
    # if not args.experiment_name:
    #     args.experiment_name = exp.exp_name

    output_dir = os.path.join('runs', '')
    os.makedirs(output_dir, exist_ok=True)
    if args.save_result:
        vis_folder = os.path.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)
        
    device = torch.device("cuda:{}".format(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    logger.info("Args: {}".format(args))

    conf_thresh = args.conf_thres
    iou_thresh = args.iou_thres
    num_classes = args.num_classes
    
    ckpt_file = args.ckpt
    
    # device = select_device(args.device)
    model = DetectMultiBackend(ckpt_file, device)
    model.eval()

    if args.fp16:
        model = model.half()

    predictor = Predictor(model, num_classes, conf_thresh, iou_thresh, args.tsize, device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result, args.save_name, args.tsize)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    assert args.demo in ["image", "video", "webcam"], "demo type not supported, only support [image, video, webcam]"
    assert args.tsize in [(1280, 1280), (640, 640), (980, 980), (800, 800), (720, 960)], "tsize not supported, only (800, 1440) and (608, 1088)"
    main(args)
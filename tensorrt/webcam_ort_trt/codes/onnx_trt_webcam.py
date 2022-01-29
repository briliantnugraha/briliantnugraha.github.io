# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 20:32:25 2022

@author: brili
"""

import cv2
from time import time
import numpy as np

import sys
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
trt_main_complete_path = os.path.join(curr_dir, '..', '..', 'understand_trt_complete', 'tensorrt_expl_complete')
sys.path.insert(0, trt_main_complete_path)
print('sys path: ', sys.path)

from trt_main_complete import YOLOX_runner
from trt_main_complete import COCO_CLASSES
import argparse


class WebcamViewer:
    def __init__(self, 
                 rtsp_url: str,
                 model_path: str,
                 font=cv2.FONT_HERSHEY_SIMPLEX,
                 fontScale = 0.35,
                 bg_fps=[np.array([[10,10],[250,10],[250,40],[10,40]])],
                 title_window='[WebcamViewer-Made by Brilian]',
                 batch_size=1,
                 nms_threshold=0.45,
                 score_threshold=0.3):
        print('[WebcamViewer] Initialize variables...')
        self.font = font
        self.fontScale = fontScale
        self.bg_fps = bg_fps
        
        self.start_t = time()
        self.check_t = self.start_t
        self.ctr = 0
        self.fps = 0
        self.read_frame_t = 0.
        self.title_window = title_window
        
        # YOLO model + utils
        self.model_path = model_path
        self.dtype = np.float16 if '16' in model_path.split('.')[0][-3:] else np.float32
        if os.path.exists(model_path):
            print('[WebcamViewer] Model path is correct, initialize the model...')
            self.ylrunner = YOLOX_runner()
            self.USE_MODE = model_path.split('.')[-1]
            
            if 'onnx' != self.USE_MODE:
                print('[WebcamViewer] load Tensor-RT...')
                from trt_main_complete import TRTEngine
                self.model = TRTEngine(model_path, dtype=self.dtype)
                shape = self.model.engine.get_binding_shape(0)
            else:    
                print('[WebcamViewer] load ONNX...')            
                from trt_main_complete import load_model_onnxruntime
                self.model, shape, self.inpname = load_model_onnxruntime(model_path, sess_options_enable=False)
            self.shapetuple = tuple(shape[2:])
            self.batch_size = batch_size
            self.nms_threshold = nms_threshold
            self.score_threshold = score_threshold
            print('[WebcamViewer] Initialize model finished...')
        
    def do_preprocessing(self, img):
        img_prep, ratio = self.ylrunner.preprocess(img, self.shapetuple)
        img_prep = img_prep.astype(self.dtype)[None,...]
        return img_prep, ratio
        
    
    def do_postprocessing(self, result_infer, ipshape, ratio):
        result_pp = self.ylrunner.demo_postprocess(result_infer[0].reshape(ipshape[0],-1, 85), self.shapetuple)
        result_out = self.ylrunner.filter_with_nms(result_pp[0], 
                                              nms_threshold=self.nms_threshold, 
                                              score_threshold=self.score_threshold,
                                              ratio=ratio)
        return result_out
    
    def do_inference(self, img):
        img_prep, ratio = self.do_preprocessing(img)
        if 'onnx' != self.USE_MODE:
            result_infer = self.model(img_prep,self.batch_size)
        else:
            result_infer = self.model.run(None, {self.inpname:img_prep})
        result_out = self.do_postprocessing(result_infer, img_prep.shape, ratio)
        return result_out
    
    def draw_output_from_inference(self, result_out):
        if result_out is not None:
            self.frame = self.ylrunner.vis(self.frame, 
                                       result_out[:, :4], 
                                       result_out[:, 4], 
                                       result_out[:, 5],
                                       conf=self.score_threshold, 
                                       class_names=COCO_CLASSES)
    
    def check_or_reset_time(self):
        if self.check_t-self.start_t >= 1.:
            self.fps = self.ctr
            self.ctr = 0
            self.start_t = time()
            
    def draw_fps_to_frame(self):
        cv2.fillPoly(self.frame, self.bg_fps, (255,255,255))
        self.frame = cv2.putText(self.frame, '[INFO] FPS: {}s, load_t: {:.3f}, infer_t: {:.3f}s'.format(self.fps, self.read_frame_t, self.read_frame_t2), 
                                  (10,30), self.font, self.fontScale, (0,0,255), 1, cv2.LINE_AA)
        
    def show_image(self):
        cv2.imshow(self.title_window, self.frame)
        
    
    def run(self):
        # check if ipcam is running, if not, then exit
        try:
            self.ipcam = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if not self.ipcam.isOpened():
                print('[WebcamViewer] Webcam is not connected, please try again...')
                return
        
            # do this if it is running
            print('[WebcamViewer] Start Webcam Inference with YOLOX')
            while self.ipcam.isOpened():
                self.check_or_reset_time()
                check_frame_t = time()
                _, self.frame = self.ipcam.read()
                self.read_frame_t = time()-check_frame_t
                # self.draw_fps_to_frame()
                
                if os.path.exists(self.model_path):
                    check_frame_t = time()
                    result_out = self.do_inference(self.frame)
                    self.read_frame_t2 = time()-check_frame_t
                    self.draw_output_from_inference(result_out)
                self.draw_fps_to_frame()
                
                self.show_image()
                
                self.check_t = time()
                self.ctr += 1
                
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):
                    self.ipcam.release()
                    break
        except Exception as e:
            print('[WebcamViewer] error: ', e)
            self.ipcam.release()
            self.exit_runner()
            
        self.exit_runner()
    
    def exit_runner(self):
        print('[WebcamViewer] Stop and Exit Webcam Viewer...')
        cv2.destroyAllWindows()
        
def parse_argument():
    parser = argparse.ArgumentParser(description='Parse custom setting to provided configs.')
    parser.add_argument('--batch_size', nargs="?", type=int, default=1,
                        help='Number of batch for inference')
    parser.add_argument('--score_threshold', nargs="?", type=float, default=0.3,
                        help='Find object above these threshold')
    parser.add_argument('--model_path', nargs="?", type=str, default=os.path.join(r"F:\gitdata\test_trt","yolox_m.onnx"),
                        help='Define model path of your model+model filename')
    parser.add_argument('--img_test_path', nargs="?", type=str, default='test_image4.jpg',
                        help='Define img test path + imgname')
    parser.add_argument('--nms_threshold', nargs="?", type=float, default=0.45,
                        help='Default nms threshold from YOLOX')
    parser.add_argument('--rtsp_url', nargs="?", type=str, default='http://gyofarras:gyofarras@192.168.1.101:4747/video',
                        help='Define your rtsp_url path (use username+password if possible for safety)')
    parser.add_argument('--title_window', nargs="?", type=str, default='[WebcamViewer-Made by Brilian]',
                        help='Define your image window...')
    
    args = parser.parse_args()
    args.img_test_path = args.img_test_path.split('.')
    print('[parse_argument] Used config: ', args)
    return args

if __name__ == '__main__':
    rtsp_url = 'http://gyofarras:gyofarras@192.168.1.101:4747/video'
    cfg = parse_argument()
    model_path = os.path.join(r"F:\gitdata\test_trt","yolox_m.trt")
    # model_path = ''
    icam_viewer = WebcamViewer(cfg.rtsp_url, cfg.model_path,
                               nms_threshold=cfg.nms_threshold,
                               score_threshold=cfg.score_threshold,
                               title_window=cfg.title_window,
                               batch_size=cfg.batch_size)
    icam_viewer.run()
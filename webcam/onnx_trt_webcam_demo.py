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
trt_main_complete_path = os.path.join(curr_dir, '..', 'tensorrt', 'understand_trt_complete', 'tensorrt_expl_complete')
sys.path.insert(0, trt_main_complete_path)
# print('sys path: ', sys.path)

# from trt_main_complete import load_model_onnxruntime


class WebcamViewer:
    def __init__(self, 
                 rtsp_url: str,
                 font=cv2.FONT_HERSHEY_SIMPLEX,
                 fontScale = 0.5,
                 bg_fps=[np.array([[10,10],[200,10],[200,40],[10,40]])],
                 title_window='[WebcamViewer-Made by Brilian]'):
        self.rtsp_url = rtsp_url
        self.font = font
        self.fontScale = fontScale
        self.bg_fps = bg_fps
        
        self.ipcam = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.start_t = time()
        self.check_t = self.start_t
        self.ctr = 0
        self.fps = 0
        self.read_frame_t = 0.
        self.title_window = title_window
        
    
    def get_image(self):
        pass
    
    def do_inference(self):
        pass
    
    def draw_output(self):
        pass
    
    def check_or_reset_time(self):
        if self.check_t-self.start_t >= 1.:
            self.fps = self.ctr
            self.ctr = 0
            self.start_t = time()
            
    def draw_fps_to_frame(self):
        cv2.fillPoly(self.frame, self.bg_fps, (255,255,255))
        self.frame = cv2.putText(self.frame, '[INFO] FPS: {}s, load_t: {:.3f}s'.format(self.fps, self.read_frame_t), 
                                  (10,30), self.font, self.fontScale, (0,0,255), 1, cv2.LINE_AA)
        
    def show_image(self):
        cv2.imshow(self.title_window, self.frame)
        
    
    def run(self):
        # check if ipcam is running, if not, then exit
        try:
            if not self.ipcam.isOpened():
                print('[WebcamViewer] Webcam is not connected, please try again...')
                return
        
            # do this if it is running
            while self.ipcam.isOpened():
                self.check_or_reset_time()
                check_frame_t = time()
                _, self.frame = self.ipcam.read()
                self.read_frame_t = time()-check_frame_t
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
        cv2.destroyAllWindows()
        

if __name__ == '__main__':
    # rtsp_url = 'http://gyofarras:gyofarras@192.168.0.30:4747/video'
    rtsp_url = 'http://gyofarras:gyofarras@192.168.0.18:4747/video'
    icam_viewer = WebcamViewer(rtsp_url)
    icam_viewer.run()
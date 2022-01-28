# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:56:34 2022

@author: brili
"""
import os
pathdir = os.path.dirname(os.path.abspath(__file__))
import sys
basepath = os.path.join(pathdir, '..', '..','understand_trt', 'tensorrt_explanation')
sys.path.insert(0,basepath)

import time
import numpy as np
import cv2
import argparse
import onnxruntime as ort
# import PIL

COCO_CLASSES = "person...bicycle...car...motorcycle...airplane...bus...train...truck...boat...traffic light...fire hydrant...stop sign...parking meter...bench...bird...cat...dog...horse...sheep...cow...elephant...bear...zebra...giraffe...backpack...umbrella...handbag...tie...suitcase...frisbee...skis...snowboard...sports ball...kite...baseball bat...baseball glove...skateboard...surfboard...tennis racket...bottle...wine glass...cup...fork...knife...spoon...bowl...banana...apple...sandwich...orange...broccoli...carrot...hot dog...pizza...donut...cake...chair...couch...potted plant...bed...dining table...toilet...tv...laptop...mouse...remote...keyboard...cell phone...microwave...oven...toaster...sink...refrigerator...book...clock...vase...scissors...teddy bear...hair drier...toothbrush"
COCO_CLASSES = COCO_CLASSES.split('...')
_COLORS = [np.array([np.random.randint(256) for _ in range(3)]) for _ in range(len(COCO_CLASSES))]

# ref: https://github.com/microsoft/onnxruntime/issues/1173#issue-452814030
def export_onnxfp32_to_onnxfp16(pathinp=r'F:\gitdata\test_trt\yolox_m.onnx'):
    pathout = pathinp.split('.')
    pathout[-2] = '{}16'.format(pathout[-2])
    pathout = '.'.join(pathout)
    import onnxmltools
    from onnxmltools.utils.float16_converter import convert_float_to_float16
    onnx_model = onnxmltools.utils.load_model(pathinp)
    onnx_model = convert_float_to_float16(onnx_model)
    onnxmltools.utils.save_model(onnx_model, pathout)

class YOLOX_runner:
    # ref: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py#L144-L160
    @staticmethod
    def preprocess(img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114
    
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r
    
    
    # ref: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/demo_utils.py#L99-L124
    @staticmethod
    def demo_postprocess(outputs, img_size, p6=False):
    
        grids = []
        expanded_strides = []
    
        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]
    
        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]
    
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
    
        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    
        return outputs
    
    # ref: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/demo_utils.py#L80-L96
    @staticmethod
    def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
    
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = YOLOX_runner.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets
    
    # ref: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/onnx_inference.py#L75-L84
    @staticmethod
    def filter_with_nms(predictions, nms_threshold=0.45, score_threshold=0.1):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
    
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = YOLOX_runner.multiclass_nms_class_agnostic(boxes_xyxy, scores, nms_thr=nms_threshold, score_thr=score_threshold)
        return dets
    
    # ref: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/demo_utils.py#L17-L44
    @staticmethod
    def nms(boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
    
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
    
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
    
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
    
            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]
    
        return keep
    
    
    # ref: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/visualize.py#L11-L42
    @staticmethod
    def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
    
            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
    
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    
            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.8, txt_color, thickness=1)
    
        return img


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
    parser.add_argument('--export_onnx', action='store_true',
                        help='Set whether to export ONNX FP32 to FP16. Need to install "pip install onnxmltools"')
    parser.add_argument('--export_onnx_path', nargs="?", type=str, default="",
                        help='Define your FP32 onnx, output poth format=onnx_path_dir/[modelname]16.onnx')
    
    
    args = parser.parse_args()
    args.img_test_path = args.img_test_path.split('.')
    print('Used config: ', args)
    return args

def load_model_onnxruntime(onnxpath, 
                           EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider'],
                           sess_options_enable=True ):
    
    if sess_options_enable:
        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = ort.InferenceSession(onnxpath, sess_options if sess_options_enable else None, providers=EP_list)
    shape = model.get_inputs()[0].shape
    inpname = model.get_inputs()[0].name
    return model, shape, inpname

if __name__ == "__main__":
    cfg = parse_argument()
    print('[CHECK CONFIG]')
    print(cfg.batch_size)
    print(cfg.score_threshold)
    print(cfg.model_path)
    print(cfg.img_test_path)
    print(cfg.nms_threshold)
    print(cfg.export_onnx)
    
    USE_MODE = cfg.model_path.split('.')[-1]
    dtype=np.float16 if '16' in cfg.model_path.split('.')[0][-3:] else np.float32
    if USE_MODE != 'onnx' and dtype==np.float16: dtype = np.float32 
    dtype_str = str(dtype).split('\'')[1].split('float')[-1]
    # dtype=np.float32
    print(dtype)
    print(cfg.model_path.split('.')[-1])
    print('='*50 + '\n')
    
    if cfg.export_onnx:
        export_onnxfp32_to_onnxfp16(cfg.export_onnx_path)
        sys.exit()
    
    if 'onnx' != USE_MODE:
        from trt_main import TRTEngine
        model = TRTEngine(cfg.model_path, dtype=dtype)
        shape = model.engine.get_binding_shape(0)
    else:        
        # import onnxruntime as ort
        # EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # sess_options = ort.SessionOptions()
        # sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # model = ort.InferenceSession(cfg.model_path, sess_options, providers=EP_list)
        # shape = model.get_inputs()[0].shape
        # inpname = model.get_inputs()[0].name
        model, shape, inpname = load_model_onnxruntime(cfg.model_path, sess_options_enable=False)
    shapetuple = tuple(shape[2:])
    ylrunner = YOLOX_runner()

    
    # data = (np.random.randint(0,255,(batch_size,*shape[1:]))/255).astype(dtype)
    impath = r'test_image4.jpg'.split('.')
    img = cv2.imread('.'.join(impath))
    for i in range(30):
        start = time.time()
        img_prep, ratio = ylrunner.preprocess(img, shapetuple)
        img_prep = img_prep.astype(dtype)[None,...]
        start2 = time.time()
        if 'onnx' != USE_MODE:
            result_infer = model(img_prep,cfg.batch_size)
        else:
            result_infer = model.run(None, {inpname:img_prep})
        start3 = time.time()
        result_pp = ylrunner.demo_postprocess(result_infer[0].reshape(img_prep.shape[0],-1, 85), shapetuple)
        start4 = time.time()
        result_out = ylrunner.filter_with_nms(result_pp[0], 
                                              nms_threshold=cfg.nms_threshold, 
                                              score_threshold=cfg.score_threshold)
        start5 = time.time()
        # result = result.reshape(1, -1, 85)
        print('{:2d}. preproc: {:.3f}s, infer: {:.3f}s, postproc: {:.3f}s, nms-filter: {:.3f}s, total: {:.3f}s'.format(i,
                               start2-start, start3-start2, start4-start3, start5-start4, time.time()-start))
    origin_img = ylrunner.vis(img, result_out[:, :4], result_out[:, 4], result_out[:, 5],
                      conf=cfg.score_threshold, class_names=COCO_CLASSES)
    impathout = impath.copy()
    impathout[-2] = impathout[-2]+'_out_{}{}'.format(USE_MODE, dtype_str)
    cv2.imwrite('.'.join(impathout), origin_img)
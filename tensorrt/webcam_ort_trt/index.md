<style>
  .center {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 50%;
  }

  td, th {
  border: 1px solid #dddddd;
  /* padding: 8px; */
}
</style>

<div>
  <lic><a href="../../"><h4>MAIN PAGE</h4></a></lic>
  <lic><a href="../"><h4>TENSOR-RT PAGE</h4></a></lic>
</div>

## Webcam Inference with ONNXRuntime and TENSOR-RT
---
January 29th, 2022

The whole code is available at <a href="http://github.com/briliantnugraha/briliantnugraha.github.io/tensorrt/webcam_ort_trt/">here</a>.

<p class="center">
    After completing the <a href="../understand_trt_complete">Tensor-RT's Complete Inference</a>, you will probably start itching to do some real stuff, aren't you?
    Btw, the webcam implementation is available at <a href="./codes/onnx_trt_webcam.py">*onnx_trt_webcam.py*</a>
</p>

### Now, we'll do something fun using IP-Camera using existing Android/Iphone app + Python!


What will be required in here:
1. A Linux/Windows 10 PC (I am currently using Windows 10).
2. Clone my repo, and you will realize that both previous blogs are connected to this one (as I am re-using those existing codes!).
3. A phone with IOS/Android that has been installed with IP-Camera. I personally prefer to use DroidCam. I've compared "IP Wecam" and "DroidCam", and IMHO, the first one seems to be laggy compared to the latter one.

***Well, That's all you need!***

---
## Implementation

``` 
Tensor-RT (abbrev. TRT)
ONNX-Runtime (abbrev. ORT)
```


For the implementation parts. I will separate the process into:
1. Base variable initialization.
2. Model initialization (either ORT or TRT).
3. Webcam inference.
    - Capturing Ip Video Stream (cv2.VideoCapture).
    - Check if the Ip Camera is open/running.
    - Get stream/frame image from the Ip Camera.
    - Do Inference.
    - Draw output and FPS.
    - Close when the Ip Camera is closed or we manually close the streaming window (use 'esc'/'q' for exit).


---
## Result

As I mentioned in <a href="./understand_trt_complete/">my previous post</a>, the winner of this test is ***Tensor-RT FP16*** with ~17ms. In this test, the speed rank is also similar (TRT FP16 > TRT FP32 > ORT FP32 > ORT FP16). The difference is that the speed with TRT-FP16 is slightly slower when I tested it in using the WebcamViewer code.

Here are some snapshot between ONNX FP32 vs ONNX FP16 vs Tensor-RT FP32 vs Tensor-RT FP16.

| ONNX FP32      | ONNX FP16 |
| :----:       |    :----:   |
| <video autoplay loop muted="muted" plays-inline="true"  width="100%" height="100%"><source src="./demo_videos/demo_ort_fp32.mp4" type="video/mp4"></video>      | <video autoplay loop muted plays-inline="true"  width="100%" height="100%"><source src="./demo_videos/demo_ort_fp16.mp4" type="video/mp4"></video> |  
| **Tensor-RT FP32**      | **Tensor-RT FP16** |
|  <video autoplay loop muted plays-inline="true"  width="100%" height="100%"><source src="./demo_videos/demo_trt_fp32.mp4" type="video/mp4"></video>   |  <video autoplay loop muted plays-inline="true"  width="100%" height="100%"><source src="./demo_videos/demo_trt_fp16.mp4" type="video/mp4"></video> | 

As you can see, the models havbe is as the following:

---
## Summary

In summary, the streaming speed using the Webcam Viewer script is as follow:
1. Both Tensor-RT FP16 and FP32 could achieve ~30-33 FPS.
2. ONNXRuntime FP32 with could achieve ~23-24 FPS.
3. ONNXRuntime FP16 with could achieve ~21-22 FPS.


---
## How to Reproduce

For starter, you can use this command to check the available options.
``` 
git clone my github (briliantnugraha.github.io)
cd ./tensorrt/webcam_ort_trt
python ./codes/onnx_trt_yolox_webcam.py -h
```

And these are the command that I use to reproduce the streaming results (ORT and TRT)
```
python ./codes/onnx_trt_yolox_webcam.py --model_path "F:/gitdata/test_trt/yolox_m.trt" --rtsp_url "http://gyofarras:gyofarras@192.168.1.101:4747/video" #TRT-FP32
python ./codes/onnx_trt_yolox_webcam.py --model_path \"F:/gitdata/test_trt/yolox_m16.trt\" --rtsp_url \"http://gyofarras:gyofarras@192.168.1.101:4747/video" #TRT-FP16
python ./codes/onnx_trt_yolox_webcam.py --model_path \"F:/gitdata/test_trt/yolox_m.onnx\" --rtsp_url \"http://gyofarras:gyofarras@192.168.1.101:4747/video" #ORT-FP32
python ./codes/onnx_trt_yolox_webcam.py --model_path "F:/gitdata/test_trt/yolox_m16.onnx" --rtsp_url "http://gyofarras:gyofarras@192.168.1.101:4747/video" #ORT-FP16
```

---
## There you go!

Hope this explanation helps. If there is any question or mistake with the content, please don't hesitate to let me know, see you in the next blog and stay safe!


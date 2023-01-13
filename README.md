# Vehicle-CV-ADAS
Example scripts for the detection of lanes using the [ultra fast lane detection v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) model in ONNX/TensorRT.

<<<<<<< HEAD
Example scripts for the detection of objects using the [yolov5](https://github.com/ultralytics/yolov5)/[yolov5-lite](https://github.com/ppogg/YOLOv5-Lite) model in ONNX.
=======
Example scripts for the detection of objects using the [yolov5](https://github.com/ultralytics/yolov5) model in ONNX/TensorRT.
>>>>>>> a471428 (support tensorRT infer.)

![!ADAS on video](https://github.com/jason-li-831202/Vehicle-CV-ADAS/blob/master/TrafficLaneDetector/temp/pic/demo.JPG)


## Requirements

 * **OpenCV**, **Scikit-learn**, **onnxruntime**, **pycuda** and **pytorch**. 
 
## Examples
  * **Video inference**:
 
 ```
 python demo.py
 ```

   * **Comvert Onnx to TenserRT model**:
 
 ```
 python convertOnnxToTensorRT.py
 ```
 
## [Inference video Example](https://www.youtube.com/watch?v=CHO0C1z5EWE) 
![!ADAS on video](https://github.com/jason-li-831202/Vehicle-CV-ADAS/blob/master/TrafficLaneDetector/temp/demo.gif)

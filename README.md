# Vehicle-CV-ADAS
Example scripts for the detection of lanes using the [ultra fast lane detection v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) model in ONNX/TensorRT.

Example scripts for the detection of objects using the [yolov5](https://github.com/ultralytics/yolov5)/[yolov5-lite](https://github.com/ppogg/YOLOv5-Lite)/[yolov8](https://github.com/ultralytics/ultralytics) model in ONNX/TensorRT.

![!ADAS on video](https://github.com/jason-li-831202/Vehicle-CV-ADAS/blob/master/TrafficLaneDetector/temp/pic/demo.JPG)


## Requirements

 * **OpenCV**, **Scikit-learn**, **onnxruntime**, **pycuda** and **pytorch**. 
 
## Examples
  * **Comvert Onnx to TenserRT model**:
 
 ```
 python convertOnnxToTensorRT.py
 ```
 
  * **Video inference**:
 
 ```
 python demo.py
 ```
 
  * ***Setting Config***:
 
  ```python
  lane_config = {
   "model_path": "./TrafficLaneDetector/models/culane_res18.trt",
   "model_type" : LaneModelType.UFLDV2_CULANE
  }

  object_config = {
   "model_path": './ObjectDetector/models/yolov8l-coco.trt',
   "model_type" : ObjectModelType.YOLOV8,
   "classes_path" : './ObjectDetector/models/coco_label.txt',
   "box_score" : 0.4,
   "box_nms_iou" : 0.45
  }
 ```
 | Target          | Model Type                       | 
 | :-------------: |:-------------------------------- | 
 | Lanes           | `LaneModelType.UFLD_TUSIMPLE`    | 
 | Lanes           | `LaneModelType.UFLD_CULANE`      |   
 | Lanes           | `LaneModelType.UFLDV2_TUSIMPLE`  |   
 | Lanes           | `LaneModelType.UFLDV2_CULANE`    | 
 | Object          | `ObjectModelType.YOLOV5`         | 
 | Object          | `ObjectModelType.YOLOV5_LITE`    | 
 | Object          | `ObjectModelType.YOLOV8`         | 


## [Inference video Example](https://www.youtube.com/watch?v=CHO0C1z5EWE) 
![!ADAS on video](https://github.com/jason-li-831202/Vehicle-CV-ADAS/blob/master/TrafficLaneDetector/temp/demo.gif)

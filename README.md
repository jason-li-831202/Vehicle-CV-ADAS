# Vehicle-CV-ADAS

<p>
    <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-14354C.svg?logo=python&logoColor=white"></a>
    <a href="#"><img alt="OnnxRuntime" src="https://img.shields.io/badge/OnnxRuntime-FF6F00.svg?logo=onnx&logoColor=white"></a>
    <a href="#"><img alt="TensorRT" src="https://img.shields.io/badge/TensorRT-49D.svg?logo=flask&logoColor=white"></a>
    <a href="#"><img alt="Markdown" src="https://img.shields.io/badge/Markdown-000000.svg?logo=markdown&logoColor=white"></a>
    <a href="#"><img alt="Visual Studio Code" src="https://img.shields.io/badge/Visual%20Studio%20Code-ad78f7.svg?logo=visual-studio-code&logoColor=white"></a>
    <a href="#"><img alt="Windows" src="https://img.shields.io/badge/Windows-0078D6?logo=windows&logoColor=white"></a>
</p>

Example scripts for the detection of lanes using the [ultra fast lane detection v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) model in ONNX/TensorRT.

Example scripts for the detection of objects using the [YOLOv5](https://github.com/ultralytics/yolov5)/[YOLOv5-lite](https://github.com/ppogg/YOLOv5-Lite)/[YOLOv8](https://github.com/ultralytics/ultralytics) model in ONNX/TensorRT.

# ➤ Contents
1) [Requirements](#Requirements)

2) [Examples](#Examples)

3) [Demo](#Demo)

4) [License](#License)

![!ADAS on video](https://github.com/jason-li-831202/Vehicle-CV-ADAS/blob/master/demo/demo.JPG)

<h1 id="Requirements">➤ Requirements</h1>

* **OpenCV**, **Scikit-learn**, **onnxruntime**, **pycuda** and **pytorch**.

* **Install :**

    The `requirements.txt` file should list all Python libraries that your notebooks
    depend on, and they will be installed using:

    ```
    pip install -r requirements.txt
    ```
    

<h1 id="Examples">➤ Examples</h1>

 * ***Convert Onnx to TenserRT model*** :

    Need to modify `onnx_model_path` and `trt_model_path` before converting.

    ```
    python convertOnnxToTensorRT.py -i <path-of-your-onnx-model>  -o <path-of-your-trt-model>
    ```

 * ***Quantize ONNX models*** :

    Converting a model to use float16 instead of float32 can decrease the model size.
    ```
    python onnxQuantization.py -i <path-of-your-onnx-model>
    ```

 * ***Video Inference*** :

   * Setting Config :
     > Note : can support onnx/tensorRT format model. But it needs to match the same model type.

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
   | Target          | Model Type                       |  Describe                                         | 
   | :-------------: |:-------------------------------- | :------------------------------------------------ | 
   | Lanes           | `LaneModelType.UFLD_TUSIMPLE`    | Support Tusimple data with ResNet18 backbone.     | 
   | Lanes           | `LaneModelType.UFLD_CULANE`      | Support CULane data with ResNet18 backbone.       | 
   | Lanes           | `LaneModelType.UFLDV2_TUSIMPLE`  | Support Tusimple data with ResNet18/34 backbone.  |
   | Lanes           | `LaneModelType.UFLDV2_CULANE`    | Support CULane data with ResNet18/34 backbone.    | 
   | Object          | `ObjectModelType.YOLOV5`         | Support yolov5n/s/m/l/x model.                    | 
   | Object          | `ObjectModelType.YOLOV5_LITE`    | Support yolov5lite-e/s/c/g model.                 | 
   | Object          | `ObjectModelType.YOLOV8`         | Support yolov8n/s/m/l/x model.                    | 
   
   * Run :
   
    ```
    python demo.py
    ```

<h1 id="Demo">➤ Demo</h1>

* [***Demo Youtube Video***](https://www.youtube.com/watch?v=CHO0C1z5EWE)

* ***Display***

    ![!ADAS on video](https://github.com/jason-li-831202/Vehicle-CV-ADAS/blob/master/demo/demo-gif.gif)

* ***Front Collision Warning System (FCWS)***

    ![!FCWS](https://github.com/jason-li-831202/Vehicle-CV-ADAS/blob/master/demo/FCWS.jpg)

* ***Lane Departure Warning System (LDWS)***

    ![!LDWS](https://github.com/jason-li-831202/Vehicle-CV-ADAS/blob/master/demo/LDWS.jpg)

* ***Lane Keeping Assist System (LKAS)***

    ![!LKAS](https://github.com/jason-li-831202/Vehicle-CV-ADAS/blob/master/demo/LKAS.jpg)

<h1 id="License">➤ License</h1>
WiFi Analyzer is licensed under the GNU General Public License v3.0 (GPLv3).

**GPLv3 License key requirements** :
* Disclose Source
* License and Copyright Notice
* Same License
* State Changes
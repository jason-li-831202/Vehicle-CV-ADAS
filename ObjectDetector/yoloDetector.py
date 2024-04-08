import os
import cv2
import random
import logging
import numpy as np
from typing import *
try :
	import sys
	from utils import ObjectModelType, hex_to_rgb, NMS, Scaler
	from core import ObjectDetectBase, RectInfo
	sys.path.append("..")
	from coreEngine import TensorRTEngine, OnnxEngine
except :
	from .utils import ObjectModelType, hex_to_rgb, NMS, Scaler
	from .core import ObjectDetectBase, RectInfo
	from coreEngine import TensorRTEngine, OnnxEngine

class YoloLiteParameters():
	def __init__(self, model_type, input_shape, num_classes):
		self.lite = False
		if (model_type == ObjectModelType.YOLOV5_LITE) :
			self.lite = True
		anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
		self.nl = len(anchors)
		self.na = len(anchors[0]) // 2
		self.no = num_classes + 5
		self.grid = [np.zeros(1)] * self.nl
		self.stride = np.array([8., 16., 32.])
		self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
		self.input_shape = input_shape[-2:]

	def __make_grid(self, nx=20, ny=20):
		xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
		return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

	def lite_postprocess(self, outs):
		if self.lite :
			row_ind = 0
			for i in range(self.nl):
				h, w = int(self.input_shape[0] / self.stride[i]), int(self.input_shape[1] / self.stride[i])
				length = int(self.na * h * w)
				if self.grid[i].shape[2:4] != (h, w):
					self.grid[i] = self.__make_grid(w, h)

				outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
					self.grid[i], (self.na, 1))) * int(self.stride[i])
				outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
					self.anchor_grid[i], h * w, axis=0)
				row_ind += length
		return outs

class YoloDetector(ObjectDetectBase, YoloLiteParameters):
	_defaults = {
		"model_path": './models/yolov5n-coco.onnx',
		"model_type" : ObjectModelType.YOLOV5,
		"classes_path" : './models/coco_label.txt',
		"box_score" : 0.4,
		"box_nms_iou" : 0.45
	}

	def __init__(self, logger=None, **kwargs):
		ObjectDetectBase.__init__(self, logger)
		self.__dict__.update(kwargs) # and update with user overrides

		self._initialize_class(self.classes_path)
		self._initialize_model(self.model_path)
		YoloLiteParameters.__init__(self, self.model_type, self.input_shapes, len(self.class_names))

	def _initialize_model(self, model_path : str) -> None:
		model_path = os.path.expanduser(model_path)
		if (self.logger) :
			self.logger.debug("model path: %s." % model_path)

		if model_path.endswith('.trt') :
			self.engine = TensorRTEngine(model_path)
		else :
			self.engine = OnnxEngine(model_path)

		if (self.logger) :
			self.logger.info(f'YoloDetector Type : [{self.engine.framework_type}] || Version : [{self.engine.providers}]')
		self.set_input_details(self.engine)
		self.set_output_details(self.engine)

	def _initialize_class(self, classes_path : str) -> None:
		classes_path = os.path.expanduser(self.classes_path)
		if (self.logger) :
			self.logger.debug("class path: %s." % classes_path)
		assert os.path.isfile(classes_path), Exception("%s is not exist." % classes_path)

		with open(classes_path) as f:
			class_names = f.readlines()
		self.class_names = [c.strip() for c in class_names]
		get_colors = list(map(lambda i: hex_to_rgb("#" +"%06x" % random.randint(0, 0xFFFFFF)), range(len(self.class_names)) ))
		self.colors_dict = dict(zip(list(self.class_names), get_colors))

	def __prepare_input(self, srcimg : cv2) -> Tuple[np.ndarray, Scaler] :
		scaler = Scaler(self.input_shapes[-2:], True)
		image = scaler.process_image(srcimg)
		# HWC -> NCHW format
		blob = cv2.dnn.blobFromImage(image, 1/255.0, (image.shape[1], image.shape[0]), 
										swapRB=True, crop=False).astype(self.input_types)
		return blob, scaler

	def __process_output(self, output: np.ndarray) -> Tuple[List[np.ndarray,], list, list, list]:
		_raw_boxes = []
		_raw_kpss = []
		_raw_class_ids = []
		_raw_class_confs = []

		'''
		YOLOv5/6/7 outputs shape -> (-1, obj_conf + 5[bbox, cls_conf])
		YOLOv8/9 outputs shape -> (obj_conf + 4[bbox], -1)
		'''
		if (self.model_type in [ObjectModelType.YOLOV8, ObjectModelType.YOLOV9]) :
			output = output.T
		
		output = self.lite_postprocess(output)

		# inference output
		for detection in output:
			if (self.model_type in [ObjectModelType.YOLOV8, ObjectModelType.YOLOV9]) :
				obj_cls_probs = detection[4:]
			else :
				obj_cls_probs = detection[5:] * detection[4] # cls_conf * obj_conf 

			classId = np.argmax(obj_cls_probs)
			classConf = float(obj_cls_probs[classId])
			if classConf > self.box_score :
				x, y, w, h = detection[0:4]
				_raw_class_ids.append(classId)
				_raw_class_confs.append(classConf)
				_raw_boxes.append(np.stack([(x - 0.5 * w), (y - 0.5 * h), (x + 0.5 * w), (y + 0.5 * h)], axis=-1))
		return _raw_boxes, _raw_class_ids, _raw_class_confs, _raw_kpss

	def get_nms_results(self, boxes : np.array, class_confs : list, class_ids : list, kpss : np.array) -> List[Tuple[list, list]]:
		results = []
		# nms_results = cv2.dnn.NMSBoxes(boxes, class_confs, self.box_score, self.box_nms_iou) 
		# nms_results = NMS.fast_nms(boxes, class_confs, self.box_nms_iou, "xywh") 
		nms_results = NMS.fast_soft_nms(boxes, class_confs, self.box_nms_iou, dets_type="xywh") 

		if len(nms_results) > 0:
			for i in nms_results:
				try :
					predicted_class = self.class_names[class_ids[i]] 
				except :
					predicted_class = "unknown"
				conf = class_confs[i]
				bbox = boxes[i]

				kpsslist = []
				if (kpss.size != 0) :
					for j in range(5):
						kpsslist.append( ( int(kpss[i, j, 0]) , int(kpss[i, j, 1]) ) )
				results.append(RectInfo(*bbox, conf=conf, 
											   label=predicted_class,
											   kpss=kpsslist))
		return results

	def DetectFrame(self, srcimg : cv2) -> None:
		input_tensor, scaler = self.__prepare_input(srcimg)

		output_from_network = self.engine.engine_inference(input_tensor)[0].squeeze(axis=0)

		_raw_boxes, _raw_class_ids, _raw_class_confs, _raw_kpss = self.__process_output(output_from_network)
		
		transform_boxes = scaler.convert_boxes_coordinate(_raw_boxes)
		transform_kpss = scaler.convert_kpss_coordinate(_raw_kpss)
		self._object_info = self.get_nms_results(transform_boxes, _raw_class_confs, _raw_class_ids, transform_kpss)

	def DrawDetectedOnFrame(self, frame_show : cv2) -> None:
		tl = 3 or round(0.002 * (frame_show.shape[0] + frame_show.shape[1]) / 2) + 1    # line/font thickness
		if ( len(self._object_info) != 0 )  :
			for _info in self._object_info:
				xmin, ymin, xmax, ymax = _info.tolist()
				label = _info.label

				if (len(_info.kpss) != 0) :
					for kp in _info.kpss :
						cv2.circle(frame_show,  kp, 1, (255, 255, 255), thickness=-1)
				c1, c2 = (xmin, ymin), (xmax, ymax)        
				tf = max(tl - 1, 1)  # font thickness
				t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
				c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

				if (label != 'unknown') :
					cv2.rectangle(frame_show, c1, c2, self.colors_dict[label], -1, cv2.LINE_AA)
					self.cornerRect(frame_show, _info.tolist(), colorR=self.colors_dict[label], colorC=self.colors_dict[label])
				else :
					cv2.rectangle(frame_show, c1, c2, (0, 0, 0), -1, cv2.LINE_AA)
					self.cornerRect(frame_show, _info.tolist(), colorR= (0, 0, 0), colorC= (0, 0, 0) )
				cv2.putText(frame_show, label, (xmin + 2, ymin - 7), cv2.FONT_HERSHEY_TRIPLEX, tl / 4, (255, 255, 255), 2)
		

if __name__ == "__main__":
	import time
	import sys

	capture = cv2.VideoCapture(r"./temp/test.avi")
	config = {
		"model_path": 'models/yolov9c-coco_fp16.onnx',
		"model_type" : ObjectModelType.YOLOV9,
		"classes_path" : 'models/coco_label.txt',
		"box_score" : 0.4,
		"box_nms_iou" : 0.45,
	}

	YoloDetector.set_defaults(config)
	network = YoloDetector()
	
	fps = 0
	frame_count = 0
	start = time.time()
	while True:
		_, frame = capture.read()
		k = cv2.waitKey(1)
		if k==27 or frame is None:    # Esc key to stop
			print("End of stream.", logging.INFO)
			break
		
		network.DetectFrame(frame)
		network.DrawDetectedOnFrame(frame)


		frame_count += 1
		if frame_count >= 30:
			end = time.time()
			fps = frame_count / (end - start)
			frame_count = 0
			start = time.time()

		cv2.putText(frame, "FPS: %.2f" % fps, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		cv2.imshow("output", frame)

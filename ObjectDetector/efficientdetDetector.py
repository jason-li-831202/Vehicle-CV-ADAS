import os
import cv2
import random
import logging
import numpy as np
from typing import *
try :
	import sys
	from utils import ObjectModelType, hex_to_rgb, Scaler
	from core import ObjectDetectBase, RectInfo
	sys.path.append("..")
	from coreEngine import OnnxEngine
except :
	from ObjectDetector.utils import ObjectModelType, hex_to_rgb, Scaler
	from ObjectDetector.core import ObjectDetectBase, RectInfo
	from coreEngine import OnnxEngine
	
class EfficientdetDetector(ObjectDetectBase):
	_defaults = {
		"model_path": './models/yolov5n-coco.onnx',
		"model_type" : ObjectModelType.YOLOV5,
		"classes_path" : './models/coco_label.txt',
		"box_score" : 0.6
	}

	def __init__(self, logger=None, **kwargs):
		ObjectDetectBase.__init__(self, logger)
		self.__dict__.update(kwargs) # and update with user overrides

		self._initialize_class(self.classes_path)
		self._initialize_model(self.model_path)

	def _initialize_model(self, model_path : str) -> None:
		model_path = os.path.expanduser(model_path)
		if (self.logger) :
			self.logger.debug("model path: %s." % model_path)

		self.engine = OnnxEngine(model_path)

		if (self.logger) :
			self.logger.info(f'EfficientdetDetector Type : [{self.engine.framework_type}] || Version : [{self.engine.providers}]')
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
		get_colors = list(map(lambda i:"#" +"%06x" % random.randint(0, 0xFFFFFF),range(len(self.class_names)) ))
		self.colors_dict = dict(zip(list(self.class_names), get_colors))

	def __prepare_input(self, srcimg : cv2, mean: Tuple = (0.406, 0.456, 0.485), std: Tuple = (0.225, 0.224, 0.229)) -> Tuple[np.ndarray, Scaler] :
		scaler = Scaler(self.input_shapes[-2:], True)
		image = scaler.process_image(srcimg)
		image = (image / 255 - mean) / std
	
		# HWC -> NCHW format
		blob = np.transpose(np.expand_dims(image, axis=0), (0,3,1,2)).astype(self.input_types)

		return blob, scaler

	def __process_output(self, output: np.ndarray, scaler: Scaler) -> RectInfo:
		_raw_boxes = output[0]
		_raw_class_ids = output[1]
		_raw_class_confs = output[2]
		_raw_boxes = scaler.convert_boxes_coordinate(_raw_boxes)

		results = []
		if len(_raw_boxes) != 0:
			for bbox, id, conf in zip(_raw_boxes, _raw_class_ids, _raw_class_confs):
				if (conf < self.box_score): 
					continue
				try :
					predicted_class = self.class_names[id] 
				except :
					predicted_class = "unknown"
				results.append(RectInfo(*bbox, conf=conf, label=predicted_class))
		return results

	def DetectFrame(self, srcimg : cv2) -> None:
		input_tensor, scaler = self.__prepare_input(srcimg)

		output_from_network = self.engine.engine_inference(input_tensor)

		self._object_info = self.__process_output(output_from_network, scaler)

	def DrawDetectedOnFrame(self, frame_show : cv2) -> None:
		tl = 3 or round(0.002 * (frame_show.shape[0] + frame_show.shape[1]) / 2) + 1    # line/font thickness
		if ( len(self._object_info) != 0 )  :
			for _info in self._object_info:
				xmin, ymin, xmax, ymax = _info.tolist()
				label = _info.label

				c1, c2 = (xmin, ymin), (xmax, ymax)        
				tf = max(tl - 1, 1)  # font thickness
				t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
				c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

				if (label != 'unknown') :
					cv2.rectangle(frame_show, c1, c2, hex_to_rgb(self.colors_dict[label]), -1, cv2.LINE_AA)
					self.cornerRect(frame_show, _info.tolist(), colorR= hex_to_rgb(self.colors_dict[label]), colorC= hex_to_rgb(self.colors_dict[label]))
				else :
					cv2.rectangle(frame_show, c1, c2, (0, 0, 0), -1, cv2.LINE_AA)
					self.cornerRect(frame_show, _info.tolist(), colorR= (0, 0, 0), colorC= (0, 0, 0) )
				cv2.putText(frame_show, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, tl / 3, (255, 255, 255), 2)
	

if __name__ == "__main__":
	import time
	import sys

	capture = cv2.VideoCapture(r"./temp/test.avi")
	config = {
		"model_path": 'models/efficientdet-d0-coco_fp32.onnx',
		"model_type" : ObjectModelType.EfficientDet,
		"classes_path" : 'models/coco_label.txt',
		"box_score" : 0.6,
	}

	EfficientdetDetector.set_defaults(config)
	network = EfficientdetDetector()
	
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
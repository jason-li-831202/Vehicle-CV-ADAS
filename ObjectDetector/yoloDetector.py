import os
import cv2
import random
import logging
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import onnxruntime as ort
from typing import *
try :
	from utils import ObjectModelType, hex_to_rgb, fast_nms, fast_soft_nms
except :
	from ObjectDetector.utils import ObjectModelType, hex_to_rgb, fast_nms, fast_soft_nms

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

class TensorRTParameters():

	def __init__(self, engine_file_path, num_classes, model_type):
		self.model_type =model_type
		# Create a Context on this device,
		cuda.init()
		device = cuda.Device(0)
		self.cuda_driver_context = device.make_context()

		stream = cuda.Stream()
		TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
		runtime = trt.Runtime(TRT_LOGGER)
		self.num_classes = num_classes

		# Deserialize the engine from file
		with open(engine_file_path, "rb") as f:
			engine = runtime.deserialize_cuda_engine(f.read())

		self.context =  self.__create_context(engine)
		self.host_inputs, self.cuda_inputs, self.host_outputs, self.cuda_outputs, self.bindings = self.__allocate_buffers(engine)
		self.input_shapes = engine.get_binding_shape(0)
		self.dtype = trt.nptype(engine.get_binding_dtype(0))

		# Store
		self.stream = stream
		self.engine = engine

	def __allocate_buffers(self, engine):
		"""Allocates all host/device in/out buffers required for an engine."""
		host_inputs = []
		cuda_inputs = []
		host_outputs = []
		cuda_outputs = []
		bindings = []

		for binding in engine:
			size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
			dtype = trt.nptype(engine.get_binding_dtype(binding))
			# Allocate host and device buffers
			host_mem = cuda.pagelocked_empty(size, dtype)
			cuda_mem = cuda.mem_alloc(host_mem.nbytes)
			# Append the device buffer to device bindings.
			bindings.append(int(cuda_mem))
			# Append to the appropriate list.
			if engine.binding_is_input(binding):
				host_inputs.append(host_mem)
				cuda_inputs.append(cuda_mem)
			else:
				host_outputs.append(host_mem)
				cuda_outputs.append(cuda_mem)
		return host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings

	def __create_context(self, engine):
		return engine.create_execution_context()

	def postprocess(self, input_image):
		self.cuda_driver_context.push()
		# Restore
		stream = self.stream
		context = self.context
		engine = self.engine
		host_inputs = self.host_inputs
		cuda_inputs = self.cuda_inputs
		host_outputs = self.host_outputs
		cuda_outputs = self.cuda_outputs
		bindings = self.bindings
		# Copy input image to host buffer
		np.copyto(host_inputs[0], input_image.ravel())
		# Transfer input data  to the GPU.
		cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
		# Run inference.
		context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
		# Transfer predictions back from the GPU.
		cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
		# Synchronize the stream
		stream.synchronize()
		# Remove any context from the top of the context stack, deactivating it.
		self.cuda_driver_context.pop()

		# Here we use the first row of output in that batch_size = 1
		trt_outputs = host_outputs[0]
		if (self.model_type in [ObjectModelType.YOLOV8, ObjectModelType.YOLOV9]) :
			return np.reshape(trt_outputs, (self.num_classes+4, -1 ))
		else :
			return np.reshape(trt_outputs, (-1, self.num_classes+5))

class YoloDetector(YoloLiteParameters):
	_defaults = {
		"model_path": './models/yolov5n-coco.onnx',
		"model_type" : ObjectModelType.YOLOV5,
		"classes_path" : './models/coco_label.txt',
		"box_score" : 0.4,
		"box_nms_iou" : 0.45
	}

	@classmethod
	def set_defaults(cls, config) :
		cls._defaults = config

	@classmethod
	def check_defaults(cls):
		return cls._defaults
		
	@classmethod
	def get_defaults(cls, n):
		if n in cls._defaults:
			return cls._defaults[n]
		else:
			return "Unrecognized attribute name '" + n + "'"

	def __init__(self, logger=None, **kwargs):
		self.__dict__.update(self._defaults) # set up default values
		self.__dict__.update(kwargs) # and update with user overrides
		self.logger = logger
		self.keep_ratio = True
		
		classes_path = os.path.expanduser(self.classes_path)
		if (self.logger) :
			self.logger.debug("class path: %s." % classes_path)
		self.__get_class(classes_path)

		model_path = os.path.expanduser(self.model_path)
		if (self.logger) :
			self.logger.debug("model path: %s." % model_path)
		assert os.path.isfile(model_path), Exception("%s is not exist." % model_path)
		assert model_path.endswith(('.onnx', '.trt')), 'Onnx/TensorRT Parameters must be a .onnx/.trt file.'

		if model_path.endswith('.trt') :
			self.framework_type = "trt"
			self.__load_model_tensorrt(model_path) 
		else :
			self.framework_type = "onnx"
			self.__load_model_onnxruntime(model_path)

		YoloLiteParameters.__init__(self, self.model_type, self.input_shapes, len(self.class_names))
		if (self.logger) :
			self.logger.info(f'YoloDetector Type : [{self.framework_type}] || Version : [{self.providers}]')
			self.logger.info(f"-> Input Shape : {self.input_shapes}")
			self.logger.info(f"-> Input Type  : {self.input_types}")

	def __get_class(self, classes_path : str) -> None:
		assert os.path.isfile(classes_path), Exception("%s is not exist." % classes_path)
		with open(classes_path) as f:
			class_names = f.readlines()
		self.class_names = [c.strip() for c in class_names]
		get_colors = list(map(lambda i:"#" +"%06x" % random.randint(0, 0xFFFFFF),range(len(self.class_names)) ))
		self.colors_dict = dict(zip(list(self.class_names), get_colors))

	def __load_model_onnxruntime(self, model_path : str) -> None :
		if  ort.get_device() == 'GPU' and 'CUDAExecutionProvider' in  ort.get_available_providers():  # gpu 
			self.providers = 'CUDAExecutionProvider'
		else :
			self.providers = 'CPUExecutionProvider'
		self.session = ort.InferenceSession(model_path, providers= [self.providers] )
		
		self.input_types = np.float16 if 'float16' in self.session.get_inputs()[0].type else np.float32
		self.input_shapes = self.session.get_inputs()[0].shape

	def __load_model_tensorrt(self, model_path : str) -> None :
		self.providers = 'CUDAExecutionProvider'
		self.session = TensorRTParameters(model_path, len(self.class_names), self.model_type)

		self.input_types = self.session.dtype 
		self.input_shapes = self.session.input_shapes

	@property
	def object_info(self) :
		if not hasattr(self, '_object_info') :
			self._object_info = []
			self.logger.war("Can't get object information, maybe you forget to use detect api.")
		
		return self._object_info
	
	@staticmethod
	def adjust_boxes_ratio(box : list, ratio : Union[float, None], stretch_type : Union[str, None]) -> tuple:
		""" Adjust the aspect ratio of the box according to the orientation """
		xmin, ymin, width, height = box 
		width = int(width)
		height = int(height)
		xmax = xmin + width
		ymax = ymin + height
		if (ratio != None) :
			ratio = float(ratio)
		else :
			return (xmin, ymin, xmax, ymax)
		center = ( (xmin + xmax) / 2, (ymin + ymax) / 2 )
		if (stretch_type == "居中水平") : # "person"
			# print("test : 居中水平")
			changewidth = int(height * (1/ratio))
			xmin = center[0] - changewidth/2
			xmax = xmin + changewidth
		elif (stretch_type == "居中垂直") :
			# print("test : 居中垂直")
			changeheight =  int(width * ratio)
			ymin = center[1] - (changeheight/2)
			ymax = ymin + changeheight
		elif (stretch_type == "向下") : # head+、body、upperbody
			# print("test : 向下")
			changeheight =  int(width * ratio)
			ymax = ymin + changeheight
		elif (stretch_type == "向上") :
			# print("test : 向上")
			changeheight = int( width * ratio)
			ymin =ymax - changeheight
		elif (stretch_type == "向左") :
			# print("test : 向左")
			changewidth = int(height * (1/ratio))
			xmin =xmax - changewidth
		elif (stretch_type == "向右") :
			# print("test : 向右")
			changewidth = int(height * (1/ratio))
			xmax = xmin + changewidth
		return (xmin, ymin, xmax, ymax)

	@staticmethod
	def convert_kpss_coordinate(kpss : list, ratiow : float, ratioh : float, padh : int, padw : int) -> list:
		if (kpss != []) :
			kpss = np.vstack(kpss)
			kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
			kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh
		return kpss

	@staticmethod
	def convert_boxes_coordinate(boxes : list, ratiow : float, ratioh : float, padh : int, padw : int) -> np.array:
		if not isinstance(boxes, np.ndarray):
			boxes = np.array(boxes)

		if (boxes.size > 0) :
			boxes = np.vstack(boxes)
			boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
			boxes[:, 0] = (boxes[:, 0] - padw) * ratiow
			boxes[:, 1] = (boxes[:, 1] - padh) * ratioh
			boxes[:, 2] = boxes[:, 2] * ratiow
			boxes[:, 3] = boxes[:, 3] * ratioh
		return boxes

	@staticmethod
	def cornerRect(img, bbox : list, t : int = 5, rt : int = 1, colorR : tuple = (255, 0, 255), colorC : tuple = (0, 255, 0)):
		ymin, xmin, ymax, xmax, label = bbox
		l = max(1, int(min( (ymax-ymin), (xmax-xmin))*0.2))

		if rt != 0:
			cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colorR, rt)
		# Top Left  xmin, ymin
		cv2.line(img,  (xmin, ymin), (xmin + l, ymin), colorC, t)
		cv2.line(img,  (xmin, ymin), (xmin, ymin + l), colorC, t)
		# Top Right  xmax, ymin
		cv2.line(img, (xmax, ymin), (xmax - l, ymin), colorC, t)
		cv2.line(img, (xmax, ymin), (xmax, ymin + l), colorC, t)
		# Bottom Left  xmin, ymax
		cv2.line(img, (xmin, ymax), (xmin + l, ymax), colorC, t)
		cv2.line(img, (xmin, ymax), (xmin, ymax - l), colorC, t)
		# Bottom Right  xmax, ymax
		cv2.line(img, (xmax, ymax), (xmax - l, ymax), colorC, t)
		cv2.line(img, (xmax, ymax), (xmax, ymax - l), colorC, t)

		return img

	@staticmethod
	def resize_image_format(srcimg : cv2 , frame_resize : tuple, keep_ratio=True) -> Tuple[np.ndarray, int, int, float, float, int, int]:
		padh, padw, newh, neww = 0, 0, frame_resize[0], frame_resize[1]
		if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
			hw_scale = srcimg.shape[0] / srcimg.shape[1]
			if hw_scale > 1:
				newh, neww = frame_resize[0], int(frame_resize[1] / hw_scale)
				img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_CUBIC)
				padw = int((frame_resize[1] - neww) * 0.5)
				img = cv2.copyMakeBorder(img, 0, 0, padw, frame_resize[1] - neww - padw, cv2.BORDER_CONSTANT,
										 value=0)  # add border
			else:
				newh, neww = int(frame_resize[0] * hw_scale) + 1, frame_resize[1]
				img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_CUBIC)
				padh = int((frame_resize[0] - newh) * 0.5)
				img = cv2.copyMakeBorder(img, padh, frame_resize[0] - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
		else:
			img = cv2.resize(srcimg, (frame_resize[1], frame_resize[0]), interpolation=cv2.INTER_CUBIC)
		ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
		return img, newh, neww, ratioh, ratiow, padh, padw
	
	def get_nms_results(self, boxes : np.array, class_confs : list, class_ids : list, kpss : list, priority : bool = False) -> List[Tuple[list, list]]:
		results = []
		# nms_results = cv2.dnn.NMSBoxes(boxes, class_confs, self.box_score, self.box_nms_iou) 
		if (boxes.shape[0] > 0) :
			if (boxes.shape[0] == 1) :
				nms_results = [0]
			else :
				# nms_results = fast_nms(boxes, np.array(class_confs), self.box_nms_iou) 
				nms_results = fast_soft_nms(boxes.copy(), np.array(class_confs).copy(), self.box_nms_iou) 
		else :
			nms_results = []

		if len(nms_results) > 0:
			for i in nms_results:
				kpsslist = []
				try :
					predicted_class = self.class_names[class_ids[i]]
				except :
					predicted_class = "unknown"
				if (kpss != []) :
					for j in range(5):
						kpsslist.append( ( int(kpss[i, j, 0]) , int(kpss[i, j, 1]) ) )
				bbox = boxes[i]
				bbox = self.adjust_boxes_ratio(bbox, None, None)

				xmin, ymin, xmax, ymax = list(map(int, bbox))
				results.append(([ymin, xmin, ymax, xmax, predicted_class], kpsslist))
		if (priority and len(results) > 0) :
			results = [results[0]]
		return results

	def DetectFrame(self, srcimg : cv2) -> None:
		_raw_kpss = []
		_raw_class_ids = []
		_raw_class_confs = []
		_raw_boxes = []

		image, newh, neww, ratioh, ratiow, padh, padw = self.resize_image_format(srcimg, self.input_shapes[-2:], self.keep_ratio)
		blob = cv2.dnn.blobFromImage(image, 1/255.0, (image.shape[1], image.shape[0]), swapRB=True, crop=False).astype(self.input_types)
		
		if self.framework_type == "trt" :
			output_from_network = self.session.postprocess(blob)
		else :
			output_from_network = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name:  blob})[0].squeeze(axis=0)
		
		if (self.model_type in [ObjectModelType.YOLOV8, ObjectModelType.YOLOV9]) :
			output_from_network = output_from_network.T
		
		output_from_network = self.lite_postprocess(output_from_network)

		# inference output
		for detection in output_from_network:
			if (self.model_type in [ObjectModelType.YOLOV8, ObjectModelType.YOLOV9]) :
				scores = detection[4:]
			else :
				scores = detection[5:]
			classId = np.argmax(scores)
			confidence = float(scores[classId])
			if confidence > self.box_score :
				if (self.model_type not in [ObjectModelType.YOLOV8, ObjectModelType.YOLOV9]) :
					if (detection[4] > 0.4 ) :
						pass
					else :
						continue

				x, y, w, h = detection[0].item(), detection[1].item(), detection[2].item(), detection[3].item() 
				_raw_class_ids.append(classId)
				_raw_class_confs.append(confidence)
				_raw_boxes.append(np.stack([(x - 0.5 * w), (y - 0.5 * h), (x + 0.5 * w), (y + 0.5 * h)], axis=-1))

		transform_boxes = self.convert_boxes_coordinate(_raw_boxes, ratiow, ratioh, padh, padw)
		transform_kpss = self.convert_kpss_coordinate(_raw_kpss, ratiow, ratioh, padh, padw)
		self._object_info = self.get_nms_results(transform_boxes, _raw_class_confs, _raw_class_ids, transform_kpss)

	def DrawDetectedOnFrame(self, frame_show : cv2) -> None:
		tl = 3 or round(0.002 * (frame_show.shape[0] + frame_show.shape[1]) / 2) + 1    # line/font thickness
		if ( len(self._object_info) != 0 )  :
			for box, kpss in self._object_info:
				ymin, xmin, ymax, xmax, label = box
				if (len(kpss) != 0) :
					for kp in kpss :
						cv2.circle(frame_show,  kp, 1, (255, 255, 255), thickness=-1)
				c1, c2 = (xmin, ymin), (xmax, ymax)        
				tf = max(tl - 1, 1)  # font thickness
				t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
				c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

				if (label != 'unknown') :
					cv2.rectangle(frame_show, c1, c2, hex_to_rgb(self.colors_dict[label]), -1, cv2.LINE_AA)
					self.cornerRect(frame_show, box, colorR= hex_to_rgb(self.colors_dict[label]), colorC= hex_to_rgb(self.colors_dict[label]))
				else :
					cv2.rectangle(frame_show, c1, c2, (0, 0, 0), -1, cv2.LINE_AA)
					self.cornerRect(frame_show, box, colorR= (0, 0, 0), colorC= (0, 0, 0) )
				cv2.putText(frame_show, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, tl / 3, (255, 255, 255), 2)
		


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

	get_colors = list(map(lambda i:"#" +"%06x" % random.randint(0, 0xFFFFFF),range(len(network.class_names)) ))
	colors_dict = dict(zip(list(network.class_names), get_colors))

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

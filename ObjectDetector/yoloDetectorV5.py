import os
import cv2
import sys
import onnx
import random
import logging
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda

try :
	from utils import hex_to_rgb
except :
	from ObjectDetector.utils import hex_to_rgb

class YoloLiteParameters():
	def __init__(self, input_shape, num_classes):
		anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
		self.nl = len(anchors)
		self.na = len(anchors[0]) // 2
		self.no = num_classes + 5
		self.grid = [np.zeros(1)] * self.nl
		self.stride = np.array([8., 16., 32.])
		self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
		self.input_shape = input_shape

	def _make_grid(self, nx=20, ny=20):
		xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
		return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

	def postprocess(self, outs):
		row_ind = 0
		for i in range(self.nl):
			h, w = int(self.input_shape[0] / self.stride[i]), int(self.input_shape[1] / self.stride[i])
			length = int(self.na * h * w)
			if self.grid[i].shape[2:4] != (h, w):
				self.grid[i] = self._make_grid(w, h)

			outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
				self.grid[i], (self.na, 1))) * int(self.stride[i])
			outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
				self.anchor_grid[i], h * w, axis=0)
			row_ind += length
		return outs

class TensorRTParameters():

	def __init__(self, engine_file_path, num_classes):
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

		self.context =  self._create_context(engine)
		self.host_inputs, self.cuda_inputs, self.host_outputs, self.cuda_outputs, self.bindings = self._allocate_buffers(engine)
		self.input_shapes = engine.get_binding_shape(0)[-2:]

		# Store
		self.stream = stream
		self.engine = engine

	def _allocate_buffers(self, engine):
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

	def _create_context(self, engine):
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

		return np.reshape(trt_outputs, (-1, self.num_classes+5))


class YoloDetector(object):
	_defaults = {
		"model_path": './models/yolov5n-coco.onnx',
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

	def __init__(self, **kwargs):
		self.__dict__.update(self._defaults) # set up default values
		self.__dict__.update(kwargs) # and update with user overrides
		self.keep_ratio = False
		self.lite =  False
		
		classes_path = os.path.expanduser(self.classes_path)
		if (os.path.isfile(classes_path) is False):
			print( classes_path + " is not exist.")
			raise Exception("%s is not exist." % classes_path)
		self._get_class(classes_path)

		model_path = os.path.expanduser(self.model_path)
		if (os.path.isfile(model_path) is False):
			print( model_path + " is not exist.")
			raise Exception("%s is not exist." % model_path)
		assert model_path.endswith(('.onnx', '.trt')), 'Onnx/TensorRT Parameters must be a .onnx/.trt file.'

		if model_path.endswith('.trt') :
			self.framework_type = "trt"
			self._load_model_tensorrt(model_path) 
		else :
			self.framework_type = "onnx"
			self._load_model_onnxruntime_version(model_path)
		if ("lite" in model_path.lower()) :
			self.lite = True
		print(f'YoloDetector Type : [{self.framework_type}] || Version : [{self.providers}]')
		self.liteParams = YoloLiteParameters(self.input_shapes, len(self.class_names))

	def _get_class(self, classes_path):
		with open(classes_path) as f:
			class_names = f.readlines()
		self.class_names = [c.strip() for c in class_names]
		get_colors = list(map(lambda i:"#" +"%06x" % random.randint(0, 0xFFFFFF),range(len(self.class_names)) ))
		self.colors_dict = dict(zip(list(self.class_names), get_colors))

	def _get_onnx_model_shape(self, model_path):
		model = onnx.load(model_path)
		try:
			onnx.checker.check_model(model)
		except onnx.checker.ValidationError as e:
			print('The model is invalid: %s' % e)
			sys.exit(0)
		else:
			self.input_shapes = tuple(np.array([[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in model.graph.input]).flatten())[-2:]

	def _load_model_onnxruntime_version(self, model_path) :
		self._get_onnx_model_shape(model_path)
		if  ort.get_device() == 'GPU' and 'CUDAExecutionProvider' in  ort.get_available_providers():  # gpu 
			self.providers = 'CUDAExecutionProvider'
		else :
			self.providers = 'CPUExecutionProvider'
		self.session = ort.InferenceSession(model_path, providers= [self.providers] )
		self.providers = 'CUDAExecutionProvider'

	def _load_model_tensorrt(self, model_path) :
		self.session = TensorRTParameters(model_path, len(self.class_names))
		self.input_shapes = self.session.input_shapes
		self.providers = 'CUDAExecutionProvider'

	def resize_image_format(self, srcimg, frame_resize):
		padh, padw, newh, neww = 0, 0, frame_resize[0], frame_resize[1]
		if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
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

	def adjust_boxes_ratio(self, bounding_box, ratio, stretch_type) :
		""" Adjust the aspect ratio of the box according to the orientation """
		xmin, ymin, width, height = bounding_box 
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

	def get_kpss_coordinate(self, kpss, ratiow, ratioh, padh, padw ) :
		if (kpss != []) :
			kpss = np.vstack(kpss)
			kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
			kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh
		return kpss

	def get_boxes_coordinate(self, bounding_boxes, ratiow, ratioh, padh, padw ) :
		if (bounding_boxes != []) :
			bounding_boxes = np.vstack(bounding_boxes)
			bounding_boxes[:, 2:4] = bounding_boxes[:, 2:4] - bounding_boxes[:, 0:2]
			bounding_boxes[:, 0] = (bounding_boxes[:, 0] - padw) * ratiow
			bounding_boxes[:, 1] = (bounding_boxes[:, 1] - padh) * ratioh
			bounding_boxes[:, 2] = bounding_boxes[:, 2] * ratiow
			bounding_boxes[:, 3] = bounding_boxes[:, 3] * ratioh
		return bounding_boxes

	def get_nms_results(self, bounding_boxes, confidences, class_ids, kpss, score, iou, priority=False):
		results = []
		nms_results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, score, iou) 
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
				
				bounding_box = bounding_boxes[i]
				bounding_box = self.adjust_boxes_ratio(bounding_box, None, None)

				xmin, ymin, xmax, ymax = list(map(int, bounding_box))
				results.append(([ymin, xmin, ymax, xmax, predicted_class], kpsslist))
		if (priority and len(results) > 0) :
			results = [results[0]]
		return results

	def DetectFrame(self, srcimg, test = None,frame_resize=None) :
		kpss = []
		class_ids = []
		confidences = []
		bounding_boxes = []
		score = float(self.box_score)
		iou = float(self.box_nms_iou)

		if (frame_resize == None) :
			model_size = self.input_shapes
		else :
			model_size = frame_resize

		image, newh, neww, ratioh, ratiow, padh, padw = self.resize_image_format(srcimg, model_size)
		blob = cv2.dnn.blobFromImage(image, 1/255.0, (image.shape[1], image.shape[0]), swapRB=True, crop=False)

		if self.framework_type == "trt" :
			output_from_network = self.session.postprocess(blob)
		else :
			output_from_network = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name:  blob})[0].squeeze(axis=0)
		
		if self.lite :
			output_from_network = self.liteParams.postprocess(output_from_network)

		# inference output
		for detection in output_from_network:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = scores[classId]
			if confidence > score and detection[4] > 0.4:
				x, y, w, h = detection[0].item(), detection[1].item(), detection[2].item(), detection[3].item() 
				class_ids.append(classId)
				confidences.append(float(confidence))
				bounding_boxes.append(np.stack([(x - 0.5 * w), (y - 0.5 * h), (x + 0.5 * w), (y + 0.5 * h)], axis=-1))

		bounding_boxes = self.get_boxes_coordinate( bounding_boxes, ratiow, ratioh, padh, padw)
		kpss = self.get_kpss_coordinate(kpss, ratiow, ratioh, padh, padw)
		self.object_info = self.get_nms_results(bounding_boxes, confidences, class_ids, kpss, score, iou)


	def DrawDetectedOnFrame(self, frame_show) :
		tl = 3 or round(0.002 * (frame_show.shape[0] + frame_show.shape[1]) / 2) + 1  # line/font thickness
		if ( len(self.object_info) != 0 )  :
			for box, kpss in self.object_info:
				ymin, xmin, ymax, xmax, label = box
				if (len(kpss) != 0) :
					for kp in kpss :
						cv2.circle(frame_show,  kp, 1, (255, 255, 255), thickness=-1)
				c1, c2 = (xmin, ymin), (xmax, ymax)        
				tf = max(tl - 1, 1)  # font thickness
				t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
				c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
				cv2.rectangle(frame_show, c1, c2, hex_to_rgb(self.colors_dict[label]), -1, cv2.LINE_AA)
				cv2.putText(frame_show, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
				cv2.rectangle(frame_show, (xmin, ymin), (xmax, ymax), hex_to_rgb(self.colors_dict[label]), 2)


		
if __name__ == "__main__":
	import time
	import sys

	capture = cv2.VideoCapture(r"./temp/歐森隆20210923-Lobby-1.avi")
	config = {
		"model_path": 'models/yolov5m-coco.trt',
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

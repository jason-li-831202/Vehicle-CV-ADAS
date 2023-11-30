import onnxruntime
import scipy.special
import cv2
import time, os
import numpy as np
from typing import Tuple
try :
	from ultrafastLaneDetector.utils import TensorRTBase, LaneModelType, OffsetType, lane_colors
except :
	from .utils import TensorRTBase, LaneModelType, OffsetType, lane_colors

class ModelConfig():

	def __init__(self, model_type):

		if model_type == LaneModelType.UFLD_TUSIMPLE:
			self.init_tusimple_config()
		else:
			self.init_culane_config()
		self.num_lanes = 4

	def init_tusimple_config(self):
		self.img_w = 1280
		self.img_h = 720
		self.griding_num = 100
		self.cls_num_per_lane = 56
		self.row_anchor = np.linspace(64, 284, self.cls_num_per_lane)
		
	def init_culane_config(self):
		self.img_w = 1640
		self.img_h = 590
		self.griding_num = 200
		self.cls_num_per_lane = 18
		self.row_anchor = np.linspace(121, 131, self.cls_num_per_lane)

class TensorRTEngine(TensorRTBase):

	def __init__(self, engine_file_path, cfg):
		super(TensorRTEngine, self).__init__(engine_file_path, cfg)

	def get_tensorrt_input_shape(self):
		return self.engine.get_binding_shape(0)

	def get_tensorrt_output_shape(self):
		return (1, 1, self.cfg.cls_num_per_lane, self.cfg.num_lanes)

	def tensorrt_inference(self, input_tensor):
		host_outputs = self.inference(input_tensor)
		# Here we use the first row of output in that batch_size = 1
		trt_outputs = host_outputs[0]
		return np.reshape(trt_outputs, (1, -1, self.cfg.cls_num_per_lane, self.cfg.num_lanes) )

class OnnxEngine():

	def __init__(self, onnx_file_path):
		if (onnxruntime.get_device() == 'GPU') :
			self.session = onnxruntime.InferenceSession(onnx_file_path, providers=['CUDAExecutionProvider'])
		else :
			self.session = onnxruntime.InferenceSession(onnx_file_path)
		self.providers = self.session.get_providers()
		self.framework_type = "onnx"

	def get_onnx_input_shape(self):
		return self.session.get_inputs()[0].shape

	def get_onnx_output_shape(self):
		output_shape = [output.shape for output in self.session.get_outputs()]
		output_names = [output.name for output in self.session.get_outputs()]
		if (len(output_names) != 1) :
			raise Exception("Output dims is error, please check model. load %d channels not match 1." % len(self.output_names))
		return output_shape[0], output_names
	
	def onnx_inference(self, input_tensor):
		input_name = self.session.get_inputs()[0].name
		output_name = self.session.get_outputs()[0].name
		output = self.session.run([output_name], {input_name: input_tensor})
		return output

class UltrafastLaneDetector(TensorRTEngine, OnnxEngine):
	_defaults = {
		"model_path": "models/tusimple_18.onnx",
		"model_type" : LaneModelType.UFLD_TUSIMPLE,
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

	def __init__(self, model_path : str = None, model_type : LaneModelType = None, logger = None):
		if (None in [model_path, model_type]) :
			self.__dict__.update(self._defaults) # set up default values
		else :
			self.model_path, self.model_type = model_path, model_type

		self.logger = logger
		self.draw_area_points = []
		self.draw_area = False
		
		# Load model configuration based on the model type
		if ( self.model_type not in [LaneModelType.UFLD_TUSIMPLE, LaneModelType.UFLD_CULANE]) :
			if (self.logger) :
				self.logger.error("UltrafastLaneDetector can't use %s type." % self.model_type.name)
			raise Exception("UltrafastLaneDetector can't use %s type." % self.model_type.name)
		self.cfg = ModelConfig(self.model_type)

		# Initialize model
		self._initialize_model(self.model_path, self.cfg)
		
	def _initialize_model(self, model_path : str, cfg : ModelConfig) -> None:
		if (self.logger) :
			self.logger.debug("model path: %s." % model_path)
		if not os.path.isfile(model_path):
			raise Exception("The model path [%s] can't not found!" % model_path)
		if model_path.endswith('.trt') :
			TensorRTEngine.__init__(self, model_path, cfg)
		else :
			OnnxEngine.__init__(self, self.model_path)

		# Get model info
		self.getModel_input_details()
		self.getModel_output_details()
		if (self.logger) :
			self.logger.info(f'UfldDetector Type : [{self.framework_type}] || Version : {self.providers}')

	def __prepare_input(self, image : cv2) -> np.ndarray :
		img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		self.img_height, self.img_width, self.img_channels = img.shape

		# Input values should be from -1 to 1 with a size of 288 x 800 pixels
		img_input = cv2.resize(img, (self.input_width,self.input_height)).astype(np.float32)
		
		# Scale input pixel values to -1 to 1
		mean=[0.485, 0.456, 0.406]
		std=[0.229, 0.224, 0.225]
		
		img_input = ((img_input/ 255.0 - mean) / std)
		img_input = img_input.transpose(2, 0, 1)
		img_input = img_input[np.newaxis,:,:,:]        

		return img_input.astype(np.float32)

	@staticmethod
	def __process_output(output, cfg : ModelConfig) -> np.ndarray:		
		# Parse the output of the model

		processed_output = np.squeeze(output[0])
		# print(processed_output.shape)
		# print(np.min(processed_output), np.max(processed_output))
		# print(processed_output.reshape((1,-1)))
		processed_output = processed_output[:, ::-1, :]
		prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
		idx = np.arange(cfg.griding_num) + 1
		idx = idx.reshape(-1, 1, 1)
		loc = np.sum(prob * idx, axis=0)
		processed_output = np.argmax(processed_output, axis=0)
		loc[processed_output == cfg.griding_num] = 0
		processed_output = loc


		col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
		col_sample_w = col_sample[1] - col_sample[0]

		lanes_points = []
		lanes_detected = []

		max_lanes = processed_output.shape[1]
		for lane_num in range(max_lanes):
			lane_points = []
			# Check if there are any points detected in the lane
			if np.sum(processed_output[:, lane_num] != 0) > 2:
				lanes_detected.append(True)

				# Process each of the points for each lane
				for point_num in range(processed_output.shape[0]):
					if processed_output[point_num, lane_num] > 0:
						lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1, int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/288)) - 1 ]
						lane_points.append(lane_point)
			else:
				lanes_detected.append(False)

			lanes_points.append(lane_points)
		return np.array(lanes_points, dtype=object), np.array(lanes_detected, dtype=object)
	
	@staticmethod
	def __adjust_lanes_points(left_lanes_points : list, right_lanes_points : list, image_height : list) -> Tuple[list, list]:
		if (len(left_lanes_points[1]) != 0 ) :
			leftx, lefty  = list(zip(*left_lanes_points))
		else :
			return left_lanes_points, right_lanes_points
		if (len(right_lanes_points) != 0 ) :
			rightx, righty  = list(zip(*right_lanes_points))
		else :
			return left_lanes_points, right_lanes_points

		if len(lefty) > 10:
			left_fit = np.polyfit(lefty, leftx, 2)
		if len(righty) > 10:
			right_fit = np.polyfit(righty, rightx, 2)

		# Generate x and y values for plotting
		maxy = image_height - 1
		miny = image_height // 3
		if len(lefty):
			maxy = max(maxy, np.max(lefty))
			miny = min(miny, np.min(lefty))

		if len(righty):
			maxy = max(maxy, np.max(righty))
			miny = min(miny, np.min(righty))

		ploty = np.linspace(miny, maxy, image_height)

		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		# Visualization
		fix_left_lanes_points = []
		fix_right_lanes_points = []
		for i, y in enumerate(ploty):
			l = int(left_fitx[i])
			r = int(right_fitx[i])
			y = int(y)
			if (y >= min(lefty)) :
				fix_left_lanes_points.append((l, y))
			if (y >= min(righty)) :
				fix_right_lanes_points.append((r, y))
				# cv2.line(out_img, (l, y), (r, y), (0, 255, 0))
		return fix_left_lanes_points, fix_right_lanes_points

	def getModel_input_details(self) -> None :
		if (self.framework_type == "trt") :
			self.input_shape = self.get_tensorrt_input_shape()
		else :
			self.input_shape = self.get_onnx_input_shape()
		self.channes, self.input_height, self.input_width = self.input_shape[1:]
		if (self.logger) : 
			if (self.cfg.img_h == self.input_height and self.cfg.img_w == self.input_width) :
				self.logger.info(f"UfldDetector Input Shape : {self.input_shape} ")
			else :
				self.logger.war(f"UfldDetector Model Iuput Shape {self.input_height, self.input_width} not equal cfg Input Shape {self.cfg.img_h, self.cfg.img_w}")

	def getModel_output_details(self) -> None :
		if (self.framework_type == "trt") :
			self.output_shape = self.get_tensorrt_output_shape()
		else :
			self.output_shape, self.output_names = self.get_onnx_output_shape()

		self.num_points = self.output_shape[1]
		self.num_anchors = self.output_shape[2]
		self.num_lanes = self.output_shape[3]

	def DetectFrame(self, image : cv2) -> None:
		input_tensor = self.__prepare_input(image)

		# Perform inference on the image
		output = self.tensorrt_inference(input_tensor) if (self.framework_type == "trt") else self.onnx_inference(input_tensor)

		# Process output data
		self.lanes_points, self.lanes_detected = self.__process_output(output, self.cfg)

	def DrawDetectedOnFrame(self, image : cv2, type : OffsetType = OffsetType.UNKNOWN) -> None:
		for lane_num,lane_points in enumerate(self.lanes_points):

			if ( lane_num==1 and type == OffsetType.RIGHT) :
				color = (0, 0, 255)
			elif (lane_num==2 and type == OffsetType.LEFT) :
				color = (0, 0, 255)
			else :
				color = lane_colors[lane_num]

			for lane_point in lane_points:
				cv2.circle(image, (lane_point[0],lane_point[1]), 3, color, -1)

	def DrawAreaOnFrame(self, image : cv2, color : tuple = (255,191,0), adjust_lanes : bool = True) -> None :
		self.draw_area = False
		H, W, _ = image.shape
		# Draw a mask for the current lane
		if(self.lanes_detected != []) :
			if(self.lanes_detected[1] and self.lanes_detected[2]):
				self.draw_area = True
				lane_segment_img = image.copy()

				if (adjust_lanes) :
					left_lanes_points, right_lanes_points = self.__adjust_lanes_points(self.lanes_points[1], self.lanes_points[2], self.img_height)
				else :
					left_lanes_points, right_lanes_points = self.lanes_points[1], self.lanes_points[2]
				self.draw_area_points = [np.vstack((left_lanes_points,np.flipud(right_lanes_points)))]
				
				cv2.fillPoly(lane_segment_img, pts = self.draw_area_points, color =color)
				image[:H,:W,:] = cv2.addWeighted(image, 0.7, lane_segment_img, 0.1, 0)

		if (not self.draw_area) : self.draw_area_points = []

	def AutoDrawLanes(self, image : cv2, draw_points : bool = True, draw_area : bool = True) -> None:
		self.DetectFrame(image)

		if (draw_points) :
			self.DrawDetectedOnFrame(image)

		if (draw_area) :
			self.DrawAreaOnFrame(image, adjust_lanes=False)
		return image




	








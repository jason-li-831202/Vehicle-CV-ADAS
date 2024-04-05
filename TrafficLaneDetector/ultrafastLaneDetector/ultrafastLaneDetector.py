import scipy.special
import cv2
import numpy as np
from typing import Tuple
try :
	import sys
	from ultrafastLaneDetector.utils import LaneModelType, OffsetType, lane_colors
	from ultrafastLaneDetector.LaneDetector import LaneDetectBase
	sys.path.append("..")
	from coreEngine import TensorRTEngine, OnnxEngine
except :
	from .utils import LaneModelType, OffsetType, lane_colors
	from .LaneDetector import LaneDetectBase
	from coreEngine import TensorRTEngine, OnnxEngine

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
		self.row_anchor = [round(value) for value in np.linspace(121, 287, self.cls_num_per_lane)]

class UltrafastLaneDetector(LaneDetectBase):
	_defaults = {
		"model_path": "models/tusimple_18.onnx",
		"model_type" : LaneModelType.UFLD_TUSIMPLE,
	}

	def __init__(self, model_path : str = None, model_type : LaneModelType = None, logger = None):
		LaneDetectBase.__init__(self, logger)
		if (None not in [model_path, model_type]) :
			self.model_path, self.model_type = model_path, model_type

		# Load model configuration based on the model type
		if ( self.model_type not in [LaneModelType.UFLD_TUSIMPLE, LaneModelType.UFLD_CULANE]) :
			if (self.logger) :
				self.logger.error("UltrafastLaneDetector can't use %s type." % self.model_type.name)
			raise Exception("UltrafastLaneDetector can't use %s type." % self.model_type.name)
		self.cfg = ModelConfig(self.model_type)

		# Initialize model
		self._initialize_model(self.model_path)
		
	def _initialize_model(self, model_path : str) -> None:
		if (self.logger) :
			self.logger.debug("model path: %s." % model_path)

		if model_path.endswith('.trt') :
			self.engine = TensorRTEngine(model_path)
		else :
			self.engine = OnnxEngine(model_path)

		if (self.logger) :
			self.logger.info(f'UfldDetector Type : [{self.engine.framework_type}] || Version : {self.engine.providers}')

		# Set model info
		self.set_input_details(self.engine)
		self.set_output_details(self.engine)
		if (len(self.output_names) != 1) :
			raise Exception("Output dims is error, please check model. load %d channels not match 1." % len(self.output_names))
		
	def __prepare_input(self, image : cv2) -> np.ndarray :
		self.h_ratio, self.w_ratio = image.shape[0]/self.cfg.img_h, image.shape[1]/self.cfg.img_w
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

		return img_input.astype(self.input_types)

	def __process_output(self, output, cfg : ModelConfig) -> Tuple[np.ndarray, list]:		
		# Parse the output of the model

		processed_output = np.squeeze(output[0])
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


		col_sample = np.linspace(0, self.input_width - 1, cfg.griding_num)
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
						lane_point = [processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / self.input_width - 1, 
					                  cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num] / self.input_height) - 1 ]
						lane_points.append([int(lane_point[0]*self.w_ratio), int(lane_point[1]*self.h_ratio) ])
			else:
				lanes_detected.append(False)

			lanes_points.append(lane_points)
		return np.array(lanes_points, dtype=object), np.array(lanes_detected, dtype=object)

	def DetectFrame(self, image : cv2, adjust_lanes : bool = True) -> None:
		input_tensor = self.__prepare_input(image)

		# Perform inference on the image
		output = self.engine.engine_inference(input_tensor)

		# Process output data
		self.lane_info.lanes_points, self.lane_info.lanes_status = self.__process_output(output, self.cfg)

		self.adjust_lanes = adjust_lanes
		self._LaneDetectBase__update_lanes_status(self.lane_info.lanes_status)
		self._LaneDetectBase__update_lanes_area(self.lane_info.lanes_points, self.img_height)
		
	def DrawDetectedOnFrame(self, image : cv2, type : OffsetType = OffsetType.UNKNOWN) -> None:
		for lane_num, lane_points in enumerate(self.lane_info.lanes_points):

			if ( lane_num==1 and type == OffsetType.RIGHT) :
				color = (0, 0, 255)
			elif (lane_num==2 and type == OffsetType.LEFT) :
				color = (0, 0, 255)
			else :
				color = lane_colors[lane_num]

			for lane_point in lane_points:
				cv2.circle(image, (lane_point[0], lane_point[1]), 3, color, -1)

	def DrawAreaOnFrame(self, image : cv2, color : tuple = (255,191,0)) -> None :
		H, W, _ = image.shape
		# Draw a mask for the current lane
		if(self.lane_info.area_status):
			lane_segment_img = image.copy()

			cv2.fillPoly(lane_segment_img, pts = [self.lane_info.area_points], color =color)
			image[:H,:W,:] = cv2.addWeighted(image, 0.7, lane_segment_img, 0.1, 0)




	








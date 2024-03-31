import onnxruntime
import cv2
import time, os
import numpy as np
from typing import Tuple
try :
	from ultrafastLaneDetector.utils import TensorRTBase, LaneModelType, OffsetType, lane_colors
except :
	from .utils import TensorRTBase, LaneModelType, OffsetType, lane_colors

def _softmax(x) :
	# Note : 防止 overflow and underflow problem
	x = x - np.max(x, axis=-1, keepdims=True) 
	exp_x = np.exp(x)
	return exp_x/np.sum(exp_x, axis=-1,keepdims=True)

class ModelConfig():

	def __init__(self, model_type):

		if model_type == LaneModelType.UFLDV2_TUSIMPLE:
			self.init_tusimple_config()
		elif model_type == LaneModelType.UFLDV2_CURVELANES :
			self.init_curvelanes_config()
		else :
			self.init_culane_config()
		self.num_lanes = 4

	def init_tusimple_config(self):
		self.img_w = 800
		self.img_h = 320
		self.griding_num = 100
		self.crop_ratio = 0.8
		self.row_anchor = np.linspace(160,710, 56)/720
		self.col_anchor = np.linspace(0,1, 41)

	def init_curvelanes_config(self) :
		self.img_w = 1600
		self.img_h = 800
		self.griding_num = 200
		self.crop_ratio = 0.8
		self.row_anchor = np.linspace(0.4, 1, 72)
		self.col_anchor = np.linspace(0, 1, 81)
	
	def init_culane_config(self):
		self.img_w = 1600
		self.img_h = 320
		self.griding_num = 200
		self.crop_ratio = 0.6
		self.row_anchor = np.linspace(0.42,1, 72)
		self.col_anchor = np.linspace(0,1, 81)

class TensorRTEngine(TensorRTBase):

	def __init__(self, engine_file_path, cfg):
		super(TensorRTEngine, self).__init__(engine_file_path)
		self.cfg = cfg
		self.cuda_dtype = self.dtype

	def get_tensorrt_input_shape(self):
		return self.engine.get_binding_shape(0)

	def get_tensorrt_output_shape(self):
		return self.engine.get_binding_shape(-1)

	def tensorrt_inference(self, input_tensor):
		host_outputs = self.inference(input_tensor)
		# Here we use the first row of output in that batch_size = 1
		trt_outputs = []
		for i, output in enumerate(host_outputs) :
			if i in [0, 2] :
				mat = np.reshape(output, (1, -1, len(self.cfg.row_anchor), self.cfg.num_lanes) )
			else :
				mat = np.reshape(output, (1, -1, len(self.cfg.col_anchor), self.cfg.num_lanes) )
			trt_outputs.append(mat)

		return trt_outputs

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
		if (len(output_names) != 4) :
			raise Exception("Output dims is error, please check model. load %d channels not match 4." % len(self.output_names))
		return output_shape, output_names
	
	def onnx_inference(self, input_tensor):
		input_name = self.session.get_inputs()[0].name
		output_names = [output.name for output in self.session.get_outputs()]
		output = self.session.run(output_names, {input_name: input_tensor})

		return output

class UltrafastLaneDetectorV2(TensorRTEngine, OnnxEngine):
	_defaults = {
		"model_path": "models/culane_res18.onnx",
		"model_type" : LaneModelType.UFLDV2_TUSIMPLE,
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
		if ( self.model_type not in [LaneModelType.UFLDV2_TUSIMPLE, LaneModelType.UFLDV2_CULANE]) :
			if (self.logger) :
				self.logger.error("UltrafastLaneDetectorV2 can't use %s type." % self.model_type.name)
			raise Exception("UltrafastLaneDetectorV2 can't use %s type." % self.model_type.name)
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
			self.logger.info(f'UfldDetector Type : [{self.framework_type}] || Version : [{self.providers}]')

	def __prepare_input(self, image : cv2) -> np.ndarray :
		img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		self.img_height, self.img_width, self.img_channels = img.shape

		# Input values should be from -1 to 1 with a size of 288 x 800 pixels
		new_size = ( self.input_width, int(self.input_height/self.cfg.crop_ratio))
		img_input = cv2.resize(img, new_size).astype(np.float32)
		img_input = img_input[-self.input_height:, :, :]
		# Scale input pixel values to -1 to 1
		mean=[0.485, 0.456, 0.406]
		std=[0.229, 0.224, 0.225]
		
		img_input = ((img_input/ 255.0 - mean) / std)
		img_input = img_input.transpose(2, 0, 1)
		img_input = img_input[np.newaxis,:,:,:]        

		return img_input.astype(self.input_types)

	@staticmethod
	def __process_output(output, cfg : ModelConfig, local_width :int = 1, original_image_width : int = 1640, original_image_height : int = 590) -> np.ndarray:

		# output = np.array(output, dtype=np.float32) 
		output = {"loc_row" : output[0], 'loc_col' : output[1], "exist_row" : output[2], "exist_col" : output[3]}
		# print(output["loc_row"].shape)
		# print(output["exist_row"].shape)
		# print(output["loc_col"].shape)
		# print(output["exist_col"].shape)

		batch_size, num_grid_row, num_cls_row, num_lane_row = output['loc_row'].shape
		batch_size, num_grid_col, num_cls_col, num_lane_col = output['loc_col'].shape

		max_indices_row = output['loc_row'].argmax(1)
		# n , num_cls, num_lanes
		valid_row = output['exist_row'].argmax(1)
		# n, num_cls, num_lanes

		max_indices_col = output['loc_col'].argmax(1)
		# n , num_cls, num_lanes
		valid_col = output['exist_col'].argmax(1)
		# n, num_cls, num_lanes

		output['loc_row'] = output['loc_row']
		output['loc_col'] = output['loc_col']
		row_lane_idx = [1,2]
		col_lane_idx = [0,3]

		# Parse the output of the model
		lanes_points = {"left-side" : [], "left-ego" : [] , "right-ego" : [], "right-side" : []}
		# lanes_detected = []
		lanes_detected =  {"left-side" : False, "left-ego" : False , "right-ego" : False, "right-side" : False}
		for i in row_lane_idx:
			tmp = []
			if valid_row[0,:,i].sum() > num_cls_row / 2:
				for k in range(valid_row.shape[1]):
					if valid_row[0,k,i]:
						all_ind = list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1))
						out_tmp = ( _softmax(output['loc_row'][0,all_ind,k,i]) * list(map(float, all_ind))).sum() + 0.5
						out_tmp = out_tmp / (num_grid_row-1) * original_image_width
						tmp.append((int(out_tmp), int(cfg.row_anchor[k] * original_image_height)))
				if (i == 1) :
					lanes_points["left-ego"].extend(tmp)
					if (len(tmp) > 2) :
						lanes_detected["left-ego"] = True
				else :
					lanes_points["right-ego"].extend(tmp)
					if (len(tmp) > 2) :
						lanes_detected["right-ego"] = True

		for i in col_lane_idx:
			tmp = []
			if valid_col[0,:,i].sum() > num_cls_col / 4:
				for k in range(valid_col.shape[1]):
					if valid_col[0,k,i]:
						all_ind = list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1))
						out_tmp = ( _softmax(output['loc_col'][0,all_ind,k,i]) * list(map(float, all_ind))).sum() + 0.5
						out_tmp = out_tmp / (num_grid_col-1) * original_image_height
						tmp.append((int(cfg.col_anchor[k] * original_image_width), int(out_tmp)))
				if (i == 0) :
					lanes_points["left-side" ].extend(tmp)
					if (len(tmp) > 2) :
						lanes_detected["left-side"] = True
				else :
					lanes_points["right-side"].extend(tmp)
					if (len(tmp) > 2) :
						lanes_detected["right-side"] = True
		return np.array(list(lanes_points.values()), dtype="object"), list(lanes_detected.values())

	@staticmethod
	def __adjust_lanes_points(left_lanes_points : list, right_lanes_points : list, image_height : list) -> Tuple[list, list]:
		# 多项式拟合
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

	@staticmethod
	def ___check_lanes_area(lanes_detected : list) -> bool :
		if(lanes_detected != []) :
			if(lanes_detected[1] and lanes_detected[2]):
				return True
		return False

	def getModel_input_details(self) -> None :
		if (self.framework_type == "trt") :
			self.input_shape = self.get_tensorrt_input_shape()
			self.input_types = self.cuda_dtype
		else :
			self.input_shape = self.get_onnx_input_shape()
			self.input_types = np.float16 if 'float16' in self.session.get_inputs()[0].type else np.float32

		self.channes, self.input_height, self.input_width = self.input_shape[1:]
		if (self.logger) : 
			if (self.cfg.img_h == self.input_height and self.cfg.img_w == self.input_width) :
				self.logger.info(f"UfldDetector Input Shape : {self.input_shape} || dtype : {self.input_types}")
			else :
				self.logger.war(f"UfldDetector Model Iuput Shape {self.input_height, self.input_width} not equal cfg Input Shape {self.cfg.img_h, self.cfg.img_w}")

	def getModel_output_details(self) -> None :
		if (self.framework_type == "trt") :
			self.output_shape = self.get_tensorrt_output_shape()
		else :
			self.output_shape, self.output_names = self.get_onnx_output_shape()
			
	def DetectFrame(self, image : cv2) -> None:
		input_tensor = self.__prepare_input(image)

		# Perform inference on the image
		output = self.tensorrt_inference(input_tensor) if (self.framework_type == "trt") else self.onnx_inference(input_tensor)

		# Process output data
		self.lanes_points, self.lanes_detected = self.__process_output(output, self.cfg, original_image_width =  self.img_width, original_image_height = self.img_height)

		self.draw_area = self.___check_lanes_area(self.lanes_detected)

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
		H, W, _ = image.shape
		self.draw_area_points = []
		# Draw a mask for the current lane
		if(self.draw_area):
			lane_segment_img = image.copy()

			if (adjust_lanes) :
				left_lanes_points, right_lanes_points = self.__adjust_lanes_points(self.lanes_points[1], self.lanes_points[2], self.img_height)
			else :
				left_lanes_points, right_lanes_points = self.lanes_points[1], self.lanes_points[2]
			self.draw_area_points = [np.vstack((left_lanes_points,np.flipud(right_lanes_points)))]
			
			cv2.fillPoly(lane_segment_img, pts = self.draw_area_points, color =color)
			image[:H,:W,:] = cv2.addWeighted(image, 0.7, lane_segment_img, 0.1, 0)

	def AutoDrawLanes(self, image : cv2, draw_points : bool = True, draw_area : bool = True) -> None:
		self.DetectFrame(image)

		if (draw_points) :
			self.DrawDetectedOnFrame(image)

		if (draw_area) :
			self.DrawAreaOnFrame(image, adjust_lanes=True)
		return image


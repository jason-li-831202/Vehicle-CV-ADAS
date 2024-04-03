import onnxruntime
import cv2
import time, os
import numpy as np
from typing import Tuple
try :
	from ultrafastLaneDetector.utils import LaneDetectBase, EngineBase, TensorRTBase, LaneModelType, OffsetType, lane_colors
	from ultrafastLaneDetector.LaneDetector import LaneDetectBase
except :
	from .utils import EngineBase, TensorRTBase, LaneModelType, OffsetType, lane_colors
	from .LaneDetector import LaneDetectBase

def _softmax(x) :
	# Note : 防止 overflow and underflow problem
	x = x - np.max(x, axis=-1, keepdims=True) 
	exp_x = np.exp(x)
	return exp_x/np.sum(exp_x, axis=-1, keepdims=True)

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

class TensorRTEngine(EngineBase, TensorRTBase):

	def __init__(self, engine_file_path, cfg):
		TensorRTBase.__init__(self, engine_file_path)
		self.cfg = cfg
		self.engine_dtype = self.dtype

	def get_engine_input_shape(self):
		return self.engine.get_binding_shape(0)

	def get_engine_output_shape(self):
		# Get the number of bindings
		num_bindings = self.engine.num_bindings

		# Get the output names
		output_names = []
		for i in range(num_bindings):
			if self.engine.binding_is_input(i):
				continue
			output_names.append(self.engine.get_binding_name(i))
		if (len(output_names) != 4) :
			raise Exception("Output dims is error, please check model. load %d channels not match 4." % len(output_names))
		return self.engine.get_binding_shape(-1), output_names

	def engine_inference(self, input_tensor):
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

class OnnxEngine(EngineBase):

	def __init__(self, onnx_file_path):
		if (onnxruntime.get_device() == 'GPU') :
			self.session = onnxruntime.InferenceSession(onnx_file_path, providers=['CUDAExecutionProvider'])
		else :
			self.session = onnxruntime.InferenceSession(onnx_file_path)
		self.providers = self.session.get_providers()
		self.engine_dtype = np.float16 if 'float16' in self.session.get_inputs()[0].type else np.float32
		self.framework_type = "onnx"

	def get_engine_input_shape(self):
		return self.session.get_inputs()[0].shape

	def get_engine_output_shape(self):
		output_shape = [output.shape for output in self.session.get_outputs()]
		output_names = [output.name for output in self.session.get_outputs()]
		if (len(output_names) != 4) :
			raise Exception("Output dims is error, please check model. load %d channels not match 4." % len(output_names))
		return output_shape, output_names
	
	def engine_inference(self, input_tensor):
		input_name = self.session.get_inputs()[0].name
		output_names = [output.name for output in self.session.get_outputs()]
		output = self.session.run(output_names, {input_name: input_tensor})

		return output

class UltrafastLaneDetectorV2(LaneDetectBase):
	_defaults = {
		"model_path": "models/culane_res18.onnx",
		"model_type" : LaneModelType.UFLDV2_TUSIMPLE,
	}

	def __init__(self, model_path : str = None, model_type : LaneModelType = None, logger = None):
		LaneDetectBase.__init__(self, logger)
		if (None not in [model_path, model_type]) :
			self.model_path, self.model_type = model_path, model_type

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
			self.engine = TensorRTEngine(model_path, cfg)
		else :
			self.engine = OnnxEngine(model_path)

		if (self.logger) :
			self.logger.info(f'UfldDetectorV2 Type : [{self.engine.framework_type}] || Version : [{self.engine.providers}]')
		# Set model info
		self.set_input_details(self.engine)
		self.set_output_details(self.engine)

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

	def __process_output(self, output, cfg : ModelConfig, local_width :int = 1) -> Tuple[np.ndarray, list]:
		original_image_width = self.img_width
		original_image_height = self.img_height
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
		for lane_num,lane_points in enumerate(self.lane_info.lanes_points):
			
			if ( lane_num==1 and type == OffsetType.RIGHT) :
				color = (0, 0, 255)
			elif (lane_num==2 and type == OffsetType.LEFT) :
				color = (0, 0, 255)
			else :
				color = lane_colors[lane_num]

			for lane_point in lane_points:
				cv2.circle(image, (lane_point[0],lane_point[1]), 3, color, -1)

	def DrawAreaOnFrame(self, image : cv2, color : tuple = (255,191,0)) -> None :
		H, W, _ = image.shape
		# Draw a mask for the current lane
		if(self.lane_info.area_status):
			lane_segment_img = image.copy()

			cv2.fillPoly(lane_segment_img, pts = [self.lane_info.area_points], color =color)
			image[:H,:W,:] = cv2.addWeighted(image, 0.7, lane_segment_img, 0.1, 0)


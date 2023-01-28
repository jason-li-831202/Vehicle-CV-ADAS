import onnxruntime
import scipy.special
import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
try :
	from ultrafastLaneDetector.utils import LaneModelType, OffsetType, lane_colors, tusimple_row_anchor, culane_row_anchor
except :
	from .utils import LaneModelType, OffsetType, lane_colors, tusimple_row_anchor, culane_row_anchor

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
		self.row_anchor = tusimple_row_anchor
		self.griding_num = 100
		self.cls_num_per_lane = 56

	def init_culane_config(self):
		self.img_w = 1640
		self.img_h = 590
		self.row_anchor = culane_row_anchor
		self.griding_num = 200
		self.cls_num_per_lane = 18

class TensorRTEngine():

	def __init__(self, engine_file_path, cfg):
		self.cfg = cfg
		# Create a Context on this device,
		cuda.init()
		device = cuda.Device(0)
		self.cuda_driver_context = device.make_context()

		stream = cuda.Stream()
		TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
		runtime = trt.Runtime(TRT_LOGGER)
		# Deserialize the engine from file
		with open(engine_file_path, "rb") as f:
			engine = runtime.deserialize_cuda_engine(f.read())

		self.context =  self._create_context(engine)
		self.host_inputs, self.cuda_inputs, self.host_outputs, self.cuda_outputs, self.bindings = self._allocate_buffers(engine)

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

	def  get_tensorrt_input_shape(self):
		return self.engine.get_binding_shape(0)

	def  get_tensorrt_output_shape(self):
		return (1, 1, self.cfg.cls_num_per_lane, self.cfg.num_lanes)

	def inference(self, input_tensor):
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
		np.copyto(host_inputs[0], input_tensor.ravel())
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
		return np.reshape(trt_outputs, (1, -1, self.cfg.cls_num_per_lane, self.cfg.num_lanes) )

class OnnxEngine():

	def __init__(self, onnx_file_path):
		if (onnxruntime.get_device() == 'GPU') :
			self.session = onnxruntime.InferenceSession(onnx_file_path, providers=['CUDAExecutionProvider'])
		else :
			self.session = onnxruntime.InferenceSession(onnx_file_path)
		self.providers = self.session.get_providers()

	def  get_onnx_input_shape(self):
		return self.session.get_inputs()[0].shape

	def  get_onnx_output_shape(self):
		output_shape = [output.shape for output in self.session.get_outputs()]
		output_names = [output.name for output in self.session.get_outputs()]
		if (len(output_names) != 1) :
			raise Exception("Output dims is error, please check model. load %d channels not match 1." % len(self.output_names))
		return output_shape[0], output_names
	
	def inference(self, input_tensor):
		input_name = self.session.get_inputs()[0].name
		output_name = self.session.get_outputs()[0].name
		output = self.session.run([output_name], {input_name: input_tensor})
		return output

class UltrafastLaneDetector():
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

	def __init__(self, model_path=None, model_type=None, logger=None):
		if (None in [model_path, model_type]) :
			self.__dict__.update(self._defaults) # set up default values
		else :
			self.model_path = model_path
			self.model_type = model_type

		self.logger = logger
		if ( self.model_type not in [LaneModelType.UFLD_TUSIMPLE, LaneModelType.UFLD_CULANE]) :
			if (self.logger) :
				self.logger.error("UltrafastLaneDetector can use %s type." % self.model_type.name)
			raise Exception("UltrafastLaneDetector can use %s type." % self.model_type.name)
		self.fps = 0
		self.timeLastPrediction = time.time()
		self.frameCounter = 0
		self.draw_area_points = []
		self.draw_area = False
		
		# Load model configuration based on the model type
		self.cfg = ModelConfig(self.model_type)

		# Initialize model
		self._initialize_model(self.model_path, self.cfg)
		

	def _initialize_model(self, model_path, cfg):
		if model_path.endswith('.trt') :
			self.framework_type = "trt"
			self.infer = TensorRTEngine(model_path, cfg)
			self.providers = 'CUDAExecutionProvider'
		else :
			self.framework_type = "onnx"
			self.infer = OnnxEngine(model_path)
			self.providers = self.infer.providers

		# Get model info
		self.getModel_input_details()
		self.getModel_output_details()
		self.logger.info(f'UfldDetector Type : [{self.framework_type}] || Version : {self.providers}')

	def getModel_input_details(self):
		if (self.framework_type == "trt") :
			self.input_shape = self.infer.get_tensorrt_input_shape()
		else :
			self.input_shape = self.infer.get_onnx_input_shape()
		self.channes = self.input_shape[2]
		self.input_height = self.input_shape[2]
		self.input_width = self.input_shape[3]

	def getModel_output_details(self):
		if (self.framework_type == "trt") :
			self.output_shape = self.infer.get_tensorrt_output_shape()
		else :
			self.output_shape, self.output_names = self.infer.get_onnx_output_shape()
		self.num_points = self.output_shape[1]
		self.num_anchors = self.output_shape[2]
		self.num_lanes = self.output_shape[3]

	def prepare_input(self, image):
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

	def DetectFrame(self, image) :
		input_tensor = self.prepare_input(image)
		# Perform inference on the image
		output = self.infer.inference(input_tensor)

		# Process output data
		self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)

	def DrawDetectedOnFrame(self, image, type=OffsetType.UNKNOWN) :
		for lane_num,lane_points in enumerate(self.lanes_points):
			if ( lane_num==1 and type == OffsetType.RIGHT) :
				color = (0, 0, 255)
			elif (lane_num==2 and type == OffsetType.LEFT) :
				color = (0, 0, 255)
			else :
				color = lane_colors[lane_num]
			for lane_point in lane_points:
				cv2.circle(image, (lane_point[0],lane_point[1]), 3, color, -1)

	def DrawAreaOnFrame(self, image, color=(255,191,0)) :
		self.draw_area = False
		# Draw a mask for the current lane
		if(self.lanes_detected != []) :
			if(self.lanes_detected[1] and self.lanes_detected[2]):
				self.draw_area = True
				lane_segment_img = image.copy()
				left_lanes_points, right_lanes_points = self.adjust_lanes_points(self.lanes_points[1], self.lanes_points[2], self.img_height)
				self.draw_area_points = [np.vstack((left_lanes_points,np.flipud(right_lanes_points)))]
				cv2.fillPoly(lane_segment_img, pts = self.draw_area_points, color =color)
				image = cv2.addWeighted(image, 0.7, lane_segment_img, 0.1, 0)

		if (not self.draw_area) : self.draw_area_points = []
		return image

	def AutoDrawLanes(self, image, draw_points=True):

		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		output = self.infer.inference(input_tensor)

		# Process output data
		self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)

		# # Draw depth image
		visualization_img = self.draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points)

		return visualization_img

	def adjust_lanes_points(self, left_lanes_points, right_lanes_points, image_height) :
		if (len(left_lanes_points[1]) != 0 ) :
			leftx, lefty  = list(zip(*left_lanes_points))
		else :
			return left_lanes_points, right_lanes_points
		if (len(right_lanes_points) != 0 ) :
			rightx, righty  = list(zip(*right_lanes_points))
		else :
			return left_lanes_points, right_lanes_points

		if len(lefty) > 10:
			self.left_fit = np.polyfit(lefty, leftx, 2)
		if len(righty) > 10:
			self.right_fit = np.polyfit(righty, rightx, 2)

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

		left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
		right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

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
	def process_output(output, cfg):		
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
	def draw_lanes(input_img, lanes_points, lanes_detected, cfg, draw_points=True):
		# Write the detected line points in the image
		visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation = cv2.INTER_AREA)

		# Draw a mask for the current lane
		if(lanes_detected[1] and lanes_detected[2]):
			lane_segment_img = visualization_img.copy()
			
			cv2.fillPoly(lane_segment_img, pts = [np.vstack((lanes_points[1],np.flipud(lanes_points[2])))], color =(255,191,0))
			visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)

		if(draw_points):
			for lane_num,lane_points in enumerate(lanes_points):
				for lane_point in lane_points:
					cv2.circle(visualization_img, (lane_point[0],lane_point[1]), 3, lane_colors[lane_num], -1)

		return visualization_img

	








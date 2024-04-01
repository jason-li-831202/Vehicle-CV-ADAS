from enum import Enum
import cv2
import abc
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from typing import Tuple

class LaneModelType(Enum):
	UFLD_TUSIMPLE = 0
	UFLD_CULANE = 1
	UFLDV2_TUSIMPLE = 2
	UFLDV2_CULANE = 3
	UFLDV2_CURVELANES = 4

class OffsetType(Enum):
	UNKNOWN = "To Be Determined ..."
	RIGHT = "Please Keep Right"
	LEFT = "Please Keep Left"
	CENTER = "Good Lane Keeping"

class CurvatureType(Enum):
	UNKNOWN = "To Be Determined ..."
	STRAIGHT = "Keep Straight Ahead"
	EASY_LEFT =  "Gentle Left Curve Ahead"
	HARD_LEFT = "Hard Left Curve Ahead"
	EASY_RIGHT = "Gentle Right Curve Ahead"
	HARD_RIGHT = "Hard Right Curve Ahead"

lane_colors = [(255, 0, 0),(46,139,87),(50,205,50),(0,255,255)]

class EngineBase(abc.ABC):
	def __init__(self):
		pass

	@abc.abstractmethod
	def get_engine_input_shape(self):
		return NotImplemented
	
	@abc.abstractmethod
	def get_engine_output_shape(self):
		return NotImplemented
	
	@abc.abstractmethod
	def engine_inference(self):
		return NotImplemented

class DetectorBase(abc.ABC):
	_defaults = {
		"model_path": None,
		"model_type" : None,
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
		
	def __init__(self, logger):
		self.__dict__.update(self._defaults) # set up default values

		self.logger = logger
		self.draw_area_points = []
		self.draw_area = False

	def set_input_details(self, engine) -> None :
		if hasattr(engine, "get_engine_input_shape"):
			self.input_shape = engine.get_engine_input_shape()
			self.input_types = engine.engine_dtype

			self.channes, self.input_height, self.input_width = self.input_shape[1:]
			if (self.logger) : 
				self.logger.info(f"-> Input Shape : {self.input_shape}")
				self.logger.info(f"-> Input Type  : {self.input_types}")
		else :
			self.logger.error(f"engine does not adhere to the naming convention of the 'EngineBase' class")

	def set_output_details(self, engine) -> None :
		if hasattr(engine, "get_engine_output_shape"):
			self.output_shape, self.output_names = engine.get_engine_output_shape()
		else :
			self.logger.error(f"engine does not adhere to the naming convention of the 'EngineBase' class")

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
	def __check_lanes_area(lanes_status : list) -> bool :
		if(lanes_status != [] and len(lanes_status) % 2 == 0) :
			index = len(lanes_status) // 2
			if(lanes_status[index-1] and lanes_status[index]):
				return True
		return False

	@abc.abstractmethod
	def DetectFrame(self):
		return NotImplemented	
	
	@abc.abstractmethod
	def DrawDetectedOnFrame(self):
		return NotImplemented	
	
	@abc.abstractmethod
	def DrawAreaOnFrame(self):
		return NotImplemented	

	def AutoDrawLanes(self, image : cv2, draw_points : bool = True, draw_area : bool = True) -> None:
		self.DetectFrame(image)

		if (draw_points) :
			self.DrawDetectedOnFrame(image)

		if (draw_area) :
			self.DrawAreaOnFrame(image, adjust_lanes=True)
		return image

class TensorRTBase():
	def __init__(self, engine_file_path):
		self.providers = 'CUDAExecutionProvider'
		self.framework_type = "trt"
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
		self.dtype = trt.nptype(engine.get_binding_dtype(0)) 
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
		for host_output, cuda_output in zip(host_outputs, cuda_outputs) :
			cuda.memcpy_dtoh_async(host_output, cuda_output, stream)
		# Synchronize the stream
		stream.synchronize()
		# Remove any context from the top of the context stack, deactivating it.
		self.cuda_driver_context.pop()
	
		return host_outputs

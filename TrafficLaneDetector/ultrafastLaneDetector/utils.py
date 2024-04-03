from enum import Enum
import abc
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

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

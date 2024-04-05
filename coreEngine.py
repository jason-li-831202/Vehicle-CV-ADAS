import abc, os
import numpy as np
import onnxruntime
import tensorrt as trt
import pycuda.driver as cuda

class EngineBase(abc.ABC):
	'''
    Currently only supports Onnx/TensorRT framework
	'''
	def __init__(self, model_path):
		if not os.path.isfile(model_path):
			raise Exception("The model path [%s] can't not found!" % model_path)
		assert model_path.endswith(('.onnx', '.trt')), 'Onnx/TensorRT Parameters must be a .onnx/.trt file.'
		self._framework_type = None

	@property
	def framework_type(self):
		if (self._framework_type == None):
			raise Exception("Framework type can't be None")
		return self._framework_type
	
	@framework_type.setter
	def framework_type(self, value):
		if ( not isinstance(value, str)):
			raise Exception("Framework type need be str")
		self._framework_type = value
	
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
	
class TensorRTEngine(EngineBase, TensorRTBase):

	def __init__(self, engine_file_path):
		EngineBase.__init__(self, engine_file_path)
		TensorRTBase.__init__(self, engine_file_path)
		self.engine_dtype = self.dtype
		self.__load_engine_interface()

	def __load_engine_interface(self):
		# Get the number of bindings
		num_bindings = self.engine.num_bindings

		self.__input_shape = []
		self.__input_names = []
		self.__output_names = []
		self.__output_shapes = []
		for i in range(num_bindings):
			if self.engine.binding_is_input(i):
				self.__input_shape.append(self.engine.get_binding_shape(i))
				self.__input_names.append(self.engine.get_binding_name(i))
				continue
			self.__output_names.append(self.engine.get_binding_name(i))
			self.__output_shapes.append(self.engine.get_binding_shape(i))

	def get_engine_input_shape(self):
		return self.__input_shape[0]

	def get_engine_output_shape(self):
		return self.__output_shapes, self.__output_names

	def engine_inference(self, input_tensor):
		host_outputs = self.inference(input_tensor)
		# Here we use the first row of output in that batch_size = 1
		trt_outputs = []
		for i, output in enumerate(host_outputs) :
			trt_outputs.append(np.reshape(output, self.__output_shapes[i]))

		return trt_outputs

class OnnxEngine(EngineBase):

	def __init__(self, onnx_file_path):
		EngineBase.__init__(self, onnx_file_path)
		if (onnxruntime.get_device() == 'GPU') :
			self.session = onnxruntime.InferenceSession(onnx_file_path, providers=['CUDAExecutionProvider'])
		else :
			self.session = onnxruntime.InferenceSession(onnx_file_path)
		self.providers = self.session.get_providers()
		self.engine_dtype = np.float16 if 'float16' in self.session.get_inputs()[0].type else np.float32
		self.framework_type = "onnx"
		self.__load_engine_interface()

	def __load_engine_interface(self):
		self.__input_shape = [input.shape for input in self.session.get_inputs()]
		self.__input_names = [input.name for input in self.session.get_inputs()]
		self.__output_shape = [output.shape for output in self.session.get_outputs()]
		self.__output_names = [output.name for output in self.session.get_outputs()]

	def get_engine_input_shape(self):
		return self.__input_shape[0]

	def get_engine_output_shape(self):
		return self.__output_shape, self.__output_names
	
	def engine_inference(self, input_tensor):
		output = self.session.run(self.__output_names, {self.__input_names[0]: input_tensor})
		return output

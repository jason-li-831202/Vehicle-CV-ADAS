import tensorrt as trt
import sys, time
import argparse
import inspect
from pathlib import Path
from typing import *
# TODO : for Calibrator class
# import pycuda.driver as drv
# import pycuda.autoinit
# import numpy as np
# import cv2 as cv2
# from ObjectDetector.utils import Scaler

"""
takes in onnx model
converts to tensorrt
"""

parser = argparse.ArgumentParser(description='https://github.com/jason-li-831202/Vehicle-CV-ADAS')
parser.add_argument('--input_onnx_model', '-i', default="./ObjectDetector/models/yolov10n-coco_fp32.onnx", type=str, help='Onnx model path.')
parser.add_argument('--output_trt_model', '-o', default="./ObjectDetector/models/yolov10n-coco_fp16.trt", type=str, help='Tensorrt model path.')
# parser.add_argument("--calib_image_dir", default=None, type=Path, help="The calibrate data required for conversion to int8, if None will use dynamic quantization")
parser.add_argument('--verbose', action='store_true', default=False, help='TensorRT: verbose log')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

# class Calibrator(trt.IInt8MinMaxCalibrator):
#     def __init__(self, batch_size=1, height=640, width=640, calibration_images="", cache_file=""):
#         trt.IInt8MinMaxCalibrator.__init__(self)
#         self.batch_idx = 0
#         self.batch_size = batch_size
#         self.cache_file = cache_file
#         self.height = height
#         self.width = width

#         self.img_list = [ str(name) for name in Path(calibration_images).iterdir()]
#         self.max_batch_idx = len(self.img_list) // self.batch_size
#         self.data_size = trt.volume([self.batch_size, 3, self.height, self.width]) * trt.float32.itemsize
#         self.batch_allocation = drv.mem_alloc(self.data_size)
#         self.scaler = Scaler((self.height, self.width), keep_ratio=True)
#         print('Found all {} images to calib.'.format(len(self.img_list)))

#     def preprocess(self, img):
#         image = self.scaler.process_image(img)
#         # TODO : for yolov5/6/7/8 = 1/255.0, for yolox = 1.
#         image = cv2.dnn.blobFromImage(image, 1, (image.shape[1], image.shape[0]), swapRB=True, crop=False).astype(np.float32)
#         return image
    
#     def next_batch(self):
#         if self.batch_idx < self.max_batch_idx:
#             batch_files = self.img_list[self.batch_idx * self.batch_size: (self.batch_idx + 1) * self.batch_size]
#             batch_imgs = np.zeros((self.batch_size, 3, self.height, self.width), dtype=np.float32)
#             for i, f in enumerate(batch_files):
#                 img = cv2.imread(f) # (h, w, c)
#                 img = self.preprocess(img)
#                 batch_imgs[i] = img
#             self.batch_idx += 1
#             return batch_imgs
#         else:
#             return np.array([])

#     def __len__(self):
#         return self.length

#     def get_batch_size(self):
#         """
#         Overrides from trt.IInt8EntropyCalibrator2.
#         Get the batch size to use for calibration.
#         :return: Batch size.
#         """
#         return self.batch_size

#     def get_batch(self, name):
#         batch = self.next_batch()
#         print("Calibrating image {} / {}".format(self.batch_idx, self.max_batch_idx ))
#         if not batch.size:
#             return None
#         drv.memcpy_htod(self.batch_allocation, batch)
#         return [int(self.batch_allocation)]

#     def read_calibration_cache(self):
#         """
#         Overrides from trt.IInt8EntropyCalibrator2.
#         Read the calibration cache file stored on disk, if it exists.
#         :return: The contents of the cache file, if any.
#         """
#         if Path(self.cache_file).exists():
#             with open(self.cache_file, "rb") as f:
#                 return f.read()

#     def write_calibration_cache(self, cache):
#         """
#         Overrides from trt.IInt8EntropyCalibrator2.
#         Store the calibration cache to a file on disk.
#         :param cache: The contents of the calibration cache to store.
#         """
#         with open(self.cache_file, "wb") as f:
#             f.write(cache)
	
class EngineBuilder:
	"""
	Parses an ONNX graph and builds a TensorRT engine from it.
	"""
	def __init__(self, verbose : bool = False, workspace : int = 1):
		self.logger = trt.Logger(trt.Logger.INFO)
		if verbose: self.logger.min_severity = trt.Logger.Severity.VERBOSE
		trt.init_libnvinfer_plugins(self.logger, namespace="")

		print(self.colorstr("👉 Starting export with TensorRT Version : "), trt.__version__)
		self.builder = trt.Builder(self.logger)
		self.config = self.builder.create_builder_config()
		if trt.__version__[0] <= '7':
			self.builder.max_workspace_size = int(workspace * (1 << 30)) # 1GB
		else :
			self.config.max_workspace_size = int(workspace * (1 << 30))

		self.network = None
		self.parser = None
	
	def create_network(self, onnx_model_path : str):
		assert Path(onnx_model_path).exists(), print(self.colorstr("red", "File=[ %s ] is not exist. Please check it !" %onnx_model_path ))
		EXPLICIT_BATCH = []
		if trt.__version__[0] >= '7':
			EXPLICIT_BATCH.append( 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) )
		
		self.network = self.builder.create_network(*EXPLICIT_BATCH)
		self.parser = trt.OnnxParser(self.network, self.logger)

		print(self.colorstr('cyan', '👉 Loading the ONNX file...'))
		with open(onnx_model_path, 'rb') as f:
			if not self.parser.parse(f.read()):
				for error in range(self.parser.num_errors):
					print(self.parser.get_error(error))
				raise RuntimeError(f'Failed to load ONNX file: {onnx_model_path}')	

		print(self.colorstr('magenta', "*"*40))
		print(self.colorstr('magenta', 'underline', '❄️  Network Description: ❄️'))
		shape = list(self.network.get_input(0).shape) 
		inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
		outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

		print(self.colorstr('bright_magenta', " - Input Info"))
		for inp in inputs:
			print(self.colorstr('bright_magenta', f'   Input "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}'))
		print(self.colorstr('bright_magenta', " - Output Info"))
		for out in outputs:
			print(self.colorstr('bright_magenta', f'   Output "{out.name}" with shape {out.shape} and dtype {out.dtype}'))

	def create_engine(self, trt_model_path : str, calib_image_path: Optional[str] = None):
		start = time.time()
		inp = [self.network.get_input(i) for i in range(self.network.num_inputs)][0]
		print(f' Note: building FP{16 if (self.builder.platform_has_fast_fp16 and inp.dtype==trt.DataType.HALF) else 32} engine as {Path(trt_model_path).resolve()}')
		if self.builder.platform_has_fast_fp16 and inp.dtype==trt.DataType.HALF:
			self.config.set_flag(trt.BuilderFlag.FP16)
        # if calib_image_path != None and (calib_image_path).is_dir():
        #     # Also enable fp16, as some layers may be even more efficient in fp16 than int8
        #     self.config.set_flag(trt.BuilderFlag.FP16)
        #     self.config.set_flag(trt.BuilderFlag.INT8)
        #     self.config.int8_calibrator = Calibrator(1, inp.shape[2], inp.shape[3], calib_image_path)
		print(self.colorstr('magenta', "*"*40))

		print(self.colorstr('👉 Building the TensorRT engine. This would take a while...'))
		engine = self.builder.build_engine(self.network, self.config)  # 没有序列化,<class 'tensorrt.tensorrt.ICudaEngine'>
		# engine = self.builder.build_serialized_network(network, config) # 已经序列化,类型为:<class 'tensorrt.tensorrt.IHostMemory'
		if engine is not None: print(self.colorstr('👉 Completed creating engine.'))
		
		try:
			with open(trt_model_path, 'wb') as f:
				f.write(engine.serialize())  # 序列化
				# f.write(engine)  
		except Exception as e:
			print(self.colorstr('red', f'Eexport failure ❌ : {e}'))

		convert_time = time.time() - start
		print(self.colorstr(f'\nExport complete success ✅ {convert_time:.1f}s'
					f"\nResults saved to [{trt_model_path}]"
					f"\nModel size:      {file_size(trt_model_path):.1f} MB"
					f'\nVisualize:       https://netron.app'))
	
	@staticmethod
	def colorstr(*input):
		# Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  self.colorstr('blue', 'hello world')
		*args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
		colors = {
			'black': '\033[30m',  # basic colors
			'red': '\033[31m',
			'green': '\033[32m',
			'yellow': '\033[33m',
			'blue': '\033[34m',
			'magenta': '\033[35m',
			'cyan': '\033[36m',
			'white': '\033[37m',
			'bright_black': '\033[90m',  # bright colors
			'bright_red': '\033[91m',
			'bright_green': '\033[92m',
			'bright_yellow': '\033[93m',
			'bright_blue': '\033[94m',
			'bright_magenta': '\033[95m',
			'bright_cyan': '\033[96m',
			'bright_white': '\033[97m',
			'end': '\033[0m',  # misc
			'bold': '\033[1m',
			'underline': '\033[4m'}
		return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def file_size(path: str):
	# Return file/dir size (MB)
	mb = 1 << 20  # bytes to MiB (1024 ** 2)
	path = Path(path)
	if path.is_file():
		return path.stat().st_size / mb
	elif path.is_dir():
		return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
	else:
		return 0.0

def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
	# Print function arguments (optional args dict)
	x = inspect.currentframe().f_back  # previous frame
	file, _, func, _, _ = inspect.getframeinfo(x)
	if args is None:  # get args automatically
		args, _, _, frm = inspect.getargvalues(x)
		args = {k: v for k, v in frm.items() if k in args}
	try:
		file = Path(file).resolve().relative_to(ROOT).with_suffix('')
	except ValueError:
		file = Path(file).stem
	s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
	print(EngineBuilder.colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))


if __name__ == '__main__':
	args = parser.parse_args()
	print_args(vars(args))

	onnx_model_path = args.input_onnx_model
	trt_model_path = args.output_trt_model
	verbose = args.verbose

	builder = EngineBuilder(verbose)
	builder.create_network(onnx_model_path)
	builder.create_engine(trt_model_path)


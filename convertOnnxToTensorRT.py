import tensorrt as trt
import sys, time
import argparse
import inspect
from pathlib import Path
from typing import *
# import pycuda.driver as drv
# import pycuda.autoinit
# import numpy as np
# import cv2 as cv2

# from ObjectDetector.yoloDetector import YoloDetector
"""
takes in onnx model
converts to tensorrt
"""

parser = argparse.ArgumentParser(description='https://github.com/jason-li-831202/Vehicle-CV-ADAS')
parser.add_argument('--input_onnx_model', '-i', default="./ObjectDetector/models/yolov8m-coco_fp16.onnx", type=str, help='Onnx model path.')
parser.add_argument('--output_trt_model', '-o', default="./ObjectDetector/models/yolov8m-coco_fp16.trt", type=str, help='Tensorrt model path.')
parser.add_argument('--verbose', action='store_true', default=False, help='TensorRT: verbose log')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory

# class Calibrator(trt.IInt8EntropyCalibrator):
#     def __init__(self, quantification=1, batch_size=1, height=640, width=640, calibration_images="", cache_file=""):
#         trt.IInt8EntropyCalibrator.__init__(self)
#         self.index = 0
#         self.length = quantification
#         self.batch_size = batch_size
#         self.cache_file = cache_file
#         self.height = height
#         self.width = width
#         self.img_list = [ str(name) for name in Path(calibration_images).iterdir()]
#         self.calibration_data = np.zeros((self.batch_size, 3, self.height, self.width), dtype=np.float32)
#         self.d_input = drv.mem_alloc(self.calibration_data.nbytes)

#     def next_batch(self):
#         if self.index < self.length:
#             for i in range(self.batch_size):
#                 img = cv2.imread(self.img_list[i + self.index*self.batch_size])
#                 img = self.preprocess(img)
#                 self.calibration_data[i] = img
#             self.index += 1
#             return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
#         else:
#             return np.array([])

#     def __len__(self):
#         return self.length

#     def get_batch_size(self):
#         return self.batch_size

#     def get_batch(self, name):
#         batch = self.next_batch()
#         if not batch.size:
#             return None
#         drv.memcpy_htod(self.d_input, batch)
#         return [int(self.d_input)]

#     def read_calibration_cache(self):
#         # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
#         if Path(self.cache_file).exists():
#             with open(self.cache_file, "rb") as f:
#                 return f.read()

#     def write_calibration_cache(self, cache):
#         with open(self.cache_file, "wb") as f:
#             f.write(cache)

#     def preprocess(self, img):
#         image, newh, neww, ratioh, ratiow, padh, padw = YoloDetector.resize_image_format(img, (self.height, self.width), True)
#         image = cv2.dnn.blobFromImage(image, 1/255.0, (image.shape[1], image.shape[0]), swapRB=True, crop=False).astype(np.float32)
#         return image
    
def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
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
    print(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))

if __name__ == '__main__':
	args = parser.parse_args()
	print_args(vars(args))

	onnx_model_path = args.input_onnx_model
	trt_model_path = args.output_trt_model
	verbose = args.verbose
	assert Path(onnx_model_path).exists(), print(colorstr("red", "File=[ %s ] is not exist. Please check it !" %onnx_model_path ))

	logger = trt.Logger(trt.Logger.INFO)
	if verbose: logger.min_severity = trt.Logger.Severity.VERBOSE

	print(colorstr("👉 Starting export with TensorRT Version : "), trt.__version__)
	EXPLICIT_BATCH = []
	if trt.__version__[0] >= '7':
		EXPLICIT_BATCH.append( 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) )

	start = time.time()
	with trt.Builder(logger) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, logger) as parser:
		print(colorstr('cyan', '👉 Loading the ONNX file...'))
		with open(onnx_model_path, 'rb') as f:
			if not parser.parse(f.read()):
				for error in range(parser.num_errors):
					print(parser.get_error(error))
				raise RuntimeError(f'Failed to load ONNX file: {onnx_model_path}')	
			

		print(colorstr('magenta', "*"*40))
		print(colorstr('magenta', 'underline', '❄️  Network Description: ❄️'))
		shape = list(network.get_input(0).shape) # reshape input from 32 to 1
		inputs = [network.get_input(i) for i in range(network.num_inputs)]
		outputs = [network.get_output(i) for i in range(network.num_outputs)]
		print(colorstr('bright_magenta', " - Input Info"))
		for inp in inputs:
			print(colorstr('bright_magenta', f'   Input "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}'))
		print(colorstr('bright_magenta', " - Output Info"))
		for out in outputs:
			print(colorstr('bright_magenta', f'   Output "{out.name}" with shape {out.shape} and dtype {out.dtype}'))

		config = builder.create_builder_config()
		if trt.__version__[0] <= '7':
			builder.max_workspace_size = 1 << 30 # 1GB
		else :
			config.max_workspace_size = 1 << 30
               
		print(f' Note: building FP{16 if (builder.platform_has_fast_fp16 and inp.dtype==trt.DataType.HALF) else 32} engine as {f}')
		if builder.platform_has_fast_fp16 and inp.dtype==trt.DataType.HALF:
			config.set_flag(trt.BuilderFlag.FP16)
		# config.set_flag(trt.BuilderFlag.INT8)
		# config.int8_calibrator = Calibrator(1, 1, inp.shape[2], inp.shape[3], "./demo/val2017")
		print(colorstr('magenta', "*"*40))

		print(colorstr('👉 Building the TensorRT engine. This would take a while...'))
		engine = builder.build_engine(network, config)  # 没有序列化,<class 'tensorrt.tensorrt.ICudaEngine'>
		# engine = builder.build_serialized_network(network, config) # 已经序列化,类型为:<class 'tensorrt.tensorrt.IHostMemory'
		if engine is not None: print(colorstr('👉 Completed creating engine.'))
		
		try:
			with open(trt_model_path, 'wb') as f:
				f.write(engine.serialize())  # 序列化
				# f.write(engine)  
		except Exception as e:
			print(colorstr('red', f'Eexport failure ❌ : {e}'))
	
	convert_time = time.time() - start
	print(colorstr(f'\nExport complete success ✅ {convert_time:.1f}s'
				   f"\nResults saved to [{trt_model_path}]"
				   f"\nModel size:      {file_size(trt_model_path):.1f} MB"
				   f'\nVisualize:       https://netron.app'))
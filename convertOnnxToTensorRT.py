import tensorrt as trt
import sys, time
import argparse
from pathlib import Path

"""
takes in onnx model
converts to tensorrt
"""

parser = argparse.ArgumentParser(description='https://github.com/jason-li-831202/Vehicle-CV-ADAS')
parser.add_argument('--input_onnx_model', '-i', default="./ObjectDetector/models/yolov8m-coco_fp16.onnx", type=str, help='onnx model path.')
parser.add_argument('--output_trt_model', '-o', default="./ObjectDetector/models/yolov8m-coco_fp16.trt", type=str, help='trt model path.')

def file_size(path):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0
	
if __name__ == '__main__':
	args = parser.parse_args()
	onnx_model_path = args.input_onnx_model
	trt_model_path = args.output_trt_model
	assert Path(onnx_model_path).exists(), print("File=[ %s ] is not exist. Please check it !" %onnx_model_path )

	logger = trt.Logger(trt.Logger.INFO)
	EXPLICIT_BATCH = []
	print("starting export with TensorRT Version : ", trt.__version__)
	if trt.__version__[0] >= '7':
		EXPLICIT_BATCH.append( 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) )

	start = time.time()
	with trt.Builder(logger) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, logger) as parser:
		with open(onnx_model_path, 'rb') as f:
			if not parser.parse(f.read()):
				for error in range(parser.num_errors):
					print(parser.get_error(error))
				raise RuntimeError(f'failed to load ONNX file: {onnx_model_path}')	
			
		# reshape input from 32 to 1
		shape = list(network.get_input(0).shape)

		profile = builder.create_optimization_profile()
		# FIXME: Hardcoded for ImageNet. The minimum/optimum/maximum dimensions of a dynamic input tensor are the same.
		# profile.set_shape(input_tensor_name, (1, 3, 224, 224), (max_batch_size, 3, 224, 224), (max_batch_size, 3, 224, 224))
		
		config = builder.create_builder_config()
		config.add_optimization_profile(profile)

		print("*"*40)
		print('Network Description:')
		inputs = [network.get_input(i) for i in range(network.num_inputs)]
		outputs = [network.get_output(i) for i in range(network.num_outputs)]
		print("- Input Info")
		for inp in inputs:
			print(f'	Input "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
		print("- Output Info")
		for out in outputs:
			print(f'	Output "{out.name}" with shape {out.shape} and dtype {out.dtype}')
		print(f'building FP{16 if (builder.platform_has_fast_fp16 and inp.dtype==trt.DataType.HALF) else 32} engine as {f}')
		if builder.platform_has_fast_fp16 and inp.dtype==trt.DataType.HALF:
			config.set_flag(trt.BuilderFlag.FP16)
		print("*"*40)

		engine = builder.build_engine(network, config)
		print('Completed creating engine.')
		
		try:
			with open(trt_model_path, 'wb') as f:
				f.write(engine.serialize())
		except Exception as e:
			print(f'Eexport failure ❌ : {e}')
	
	convert_time = time.time() - start
	print(f'Serialized the TensorRT engine Export success ✅ {convert_time:.1f}s, saved as [{trt_model_path}] ({file_size(trt_model_path):.1f} MB)')

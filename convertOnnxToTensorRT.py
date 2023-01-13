import tensorrt as trt
import sys
import argparse


import tensorrt as trt
import sys

"""
takes in onnx model
converts to tensorrt
"""

if __name__ == '__main__':
	onnx_model_path = "./TrafficLaneDetector/models/tusimple_18.onnx"
	trt_model_path = "./TrafficLaneDetector/models/tusimple_18.trt"

	logger = trt.Logger(trt.Logger.INFO)
	EXPLICIT_BATCH = []
	if trt.__version__[0] >= '7':
		EXPLICIT_BATCH.append(
			1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

	with trt.Builder(logger) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, logger) as parser:
		with open(onnx_model_path, 'rb') as f:
			if not parser.parse(f.read()):
				for error in range(parser.num_errors):
					print(parser.get_error(error))
				sys.exit()
			
		# reshape input from 32 to 1
		print('network', network)
		shape = list(network.get_input(0).shape)
		print('shape', shape)

		profile = builder.create_optimization_profile()
		# FIXME: Hardcoded for ImageNet. The minimum/optimum/maximum dimensions of a dynamic input tensor are the same.
		# profile.set_shape(input_tensor_name, (1, 3, 224, 224), (max_batch_size, 3, 224, 224), (max_batch_size, 3, 224, 224))
		
		config = builder.create_builder_config()
		config.add_optimization_profile(profile)
		

		inputs = [network.get_input(i) for i in range(network.num_inputs)]
		outputs = [network.get_output(i) for i in range(network.num_outputs)]
		print('Network Description:')
		for inp in inputs:
			print(f'input "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
		for out in outputs:
			print(f'output "{out.name}" with shape {out.shape} and dtype {out.dtype}')

		engine = builder.build_engine(network, config)
		print('Completed creating engine.')
		
		with open(trt_model_path, 'wb') as f:
			f.write(engine.serialize())

	print('Serialized the TensorRT engine to file: %s' % trt_model_path)
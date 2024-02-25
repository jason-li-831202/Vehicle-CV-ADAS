import onnx
import argparse
from onnxconverter_common import float16
# from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

parser = argparse.ArgumentParser(description='https://github.com/jason-li-831202/Vehicle-CV-ADAS')
parser.add_argument('--input_onnx_model', '-i', default='./ObjectDetector/models/yolov5m-coco_fp32.onnx', type=str, help='onnx model path.')


if __name__ == "__main__":
    args = parser.parse_args()

    input_onnx_model  = args.input_onnx_model
    basePath = Path(input_onnx_model).parent
    baseName = Path(input_onnx_model).stem
    baseaSuffix = Path(input_onnx_model).suffix

    # Load your model
    onnx_model = onnx.load(input_onnx_model)
    # Simplify (Note : yolov9 need used it)
    try:
        import onnxsim
        print("Starting to simplify onnx with %s..." % onnxsim.__version__)

        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, "assert check failed"
    except ImportError:
        print(
            "onnxsim is not found, if you want to simplify the onnx, "
            + "you should install it:\n\t"
            + "pip install -U onnxsim onnxruntime\n"
        )
    except Exception as e:
        print(f'Simplify onnx export failure ❌ : {e}')

    # Convert tensor float type from your input ONNX model to tensor float16
    onnx_model = float16.convert_float_to_float16(onnx_model)
    # Save as protobuf
    onnx.save(onnx_model, str(Path.joinpath(basePath, baseName+"_fp16"+baseaSuffix)) )

    # quantized_model = quantize_dynamic(input_onnx_model, str(Path.joinpath(basePath, baseName+"_fp16"+baseaSuffix)), weight_type=QuantType.QUInt8)
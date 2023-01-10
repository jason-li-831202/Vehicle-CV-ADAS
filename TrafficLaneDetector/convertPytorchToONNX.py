import cv2
import torch
import numpy as np
import onnx

from ultrafastLaneDetector.utils import ModelType
from ultrafastLaneDetector.ultrafastLaneV2.configs.config import Config
from ultrafastLaneDetector.ultrafastLaneV2 import model_tusimple
from ultrafastLaneDetector.ultrafastLane.model import parsingNet
from pathlib import Path


class ModelConfig():

	def __init__(self, model_type):

		if model_type == ModelType.UFLD_TUSIMPLE:
			self.init_tusimple_config()
		else:
			self.init_culane_config()

	def init_tusimple_config(self):
		self.img_w = 1280
		self.img_h = 720
		self.griding_num = 100
		self.cls_num_per_lane = 56

	def init_culane_config(self):
		self.img_w = 1640
		self.img_h = 590
		self.griding_num = 200
		self.cls_num_per_lane = 18

def merge_config(config_path):
    cfg = Config.fromfile(config_path)

    if cfg.dataset == 'CULane':
        cfg.row_anchor = np.linspace(0.42,1, cfg.num_row)
        cfg.col_anchor = np.linspace(0,1, cfg.num_col)
    elif cfg.dataset == 'Tusimple':
        cfg.row_anchor = np.linspace(160,710, cfg.num_row)/720
        cfg.col_anchor = np.linspace(0,1, cfg.num_col)
    elif cfg.dataset == 'CurveLanes':
        cfg.row_anchor = np.linspace(0.4, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    
    return cfg

def convert_model(model_path, onnx_file_path, model_type=ModelType.UFLDV2_CULANE):

	# Load model configuration based on the model type
	print("Model Type : ", model_type.name)
	if ( "UFLDV2" in model_type.name) :
		print(model_path)
		cfg = merge_config("./ultrafastLaneDetector/ultrafastLaneV2/configs/"+Path(model_path).stem+".py")
		assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']
	else :
		cfg = ModelConfig(model_type)


	# Load the model architecture
	if ( "UFLDV2" in model_type.name) :
		net = model_tusimple.get_model(cfg)
		img = torch.zeros(1, 3, cfg.train_height, cfg.train_width).to('cuda')
	else :
		net = parsingNet(pretrained = False, backbone='18', cls_dim = (cfg.griding_num+1,cfg.cls_num_per_lane,4),
					use_aux=False) # we dont need auxiliary segmentation in testing
		img = torch.zeros(1, 3, 288, 800).to('cpu')

	state_dict = torch.load(model_path, map_location='cpu')['model'] # CPU

	compatible_state_dict = {}
	for k, v in state_dict.items():
		if 'module.' in k:
			compatible_state_dict[k[7:]] = v
		else:
			compatible_state_dict[k] = v

	# Load the weights into the model
	net.load_state_dict(compatible_state_dict, strict=False)
	torch.onnx.export(net, img, onnx_file_path, verbose=True)

	model = onnx.load(onnx_file_path)

	# Check that the IR is well formed
	onnx.checker.check_model(model)
	# Print a human readable representation of the graph
	print("==============================================================================================")

if __name__ == '__main__':
	onnx_model_path = "curvelanes_res18.onnx"
	model_path = "models/curvelanes_res18.pth"
	model_type = ModelType.UFLDV2_CULANE

	convert_model(model_path, onnx_model_path, model_type)




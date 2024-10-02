import cv2
import numpy as np
import onnx
# from torchsummary import summary
from io import BytesIO

from ufldDetector.utils import LaneModelType
from ufldDetector.exportLib.ultrafastLane.model import parsingNet
from ufldDetector.exportLib.ultrafastLaneV2.configs.config import Config
from ufldDetector.exportLib.ultrafastLaneV2 import model_tusimple, model_curvelanes
from pathlib import Path
import torch

class LaneV1Config():

	def __init__(self, model_type):

		if model_type == LaneModelType.UFLD_TUSIMPLE:
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

def LaneV2Config(config_path):
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

def convert_model(model_path, model_type=LaneModelType.UFLDV2_CULANE):

	# Load model configuration based on the model type
	print("Model Type : ", model_type.name)
	file = Path(model_path)
	onnx_file_path = file.with_suffix('.onnx')

	if ( "UFLDV2" in model_type.name) :
		cfg = LaneV2Config("./ufldDetector/exportLib/ultrafastLaneV2/configs/"+file.stem+".py")
		assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']
	else :
		cfg = LaneV1Config(model_type)


	# Load the model architecture
	if ( "UFLDV2" in model_type.name) :
		# TODO : not done
		# if (model_type == LaneModelType.UFLDV2_CURVELANES) :
		# 	net = model_curvelanes.get_model(cfg)
		# else :
		net = model_tusimple.get_model(cfg)
		img = torch.zeros(1, 3, cfg.train_height, cfg.train_width).to('cuda')
	else :
		net = parsingNet(pretrained = False, backbone='18', cls_dim = (cfg.griding_num+1,cfg.cls_num_per_lane,4),
					use_aux=False) # we dont need auxiliary segmentation in testing
		img = torch.zeros(1, 3, 288, 800).to('cpu')

	state_dict = torch.load(file, map_location='cpu')['model'] # CPU

	compatible_state_dict = {}
	for k, v in state_dict.items():
		if 'module.' in k:
			compatible_state_dict[k[7:]] = v
		else:
			compatible_state_dict[k] = v

	# Load the weights into the model
	# net.load_state_dict(compatible_state_dict, strict=False)
	# summary(net, torch.squeeze(img, 0).cpu().numpy().shape )
	print("Starting ONNX export with onnx %s..." % onnx.__version__)
	net.load_state_dict(compatible_state_dict, strict=False)
	torch.onnx.export(net, img, onnx_file_path, verbose=True)

	# Check that the IR is well formed
	model = onnx.load(onnx_file_path)
	onnx.checker.check_model(model)
	print("ONNX export success, saved as:\n\t%s" % onnx_file_path)
	
	# Print a human readable representation of the graph
	print("==============================================================================================")

if __name__ == '__main__':
	model_path = "models/tusimple_res18.pth"
	model_type = LaneModelType.UFLDV2_TUSIMPLE

	convert_model(model_path, model_type)




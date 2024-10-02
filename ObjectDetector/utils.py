import cv2
import numpy as np
from enum import Enum
from numba import jit
from typing import *
from dataclasses import dataclass

class CollisionType(Enum):
	UNKNOWN = "Determined ..."
	NORMAL = "Normal Risk"
	PROMPT = "Prompt Risk"
	WARNING = "Warning Risk"


class ObjectModelType(Enum):
	YOLOV5 = 0
	YOLOV5_LITE = 1
	YOLOV6 = 2
	YOLOV7 = 3
	YOLOV8 = 4
	YOLOV9 = 5
	YOLOV10 = 6
	EfficientDet = 7
	
def hex_to_rgb(value):
	value = value.lstrip('#')
	lv = len(value)
	return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

@dataclass
class Scaler(object):
	'''
	All Shape format: Tuple[int, int] -> (H, W)
	'''
	target_size: Tuple[int, int]
	keep_ratio: bool = True

	_new_shape: Optional[Tuple[int, int]] = None
	_old_shape: Optional[Tuple[int, int]] = None
	_pad_shape: Optional[Tuple[int, int]] = None

	def process_image(self, srcimg : np.ndarray) -> np.ndarray:
		padh, padw, newh, neww = 0, 0, self.target_size[0], self.target_size[1]

		if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
			hw_scale = srcimg.shape[0] / srcimg.shape[1]
			if hw_scale > 1:
				newh, neww = self.target_size[0], int(self.target_size[1] / hw_scale)	
				padw = int((self.target_size[1] - neww) * 0.5)
			else:
				newh, neww = int(self.target_size[0] * hw_scale) + 1, self.target_size[1]
				padh = int((self.target_size[0] - newh) * 0.5)
			img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_LINEAR)
			canvas = np.full((self.target_size[0], self.target_size[1], 3), 114, dtype=np.uint8)
			canvas[padh:(padh + newh), padw:(padw + neww), :] = img

		else:
			canvas = cv2.resize(srcimg, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)

		self._old_shape = (srcimg.shape[0], srcimg.shape[1])
		self._new_shape = (newh, neww)
		self._pad_shape = (padh, padw)	
		return canvas	 

	def get_scale_ratio(self): 
		if (self._old_shape == self._new_shape == None) :
			raise Exception("Please operate 'process_image' before conversion")
		return self._old_shape[0] / self._new_shape[0],  self._old_shape[1] / self._new_shape[1]
	
	def convert_boxes_coordinate(self, boxes: list, in_format: str="xyxy", out_format: str="xywh") -> np.ndarray:
		if not isinstance(boxes, np.ndarray):
			boxes = np.array(boxes)

		if (boxes.size > 0) :
			ratioh, ratiow = self.get_scale_ratio()
			padh, padw = self._pad_shape
			boxes = np.vstack(boxes)
			if (in_format == "xywh"):
				boxes[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]

			# [x1, y1, x2, y2]
			boxes[..., [0, 2]] = (boxes[..., [0, 2]] - padw) * ratiow
			boxes[..., [1, 3]] = (boxes[..., [1, 3]] - padh) * ratioh

			if (out_format == "xywh"):
				boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
		return boxes

	def convert_kpss_coordinate(self, kpss : list) -> np.ndarray:
		if not isinstance(kpss, np.ndarray):
			kpss = np.array(kpss)

		if (kpss != []) :
			ratioh, ratiow = self.get_scale_ratio()
			padh, padw = self._pad_shape
			kpss = np.vstack(kpss)
			kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
			kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh
		return kpss

class NMS(object):
	def __init__(self):
		pass

	@staticmethod	
	def fast_nms(dets: Union[list, np.ndarray], scores:  Union[list, np.ndarray], iou_thr: float, dets_type: str = "xyxy"):
		"""
		It's different from original nms because we have float coordinates on range [0; 1]

		Args:
			dets: numpy array of boxes with shape: (N, 4). Defalut Order: x1, y1, x2, y2.
			scores: numpy array of confidence.
			iou_thr: IoU value for boxes.
			dets_type: boxes order format - "xyxy", "xywh". Defalut: xyxy

		Returns:
			Index of boxes to keep
		"""
		_dets =  np.array(dets) if not isinstance(dets, np.ndarray) else dets.copy()
		_scores = np.array(scores) if not isinstance(scores, np.ndarray) else scores.copy()
		if (_dets.shape[0] > 0) :
			if (dets_type == "xywh") :
				_dets[:, 2:4] = _dets[:, 0:2] + _dets[:, 2:4]
			return NMS().__fast_nms(_dets, _scores, iou_thr)
		else :
			return []
	
	@staticmethod	
	@jit(nopython=True)
	def __fast_nms(dets: np.array, scores: np.array, iou_thr: float):
		if len(dets) == 1:
			return [0]
		
		x1 = dets[:, 0]
		y1 = dets[:, 1]
		x2 = dets[:, 2]
		y2 = dets[:, 3]

		areas = (x2 - x1) * (y2 - y1)
		order = scores.argsort()[::-1]
		
		keep = []
		while order.size > 0:
			i = order[0]
			keep.append(i)
			xx1 = np.maximum(x1[i], x1[order[1:]])
			yy1 = np.maximum(y1[i], y1[order[1:]])
			xx2 = np.minimum(x2[i], x2[order[1:]])
			yy2 = np.minimum(y2[i], y2[order[1:]])

			w = np.maximum(0.0, xx2 - xx1)
			h = np.maximum(0.0, yy2 - yy1)
			inter = w * h
			ovr = inter / (areas[i] + areas[order[1:]] - inter)

			inds = np.where(ovr <= iou_thr)[0]
			order = order[inds + 1]

		return keep

	@staticmethod	
	def fast_soft_nms(dets: Union[list, np.ndarray], scores:  Union[list, np.ndarray], iou_thr: float = 0.3, 
				  sigma: float = 0.5, score_thr: float = 0.001, dets_type: str = "xyxy", method: str = 'linear'):
		"""Pure python implementation of soft NMS as described in the paper
		`Improving Object Detection With One Line of Code`_.

		Args:
			dets (numpy.array | list): Detection results with shape `(num, 4)`,
				data in second dimension are [x1, y1, x2, y2] respectively.
			scores (numpy.array | list): scores for boxes
			iou_thr (float): IOU threshold. Only work when method is `linear`
				or 'greedy'.
			sigma (float): Gaussian function parameter. Only work when method
				is `gaussian`.
			score_thr (float): Boxes that score less than the.
			dets_type: boxes order format - "xyxy", "xywh". Defalut: xyxy
			method (str): Rescore method. Only can be `linear`, `gaussian`
				or 'greedy'.
				
		Returns:
			index of boxes to keep
		"""
		_dets =  np.array(dets) if not isinstance(dets, np.ndarray) else dets.copy()
		_scores = np.array(scores) if not isinstance(scores, np.ndarray) else scores.copy()
		if (_dets.shape[0] > 0) :
			if (dets_type == "xywh") :
				_dets[:, 2:4] = _dets[:, 0:2] + _dets[:, 2:4]
			arg = iou_thr, sigma, score_thr, method
			return NMS().__fast_soft_nms(_dets, _scores, *arg)
		else :
			return []
	
	@staticmethod	
	@jit(nopython=True)
	def __fast_soft_nms(dets: np.array, sc: np.array, iou_thr: float = 0.3, 
					sigma: float = 0.5, score_thr: float = 0.001, method: str = 'linear'):
		if dets.shape[0] == 1:
			return np.zeros(1).astype(np.int32)
		
		# indexes concatenate boxes with the last column
		N = dets.shape[0]
		indexes = np.arange(N).reshape(N, 1) # indexes = np.array([np.arange(N)])
		dets = np.concatenate((dets, indexes), axis=1) # dets = np.concatenate((dets, indexes.T), axis=1)

		# the order of boxes coordinate is [y1,x1,y2,x2]
		y1 = dets[:, 0]
		x1 = dets[:, 1]
		y2 = dets[:, 2]
		x2 = dets[:, 3]
		scores = sc
		areas = (x2 - x1 + 1) * (y2 - y1 + 1)

		for i in range(N):
			tBD = dets[i, :]
			tscore = sc[i]
			tarea = areas[i]
			pos = i + 1

			if i != N - 1:
				maxscore = np.max(sc[pos:])
				maxpos = np.argmax(sc[pos:]) + pos
			else:
				maxscore = scores[-1]
				maxpos = 0
			if tscore < maxscore:
				dets[i, :], dets[maxpos, :] = dets[maxpos, :], tBD
				scores[i], scores[maxpos] = scores[maxpos], tscore
				areas[i], areas[maxpos] = areas[maxpos], tarea

			# IoU calculate
			xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
			yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
			xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
			yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

			w = np.maximum(0.0, xx2 - xx1 + 1)
			h = np.maximum(0.0, yy2 - yy1 + 1)
			inter = w * h
			ovr = inter / (areas[i] + areas[pos:] - inter)

			# Three methods: 1.linear 2.gaussian 3.original NMS
			if method == 1:  # linear
				weight = np.ones(ovr.shape)
				weight[ovr > iou_thr] = weight[ovr > iou_thr] - ovr[ovr > iou_thr]
			elif method == 2:  # gaussian
				weight = np.exp(-(ovr * ovr) / sigma)
			else:  # original NMS
				weight = np.ones(ovr.shape)
				weight[ovr > iou_thr] = 0

			scores[pos:] = weight * scores[pos:]

		# select the boxes and keep the corresponding indexes
		keep = dets[:, 4][scores > score_thr]

		return keep.astype(np.int32)
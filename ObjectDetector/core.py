from __future__ import annotations

import cv2
import abc
from typing import *
from dataclasses import dataclass, field

@dataclass
class RectInfo:
	x: float
	y: float
	width: float
	height: float
	conf: float
	label: str
	kpss: List[Tuple[int, int]] = field(default_factory=list)

	def tolist(self, dtype = int, format_type: str = "xyxy"):
		if (format_type == "xyxy"):
			temp = [self.x, self.y, self.x + self.width, self.y + self.height]
		else :
			temp = [self.x, self.y, self.width, self.height]
		return list(map(dtype, temp))

	def pad(self, padding: int) -> RectInfo:
		return RectInfo(
			x=self.x - padding,
			y=self.y - padding,
			width=self.width + 2 * padding,
			height=self.height + 2 * padding,
			conf=self.conf,
			label=self.label,
			kpss=self.kpss)

class ObjectDetectBase(abc.ABC):
	_defaults = {
		"model_path": None,
		"model_type" : None,
		"classes_path" : None,
		"box_score" : None,
	}

	@classmethod
	def set_defaults(cls, config) :
		cls._defaults = config

	@classmethod
	def check_defaults(cls):
		return cls._defaults
		
	@classmethod
	def get_defaults(cls, n):
		if n in cls._defaults:
			return cls._defaults[n]
		else:
			return "Unrecognized attribute name '" + n + "'"
		
	def __init__(self, logger):
		self.__dict__.update(self._defaults) # set up default values
		self.logger = logger

	@property
	def object_info(self) :
		if not hasattr(self, '_object_info') :
			self._object_info = []
			self.logger.war("Can't get object information, maybe you forget to use detect api.")
		else :
			for _info in self._object_info:
				if not isinstance(_info, RectInfo):
					self.logger.war("'object_info' have unrecognized type.")
		return self._object_info
	
	def set_input_details(self, engine) -> None :
		if hasattr(engine, "get_engine_input_shape"):
			self.input_shapes = engine.get_engine_input_shape()
			self.input_types = engine.engine_dtype

			self.channes, self.input_height, self.input_width = self.input_shapes[1:]
			if (self.logger) : 
				self.logger.info(f"-> Input Shape : {self.input_shapes}")
				self.logger.info(f"-> Input Type  : {self.input_types}")
		else :
			self.logger.error(f"engine does not adhere to the naming convention of the 'EngineBase' class")

	def set_output_details(self, engine) -> None :
		if hasattr(engine, "get_engine_output_shape"):
			self.output_shapes, self.output_names = engine.get_engine_output_shape()
			if (self.logger) : 
				self.logger.info(f"-> Output Shape : {self.output_shapes}")
		else :
			self.logger.error(f"engine does not adhere to the naming convention of the 'EngineBase' class")

	@staticmethod
	def cornerRect(img, bbox : list, t : int = 5, rt : int = 1, colorR : tuple = (255, 0, 255), colorC : tuple = (0, 255, 0)):
		xmin, ymin, xmax, ymax = bbox
		l = max(1, int(min( (ymax-ymin), (xmax-xmin))*0.2))

		if rt != 0:
			cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colorR, rt)
		# Top Left  xmin, ymin
		cv2.line(img,  (xmin, ymin), (xmin + l, ymin), colorC, t)
		cv2.line(img,  (xmin, ymin), (xmin, ymin + l), colorC, t)
		# Top Right  xmax, ymin
		cv2.line(img, (xmax, ymin), (xmax - l, ymin), colorC, t)
		cv2.line(img, (xmax, ymin), (xmax, ymin + l), colorC, t)
		# Bottom Left  xmin, ymax
		cv2.line(img, (xmin, ymax), (xmin + l, ymax), colorC, t)
		cv2.line(img, (xmin, ymax), (xmin, ymax - l), colorC, t)
		# Bottom Right  xmax, ymax
		cv2.line(img, (xmax, ymax), (xmax - l, ymax), colorC, t)
		cv2.line(img, (xmax, ymax), (xmax, ymax - l), colorC, t)

		return img
	
	@abc.abstractmethod
	def DetectFrame(self):
		return NotImplemented	
	
	@abc.abstractmethod
	def DrawDetectedOnFrame(self):
		return NotImplemented	

import cv2
import abc
import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class LaneInfo:
	_lanes_points : np.ndarray  # Detect Road Points
	_lanes_status : np.ndarray	# Detect Road Status
	_area_points  : np.ndarray	# Arterial Road Points
	_area_status  : bool 		# Arterial Road Status
	
	@property
	def lanes_points(self):
		return self._lanes_points

	@lanes_points.setter
	def lanes_points(self, arr : np.ndarray ) -> None:
		if isinstance(arr, np.ndarray) :
			self._lanes_points = arr
		else :
			raise Exception("The 'lanes_points' must be np.array[List[Tuple[x, y], ...], ...].")

	@property
	def lanes_status(self):
		return self._lanes_status

	@lanes_status.setter
	def lanes_status(self, value : list ) -> None:
		for v in value :
			if type(v)!=bool:
				raise Exception("The elements of 'lanes_status' must be of type bool List[bool, ...].")
		self._lanes_status = value

	@property
	def area_status(self):
		return self._area_status

	@area_status.setter
	def area_status(self, value : bool) -> None:
		raise Exception("You need to use the '__update_lanes_status' API to modify it.")

	@property
	def area_points(self):
		return self._area_points

	@area_points.setter
	def area_points(self, value : bool) -> None:
		raise Exception("You need to use the '__update_lanes_area' API to modify it.")
	
class LaneDetectBase(abc.ABC):
	_defaults = {
		"model_path": None,
		"model_type" : None,
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
		self.adjust_lanes = False
		self.lane_info = LaneInfo(	np.array([], dtype=object), 
									np.array([], dtype=object), 
									np.array([], dtype=object),
									False)

	def set_input_details(self, engine) -> None :
		if hasattr(engine, "get_engine_input_shape"):
			self.input_shape = engine.get_engine_input_shape()
			self.input_types = engine.engine_dtype

			self.channes, self.input_height, self.input_width = self.input_shape[1:]
			if (self.logger) : 
				self.logger.info(f"-> Input Shape : {self.input_shape}")
				self.logger.info(f"-> Input Type  : {self.input_types}")
		else :
			self.logger.error(f"engine does not adhere to the naming convention of the 'EngineBase' class")

	def set_output_details(self, engine) -> None :
		if hasattr(engine, "get_engine_output_shape"):
			self.output_shape, self.output_names = engine.get_engine_output_shape()
		else :
			self.logger.error(f"engine does not adhere to the naming convention of the 'EngineBase' class")
	
	@staticmethod
	def __adjust_lanes_points(left_lanes_points : list, right_lanes_points : list, image_height : list) -> Tuple[list, list]:
		# 多项式拟合
		if (len(left_lanes_points[1]) != 0 ) :
			leftx, lefty  = list(zip(*left_lanes_points))
			if len(lefty) > 10:
				left_fit = np.polyfit(lefty, leftx, 2)
			else :
				return left_lanes_points, right_lanes_points
		else :
			return left_lanes_points, right_lanes_points
		if (len(right_lanes_points) != 0 ) :
			rightx, righty  = list(zip(*right_lanes_points))
			if len(righty) > 10:
				right_fit = np.polyfit(righty, rightx, 2)
			else :
				return left_lanes_points, right_lanes_points
		else :
			return left_lanes_points, right_lanes_points


		# Generate x and y values for plotting
		maxy = image_height - 1
		miny = image_height // 3
		if len(lefty):
			maxy = max(maxy, np.max(lefty))
			miny = min(miny, np.min(lefty))

		if len(righty):
			maxy = max(maxy, np.max(righty))
			miny = min(miny, np.min(righty))
		both_fity = np.linspace(miny, maxy, image_height)

		left_fitx = left_fit[0]*both_fity**2 + left_fit[1]*both_fity + left_fit[2]
		right_fitx = right_fit[0]*both_fity**2 + right_fit[1]*both_fity + right_fit[2]

		# fix lanes points
		fix_left_lanes_points = [ (int(l), int(y)) for l, y in zip(left_fitx, both_fity) if (y >= min(lefty) and l >= 0)]
		fix_right_lanes_points = [ (int(r), int(y)) for r, y in zip(right_fitx, both_fity) if (y >= min(righty) and r >= 0)]
		return fix_left_lanes_points, fix_right_lanes_points

	def __update_lanes_status(self, lanes_status : list) -> None :
		self.lane_info._area_status = False
		if(lanes_status != [] and len(lanes_status) % 2 == 0) :
			index = len(lanes_status) // 2
			if(lanes_status[index-1] and lanes_status[index]):
				self.lane_info._area_status = True

	def __update_lanes_area(self, lanes_points: np.ndarray, img_height: int) -> None :
		self.lane_info._area_points = np.array([], dtype=object)
		if (self.lane_info._area_status) :
			index = len(lanes_points) // 2
			if (self.adjust_lanes) :
				left_lanes_points, right_lanes_points = self.__adjust_lanes_points(lanes_points[index-1], lanes_points[index], img_height)
			else :
				left_lanes_points, right_lanes_points = lanes_points[index-1], lanes_points[index]
			self.lane_info._area_points = np.vstack((left_lanes_points, np.flipud(right_lanes_points)))

	@abc.abstractmethod
	def DetectFrame(self):
		return NotImplemented	
	
	@abc.abstractmethod
	def DrawDetectedOnFrame(self):
		return NotImplemented	
	
	@abc.abstractmethod
	def DrawAreaOnFrame(self):
		return NotImplemented	

	def AutoDrawLanes(self, image : cv2, draw_points : bool = True, draw_area : bool = True) -> cv2:
		self.DetectFrame(image, adjust_lanes=True)

		if (draw_points) :
			self.DrawDetectedOnFrame(image)

		if (draw_area) :
			self.DrawAreaOnFrame(image)
		return image
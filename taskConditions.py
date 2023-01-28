
import numpy as np
import logging
import ctypes
from ObjectDetector.utils import CollisionType
from TrafficLaneDetector.ultrafastLaneDetector.utils import OffsetType, CurvatureType

STD_OUTPUT_HANDLE= -11
std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
def set_color(color, handle=std_out_handle):
	bool = ctypes.windll.kernel32.SetConsoleTextAttribute(handle, color)
	return bool

class Logger:
	FOREGROUND_WHITE = 0x0007
	FOREGROUND_BLUE = 0x01 # text color contains blue.
	FOREGROUND_GREEN= 0x02 # text color contains green.
	FOREGROUND_RED  = 0x04 # text color contains red.
	FOREGROUND_YELLOW = FOREGROUND_RED | FOREGROUND_GREEN

	def __init__(self, path, clevel = logging.DEBUG, Flevel = logging.DEBUG):
		self.logger = logging.getLogger(path)
		self.logger.setLevel(logging.DEBUG)
		self.clevel = clevel
		fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
		#設置CMD日誌
		sh = logging.StreamHandler()
		sh.setFormatter(fmt)
		sh.setLevel(clevel)
		self.logger.addHandler(sh)
		#設置文件日誌
		if (path != None) :
			fh = logging.FileHandler(path)
			fh.setFormatter(fmt)
			fh.setLevel(Flevel)
			self.logger.addHandler(fh)

	def changelevel(self, clevel) :
		self.clevel = clevel
		self.logger.setLevel(clevel)

	def debug(self,message):
		self.logger.debug(message)
 
	def info(self,message,color=FOREGROUND_BLUE):
		set_color(color)
		self.logger.info(message)
		set_color(self.FOREGROUND_WHITE)
 
	def war(self,message,color=FOREGROUND_YELLOW):
		set_color(color)
		self.logger.warn(message)
		set_color(self.FOREGROUND_WHITE)
 
	def error(self,message,color=FOREGROUND_RED):
		set_color(color)
		self.logger.error(message)
		set_color(self.FOREGROUND_WHITE)
 
	def cri(self,message):
		self.logger.critical(message)

class TaskConditions(object):
	
	def __init__(self):
		self.collision_msg = CollisionType.UNKNOWN
		self.offset_msg = OffsetType.UNKNOWN
		self.curvature_msg = CurvatureType.UNKNOWN
		self.vehicle_collision_record = []
		self.vehicle_offset_record = []
		self.vehicle_curvature_record = []
		self.transform_status = None

		self.toggle_status = "Default"
		self.toggle_oscillator_status = [False, False]
		self.toggle_status_counter = {"Offset" : 0,  "Curvae" : 0, "BirdViewAngle" : 0}


	# Calibration road when curvae smooth 
	def _calibration_curve(self, vehicle_curvature, frequency=3, curvae_thres=15000):
		# print(vehicle_curvature)
		if (self.toggle_status_counter["BirdViewAngle"] <= frequency) :
			if (vehicle_curvature >= curvae_thres ) :
				self.toggle_status_counter["BirdViewAngle"]  += 1
			else :
				self.toggle_status_counter["BirdViewAngle"]  = 0
		else :
			self.toggle_status_counter["BirdViewAngle"]  = 0
			self.toggle_status = "Default"


	def _calc_deviation(self, offset, offset_thres):
		if ( abs(offset) > offset_thres ) :
			if (offset > 0 and self.curvature_msg not in {CurvatureType.HARD_LEFT, CurvatureType.EASY_LEFT} ) :
				msg = OffsetType.RIGHT
			elif (offset < 0 and self.curvature_msg not in {CurvatureType.HARD_RIGHT, CurvatureType.EASY_RIGHT} ) :
				msg = OffsetType.LEFT
			else :
				msg = OffsetType.UNKNOWN
		else :
			msg = OffsetType.CENTER

		return msg


	def _calc_direction(self, curvature, direction, curvae_thres):
		if (curvature <= curvae_thres) :
			if (direction == "L" and self.curvature_msg != CurvatureType.EASY_RIGHT) :
				msg = CurvatureType.HARD_LEFT
			elif (direction == "R" and self.curvature_msg != CurvatureType.EASY_LEFT) :
				msg = CurvatureType.HARD_RIGHT
			else :
				msg = CurvatureType.UNKNOWN
		else :
			if (direction == "L") :
				msg = CurvatureType.EASY_LEFT
			elif (direction == "R") :
				msg = CurvatureType.EASY_RIGHT
			else :
				msg = CurvatureType.STRAIGHT
		return msg


	def CheckStatus(self) :
		if (self.curvature_msg == CurvatureType.UNKNOWN and self.offset_msg == OffsetType.UNKNOWN) :
			self.toggle_oscillator_status = [False, False]

		if self.toggle_status != self.transform_status :
			self.transform_status = self.toggle_status
			self.toggle_status = None
			return True
		else :
			return False


	def CheckOffsetStatus(self, vehicle_offset, offset_thres=0.9) :
		if (vehicle_offset != None) :
			self.vehicle_offset_record.append(vehicle_offset)
			if len(self.vehicle_offset_record) > 5:
				self.vehicle_offset_record.pop(0)
				avg_vehicle_offset = np.median(self.vehicle_offset_record)
				self.offset_msg = self._calc_deviation(avg_vehicle_offset, offset_thres)

				plus = [i for i in self.vehicle_offset_record if i > 0.2]
				mius = [i for i in self.vehicle_offset_record if i < -0.2]
				# print(plus, mius)
				# print(left_right_status, vehicle_curvature_count)
				if (self.toggle_status_counter["Offset"] >= 10) :
					if ( len(plus) == len(self.vehicle_offset_record) ) :
						self.toggle_oscillator_status[0] = True
						self.toggle_status_counter["Offset"] = 0
					if (len(mius) == len(self.vehicle_offset_record) ) :
						self.toggle_oscillator_status[1] = True
						self.toggle_status_counter["Offset"] = 0
					if (np.array(self.toggle_oscillator_status).all() ) :
						self.toggle_status = "Top"
						self.toggle_oscillator_status = [False, False]
					else :
						self.toggle_status_counter["Offset"] = 0
				else :
					self.toggle_status_counter["Offset"] += 1
			else :
				self.offset_msg = OffsetType.UNKNOWN
		else :
			self.offset_msg = OffsetType.UNKNOWN
			self.vehicle_offset_record = []


	def CheckRouteStatus(self, vehicle_direction, vehicle_curvature, curvae_thres=500) :
		if (vehicle_curvature != None) :
			if (vehicle_direction != None and self.offset_msg == OffsetType.CENTER) :
				self.vehicle_curvature_record.append([vehicle_direction, vehicle_curvature])

				if len(self.vehicle_curvature_record) > 10:
					self.vehicle_curvature_record.pop(0)
					avg_direction = max(set(np.squeeze(self.vehicle_curvature_record)[:,0]), key = self.vehicle_curvature_record.count)
					avg_curvature = np.median([int(float(i)) for i in np.array(self.vehicle_curvature_record)[:, 1]])
					self.curvature_msg = self._calc_direction(avg_curvature, avg_direction, curvae_thres)

					if (self.toggle_status_counter["Curvae"] >= 10) :
						if (self.curvature_msg != CurvatureType.STRAIGHT and abs(self.vehicle_offset_record[-1]) < 0.2 and not np.array(self.toggle_oscillator_status).any()) :
							self.toggle_status = "Bottom"
						else :
							self.toggle_status_counter["Curvae"] = 0
					else :
						self.toggle_status_counter["Curvae"] += 1

				else :
					self.curvature_msg = CurvatureType.UNKNOWN
			else :
				self.vehicle_curvature_record = []
				self.curvature_msg = CurvatureType.UNKNOWN

			self._calibration_curve(vehicle_curvature)

		else :
			self.vehicle_curvature_record = []
			self.curvature_msg = CurvatureType.UNKNOWN


	def CheckCollisionStatus(self, vehicle_distance, lane_area, distance_thres=1.5) : 
		if (vehicle_distance != None) :
			x, y, d = vehicle_distance
			self.vehicle_collision_record.append(d)
			if len(self.vehicle_collision_record) > 3:
				self.vehicle_collision_record.pop(0)
				avg_vehicle_collision = np.median(self.vehicle_collision_record)
				if ( avg_vehicle_collision <= distance_thres) :
					self.collision_msg = CollisionType.WARNING
				elif ( distance_thres < avg_vehicle_collision <= 2*distance_thres) :
					self.collision_msg = CollisionType.PROMPT
				else :
					self.collision_msg = CollisionType.NORMAL
		else :
			if (lane_area) :
				self.collision_msg = CollisionType.NORMAL
			else :
				self.collision_msg = CollisionType.UNKNOWN
			self.vehicle_collision_record = []
import cv2, time
import numpy as np
import logging
import pycuda.driver as drv

from taskConditions import TaskConditions, Logger
from ObjectDetector.yoloDetector import YoloDetector
from ObjectDetector.utils import ObjectModelType,  CollisionType
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure

from TrafficLaneDetector.ultrafastLaneDetector.ultrafastLaneDetector import UltrafastLaneDetector
from TrafficLaneDetector.ultrafastLaneDetector.ultrafastLaneDetectorV2 import UltrafastLaneDetectorV2
from TrafficLaneDetector.ultrafastLaneDetector.perspectiveTransformation import PerspectiveTransformation
from TrafficLaneDetector.ultrafastLaneDetector.utils import LaneModelType, OffsetType, CurvatureType
LOGGER = Logger(None, logging.INFO, logging.INFO )

video_path = "./TrafficLaneDetector/temp/demo-1.mp4"
lane_config = {
	"model_path": "./TrafficLaneDetector/models/culane_res18_fp16.trt",
	"model_type" : LaneModelType.UFLDV2_CULANE
}

object_config = {
	"model_path": './ObjectDetector/models/yolov8m-coco_fp16.trt',
	"model_type" : ObjectModelType.YOLOV8,
	"classes_path" : './ObjectDetector/models/coco_label.txt',
	"box_score" : 0.4,
	"box_nms_iou" : 0.45
}

# Priority : FCWS > LDWS > LKAS
class ControlPanel(object):
	CollisionDict = {
						CollisionType.UNKNOWN : (0, 255, 255),
						CollisionType.NORMAL : (0, 255, 0),
						CollisionType.PROMPT : (0, 102, 255),
						CollisionType.WARNING : (0, 0, 255)
	 				}

	OffsetDict = { 
					OffsetType.UNKNOWN : (0, 255, 255), 
					OffsetType.RIGHT :  (0, 0, 255), 
					OffsetType.LEFT : (0, 0, 255), 
					OffsetType.CENTER : (0, 255, 0)
				 }

	CurvatureDict = { 
						CurvatureType.UNKNOWN : (0, 255, 255),
						CurvatureType.STRAIGHT : (0, 255, 0),
						CurvatureType.EASY_LEFT : (0, 102, 255),
						CurvatureType.EASY_RIGHT : (0, 102, 255),
						CurvatureType.HARD_LEFT : (0, 0, 255),
						CurvatureType.HARD_RIGHT : (0, 0, 255)
					}

	def __init__(self):
		collision_warning_img = cv2.imread('./assets/FCWS-warning.png', cv2.IMREAD_UNCHANGED)
		self.collision_warning_img = cv2.resize(collision_warning_img, (100, 100))
		collision_prompt_img = cv2.imread('./assets/FCWS-prompt.png', cv2.IMREAD_UNCHANGED)
		self.collision_prompt_img = cv2.resize(collision_prompt_img, (100, 100))
		collision_normal_img = cv2.imread('./assets/FCWS-normal.png', cv2.IMREAD_UNCHANGED)
		self.collision_normal_img = cv2.resize(collision_normal_img, (100, 100))
		left_curve_img = cv2.imread('./assets/left_turn.png', cv2.IMREAD_UNCHANGED)
		self.left_curve_img = cv2.resize(left_curve_img, (200, 200))
		right_curve_img = cv2.imread('./assets/right_turn.png', cv2.IMREAD_UNCHANGED)
		self.right_curve_img = cv2.resize(right_curve_img, (200, 200))
		keep_straight_img = cv2.imread('./assets/straight.png', cv2.IMREAD_UNCHANGED)
		self.keep_straight_img = cv2.resize(keep_straight_img, (200, 200))
		determined_img = cv2.imread('./assets/warn.png', cv2.IMREAD_UNCHANGED)
		self.determined_img = cv2.resize(determined_img, (200, 200))
		left_lanes_img = cv2.imread('./assets/LTA-left_lanes.png', cv2.IMREAD_UNCHANGED)
		self.left_lanes_img = cv2.resize(left_lanes_img, (300, 200))
		right_lanes_img = cv2.imread('./assets/LTA-right_lanes.png', cv2.IMREAD_UNCHANGED)
		self.right_lanes_img = cv2.resize(right_lanes_img, (300, 200))


		# FPS
		self.fps = 0
		self.frame_count = 0
		self.start = time.time()

		self.curve_status = None

	def _updateFPS(self) :
		"""
		Update FPS.

		Args:
			None

		Returns:
			None
		"""
		self.frame_count += 1
		if self.frame_count >= 30:
			self.end = time.time()
			self.fps = self.frame_count / (self.end - self.start)
			self.frame_count = 0
			self.start = time.time()

	def DisplayBirdViewPanel(self, main_show, min_show, show_ratio=0.25) :
		"""
		Display BirdView Panel on image.

		Args:
			main_show: video image.
			min_show: bird view image.
			show_ratio: display scale of bird view image.

		Returns:
			main_show: Draw bird view on frame.
		"""
		W = int(main_show.shape[1]* show_ratio)
		H = int(main_show.shape[0]* show_ratio)

		min_birdview_show = cv2.resize(min_show, (W, H))
		min_birdview_show = cv2.copyMakeBorder(min_birdview_show, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0]) # 添加边框
		main_show[0:min_birdview_show.shape[0], -min_birdview_show.shape[1]: ] = min_birdview_show

	def DisplaySignsPanel(self, main_show, offset_type, curvature_type) :
		"""
		Display Signs Panel on image.

		Args:
			main_show: image.
			offset_type: offset status by OffsetType. (UNKNOWN/CENTER/RIGHT/LEFT)
			curvature_type: curature status by CurvatureType. (UNKNOWN/STRAIGHT/HARD_LEFT/EASY_LEFT/HARD_RIGHT/EASY_RIGHT)

		Returns:
			main_show: Draw sings info on frame.
		"""

		W = 400
		H = 365
		widget = np.copy(main_show[:H, :W])
		widget //= 2
		widget[0:3,:] = [0, 0, 255]  # top
		widget[-3:-1,:] = [0, 0, 255] # bottom
		widget[:,0:3] = [0, 0, 255]  #left
		widget[:,-3:-1] = [0, 0, 255] # right
		main_show[:H, :W] = widget

		if curvature_type == CurvatureType.UNKNOWN and offset_type in { OffsetType.UNKNOWN, OffsetType.CENTER } :
			y, x = self.determined_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.determined_img[y, x, :3]
			self.curve_status = None

		elif (curvature_type == CurvatureType.HARD_LEFT or self.curve_status== "Left") and \
			(curvature_type not in { CurvatureType.EASY_RIGHT, CurvatureType.HARD_RIGHT }) :
			y, x = self.left_curve_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.left_curve_img[y, x, :3]
			self.curve_status = "Left"

		elif (curvature_type == CurvatureType.HARD_RIGHT or self.curve_status== "Right") and \
			(curvature_type not in { CurvatureType.EASY_LEFT, CurvatureType.HARD_LEFT }) :
			y, x = self.right_curve_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.right_curve_img[y, x, :3]
			self.curve_status = "Right"
		
		
		if ( offset_type == OffsetType.RIGHT ) :
			y, x = self.left_lanes_img[:,:,2].nonzero()
			main_show[y+10, x-150+W//2] = self.left_lanes_img[y, x, :3]
		elif ( offset_type == OffsetType.LEFT ) :
			y, x = self.right_lanes_img[:,:,2].nonzero()
			main_show[y+10, x-150+W//2] = self.right_lanes_img[y, x, :3]
		elif curvature_type == CurvatureType.STRAIGHT or self.curve_status == "Straight" :
			y, x = self.keep_straight_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.keep_straight_img[y, x, :3]
			self.curve_status = "Straight"

		self._updateFPS()
		cv2.putText(main_show, "LDWS : " + offset_type.value, (10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=self.OffsetDict[offset_type], thickness=2)
		cv2.putText(main_show, "LKAS : " + curvature_type.value, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=self.CurvatureDict[curvature_type], thickness=2)
		cv2.putText(main_show, "FPS  : %.2f" % self.fps, (10, widget.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

	def DisplayCollisionPanel(self, main_show, collision_type, obect_infer_time, lane_infer_time, show_ratio=0.25) :
		"""
		Display Collision Panel on image.

		Args:
			main_show: image.
			collision_type: collision status by CollisionType. (WARNING/PROMPT/NORMAL)
			obect_infer_time: object detection time -> float.
			lane_infer_time:  lane detection time -> float.

		Returns:
			main_show: Draw collision info on frame.
		"""

		W = int(main_show.shape[1]* show_ratio)
		H = int(main_show.shape[0]* show_ratio)

		widget = np.copy(main_show[H+20:2*H, -W-20:])
		widget //= 2
		widget[0:3,:] = [0, 0, 255]  # top
		widget[-3:-1,:] = [0, 0, 255] # bottom
		widget[:,-3:-1] = [0, 0, 255] #left
		widget[:,0:3] = [0, 0, 255]  # right
		main_show[H+20:2*H, -W-20:] = widget

		if (collision_type == CollisionType.WARNING) :
			y, x = self.collision_warning_img[:,:,3].nonzero()
			main_show[H+y+50, (x-W-5)] = self.collision_warning_img[y, x, :3]
		elif (collision_type == CollisionType.PROMPT) :
			y, x =self.collision_prompt_img[:,:,3].nonzero()
			main_show[H+y+50, (x-W-5)] = self.collision_prompt_img[y, x, :3]
		elif (collision_type == CollisionType.NORMAL) :
			y, x = self.collision_normal_img[:,:,3].nonzero()
			main_show[H+y+50, (x-W-5)] = self.collision_normal_img[y, x, :3]

		cv2.putText(main_show, "FCWS : " + collision_type.value, ( main_show.shape[1]- int(W) + 100 , 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=self.CollisionDict[collision_type], thickness=2)
		cv2.putText(main_show, "object-infer : %.2f s" % obect_infer_time, ( main_show.shape[1]- int(W) + 100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
		cv2.putText(main_show, "lane-infer : %.2f s" % lane_infer_time, ( main_show.shape[1]- int(W) + 100, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)


if __name__ == "__main__":

	# Initialize read and save video 
	cap = cv2.VideoCapture(video_path)
	if (not cap.isOpened()) :
		raise Exception("video path is error. please check it.")
	width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	vout = cv2.VideoWriter(video_path[:-4]+'_out.mp4', fourcc , 30.0, (width, height))
	cv2.namedWindow("ADAS Simulation", cv2.WINDOW_NORMAL)	
	
	#==========================================================
	#					Initialize Class
	#==========================================================
	LOGGER.info("[Pycuda] Cuda Version: {}".format(drv.get_version()))
	LOGGER.info("[Driver] Cuda Version: {}".format(drv.get_driver_version()))
	LOGGER.info("-"*40)

	# lane detection model
	LOGGER.info("UfldDetector Model Type : {}".format(lane_config["model_type"].name))
	if ( "UFLDV2" in lane_config["model_type"].name) :
		UltrafastLaneDetectorV2.set_defaults(lane_config)
		laneDetector = UltrafastLaneDetectorV2(logger=LOGGER)
	else :
		UltrafastLaneDetector.set_defaults(lane_config)
		laneDetector = UltrafastLaneDetector(logger=LOGGER)
	transformView = PerspectiveTransformation( (width, height) , logger=LOGGER)

	# object detection model
	LOGGER.info("YoloDetector Model Type : {}".format(object_config["model_type"].name))
	YoloDetector.set_defaults(object_config)
	objectDetector = YoloDetector(logger=LOGGER)
	distanceDetector = SingleCamDistanceMeasure()

	# display panel
	displayPanel = ControlPanel()
	analyzeMsg = TaskConditions()
	while cap.isOpened():

		ret, frame = cap.read() # Read frame from the video
		if ret:	
			frame_show = frame.copy()

			#========================== Detect Model =========================
			obect_time = time.time()
			objectDetector.DetectFrame(frame)
			obect_infer_time = round(time.time() - obect_time, 2)
			lane_time = time.time()
			laneDetector.DetectFrame(frame)
			lane_infer_time = round(time.time() - lane_time, 4)

			#========================= Analyze Status ========================
			distanceDetector.calcDistance(objectDetector.object_info)
			vehicle_distance = distanceDetector.calcCollisionPoint(laneDetector.draw_area_points)

			analyzeMsg.UpdateCollisionStatus(vehicle_distance, laneDetector.draw_area)


			if (analyzeMsg.CheckStatus() and laneDetector.draw_area ) :
				transformView.updateTransformParams(laneDetector.lanes_points[1], laneDetector.lanes_points[2], analyzeMsg.transform_status)
			birdview_show = transformView.transformToBirdView(frame_show)

			birdview_lanes_points = [transformView.transformToBirdViewPoints(lanes_point) for lanes_point in laneDetector.lanes_points]
			(vehicle_direction, vehicle_curvature) , vehicle_offset = transformView.calcCurveAndOffset(birdview_show, birdview_lanes_points[1], birdview_lanes_points[2])

			analyzeMsg.UpdateOffsetStatus(vehicle_offset)
			analyzeMsg.UpdateRouteStatus(vehicle_direction, vehicle_curvature)

			#========================== Draw Results =========================
			transformView.DrawDetectedOnBirdView(birdview_show, birdview_lanes_points, analyzeMsg.offset_msg)
			if (LOGGER.clevel == logging.DEBUG) : transformView.DrawTransformFrontalViewArea(frame_show)
			laneDetector.DrawDetectedOnFrame(frame_show, analyzeMsg.offset_msg)
			laneDetector.DrawAreaOnFrame(frame_show, displayPanel.CollisionDict[analyzeMsg.collision_msg])
			objectDetector.DrawDetectedOnFrame(frame_show)
			distanceDetector.DrawDetectedOnFrame(frame_show)

			displayPanel.DisplayBirdViewPanel(frame_show, birdview_show)
			displayPanel.DisplaySignsPanel(frame_show, analyzeMsg.offset_msg, analyzeMsg.curvature_msg)	
			displayPanel.DisplayCollisionPanel(frame_show, analyzeMsg.collision_msg, obect_infer_time, lane_infer_time )
			cv2.imshow("ADAS Simulation", frame_show)

		else:
			break
		vout.write(frame_show)	
		if cv2.waitKey(1) == ord('q'): # Press key q to stop
			break

	vout.release()
	cap.release()
	cv2.destroyAllWindows()

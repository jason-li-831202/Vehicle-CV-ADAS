import cv2, time
import numpy as np
from taskConditions import TaskConditions
from ObjectDetector.yoloDetectorV5 import YoloDetector
from ObjectDetector.utils import CollisionType
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure
from TrafficLaneDetector.ultrafastLaneDetector.utils import ModelType
from TrafficLaneDetector.ultrafastLaneDetector.ultrafastLane import UltrafastLaneDetector
from TrafficLaneDetector.ultrafastLaneDetector.ultrafastLaneV2 import UltrafastLaneDetectorV2
from TrafficLaneDetector.ultrafastLaneDetector.perspectiveTransformation import PerspectiveTransformation
from TrafficLaneDetector.ultrafastLaneDetector.utils import OffsetType, CurvatureType


video_path = "./TrafficLaneDetector/temp/行車紀錄器-車禍-2.mp4"
lane_config = {
	"model_path": "./TrafficLaneDetector/models/culane_res34.onnx",
	"model_type" : ModelType.UFLDV2_CULANE
}

object_config = {
	"model_path": './ObjectDetector/models/yolov5m-coco.onnx',
	"classes_path" : './ObjectDetector/models/coco_label.txt',
	"box_score" : 0.6,
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
		collision_warning_img = cv2.imread('./TrafficLaneDetector/trafficSigns/FCWS-warning.png', cv2.IMREAD_UNCHANGED)
		self.collision_warning_img = cv2.resize(collision_warning_img, (100, 100))
		collision_prompt_img = cv2.imread('./TrafficLaneDetector/trafficSigns/FCWS-prompt.png', cv2.IMREAD_UNCHANGED)
		self.collision_prompt_img = cv2.resize(collision_prompt_img, (100, 100))
		collision_normal_img = cv2.imread('./TrafficLaneDetector/trafficSigns/FCWS-normal.png', cv2.IMREAD_UNCHANGED)
		self.collision_normal_img = cv2.resize(collision_normal_img, (100, 100))
		left_curve_img = cv2.imread('./TrafficLaneDetector/trafficSigns/left_turn.png', cv2.IMREAD_UNCHANGED)
		self.left_curve_img = cv2.resize(left_curve_img, (200, 200))
		right_curve_img = cv2.imread('./TrafficLaneDetector/trafficSigns/right_turn.png', cv2.IMREAD_UNCHANGED)
		self.right_curve_img = cv2.resize(right_curve_img, (200, 200))
		keep_straight_img = cv2.imread('./TrafficLaneDetector/trafficSigns/straight.png', cv2.IMREAD_UNCHANGED)
		self.keep_straight_img = cv2.resize(keep_straight_img, (200, 200))
		determined_img = cv2.imread('./TrafficLaneDetector/trafficSigns/warn.png', cv2.IMREAD_UNCHANGED)
		self.determined_img = cv2.resize(determined_img, (200, 200))
		left_lanes_img = cv2.imread('./TrafficLaneDetector/trafficSigns/LTA-left_lanes.png', cv2.IMREAD_UNCHANGED)
		self.left_lanes_img = cv2.resize(left_lanes_img, (300, 200))
		right_lanes_img = cv2.imread('./TrafficLaneDetector/trafficSigns/LTA-right_lanes.png', cv2.IMREAD_UNCHANGED)
		self.right_lanes_img = cv2.resize(right_lanes_img, (300, 200))


		# FPS
		self.fps = 0
		self.frame_count = 0
		self.start = time.time()

		self.curve_status = None

	def _updateFPS(self) :
		self.frame_count += 1
		if self.frame_count >= 30:
			self.end = time.time()
			self.fps = self.frame_count / (self.end - self.start)
			self.frame_count = 0
			self.start = time.time()

	def DisplaySignsPanel(self, main_show, offset_type, curvature_type) :
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
		return main_show

	def DisplayCollisionPanel(self, main_show, collision_type, obect_infer_time, lane_infer_time, show_ratio=0.25) :
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
		return main_show


if __name__ == "__main__":
	# Initialize read and save video 
	cap = cv2.VideoCapture(video_path)
	width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	vout = cv2.VideoWriter(video_path[:-4]+'_out.mp4', fourcc , 30.0, (width, height))
	cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	
	
	#==========================================================
	# 					Initialize Class
	#==========================================================
	# lane detection model
	print("Model Type : ", lane_config["model_type"].name)
	if ( "UFLDV2" in lane_config["model_type"].name) :
		UltrafastLaneDetectorV2.set_defaults(lane_config)
		laneDetector = UltrafastLaneDetectorV2()
	else :
		UltrafastLaneDetector.set_defaults(lane_config)
		laneDetector = UltrafastLaneDetector()
	transformView = PerspectiveTransformation( (width, height) )

	# object detection model
	YoloDetector.set_defaults(object_config)
	objectDetector = YoloDetector()
	distanceDetector = SingleCamDistanceMeasure()

	# display panel
	displayPanel = ControlPanel()
	
	analyzeMsg = TaskConditions()
	while cap.isOpened():

		ret, frame = cap.read() # Read frame from the video
		if ret:	
			frame_show = frame.copy()

			# Detect Model
			obect_time = time.time()
			objectDetector.DetectFrame(frame)
			obect_infer_time = round(time.time() - obect_time, 2)
			lane_time = time.time()
			laneDetector.DetectFrame(frame)
			lane_infer_time = round(time.time() - lane_time, 4)


			# Analyze Status 
			distanceDetector.calcDistance(objectDetector.object_info)
			vehicle_distance = distanceDetector.calcCollisionPoint(laneDetector.draw_area_points)

			analyzeMsg.CheckCollisionStatus(vehicle_distance, laneDetector.draw_area)


			if (not laneDetector.draw_area or analyzeMsg.CheckStatus()) :
				transformView.updateParams(laneDetector.lanes_points[1], laneDetector.lanes_points[2], analyzeMsg.transform_status)
			top_view_show = transformView.forward(frame_show)

			adjust_lanes_points = []
			for lanes_point in laneDetector.lanes_points :
				adjust_lanes_point = transformView.transformPoints(lanes_point)
				adjust_lanes_points.append(adjust_lanes_point)

			(vehicle_direction, vehicle_curvature) , vehicle_offset = transformView.calcCurveAndOffset(top_view_show, adjust_lanes_points[1], adjust_lanes_points[2])

			analyzeMsg.CheckOffsetStatus(vehicle_offset)
			analyzeMsg.CheckRouteStatus(vehicle_direction, vehicle_curvature)


			# Draw Results
			transformView.DrawDetectedOnFrame(top_view_show, adjust_lanes_points, analyzeMsg.offset_msg)
			laneDetector.DrawDetectedOnFrame(frame_show, analyzeMsg.offset_msg)
			frame_show = laneDetector.DrawAreaOnFrame(frame_show, displayPanel.CollisionDict[analyzeMsg.collision_msg])
			objectDetector.DrawDetectedOnFrame(frame_show)
			distanceDetector.DrawDetectedOnFrame(frame_show)

			frame_show = transformView.DisplayBirdView(frame_show, top_view_show)
			frame_show = displayPanel.DisplaySignsPanel(frame_show, analyzeMsg.offset_msg, analyzeMsg.curvature_msg)	
			frame_show = displayPanel.DisplayCollisionPanel(frame_show, analyzeMsg.collision_msg, obect_infer_time, lane_infer_time )
			cv2.imshow("Detected lanes", frame_show)

		else:
			break
		vout.write(frame_show)	
		if cv2.waitKey(1) == ord('q'): 		# Press key q to stop
			break

	vout.release()
	cap.release()
	cv2.destroyAllWindows()

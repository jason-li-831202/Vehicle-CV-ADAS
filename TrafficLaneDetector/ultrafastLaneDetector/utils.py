from enum import Enum

class LaneModelType(Enum):
	UFLD_TUSIMPLE = 0
	UFLD_CULANE = 1
	UFLDV2_TUSIMPLE = 2
	UFLDV2_CULANE = 3
	UFLDV2_CURVELANES = 4

class OffsetType(Enum):
	UNKNOWN = "To Be Determined ..."
	RIGHT = "Please Keep Right"
	LEFT = "Please Keep Left"
	CENTER = "Good Lane Keeping"

class CurvatureType(Enum):
	UNKNOWN = "To Be Determined ..."
	STRAIGHT = "Keep Straight Ahead"
	EASY_LEFT =  "Gentle Left Curve Ahead"
	HARD_LEFT = "Hard Left Curve Ahead"
	EASY_RIGHT = "Gentle Right Curve Ahead"
	HARD_RIGHT = "Hard Right Curve Ahead"

lane_colors = [(255, 0, 0),(46,139,87),(50,205,50),(0,255,255)]

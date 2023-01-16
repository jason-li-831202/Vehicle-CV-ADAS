from enum import Enum


class CollisionType(Enum):
	UNKNOWN = "Determined ..."
	NORMAL = "Normal Risk"
	PROMPT = "Prompt Risk"
	WARNING = "Warning Risk"


class ObjectModelType(Enum):
	YOLOV5 = 0
	YOLOV5_LITE = 1
	YOLOV8 = 2

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
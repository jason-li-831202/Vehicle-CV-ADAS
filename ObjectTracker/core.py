""" ObjectTrackBase class """

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Union

import cv2, math
import numpy as np

FONT_SCALE = 6e-4  # Adjust for larger font size in all images
THICKNESS_SCALE = 2e-3  # Adjust for larger thickness in all images

def putText_shadow(img, text, org, fontFace, fontScale, color, thickness=1, shadow_color=(200, 200, 200), shadow_offset=2):
	"""
	Draws text with a shadow effect on the image.

	Parameters:
	- img (np.ndarray): The image array on which to draw the text.
	- text (str): The text to draw on the image.
	- org (tuple): The coordinates of the bottom-left corner of the text string in the image (x, y).
	- fontFace (int): Font type. See cv2.putText() documentation for details.
	- fontScale (float): Font scale factor that is multiplied by the font-specific base size.
	- color (tuple): The color of the text and shadow in BGR format.
	- thickness (int): Thickness of the lines used to draw the text.
	- shadow_color (tuple): The color of the shadow in BGR format.
	- shadow_offset (int): The offset of the shadow from the text.

	Returns:
	- None
	"""
	# Calculate the shadow position
	shadow_org = (org[0] + shadow_offset, org[1] + shadow_offset)
	
	# Draw the shadow text
	cv2.putText(img, text, shadow_org, fontFace-1, fontScale, shadow_color, thickness=thickness+1)

	# Draw the actual text
	cv2.putText(img, text, org, fontFace, fontScale, color, thickness=thickness)

def arrowedLine_shadow(img, start, end, color, thickness=3, tipLength=0.3, shadow_color=(100, 100, 100), shadow_offset=2):
	"""
	Draws an arrowed line with a shadow effect on the image.

	Parameters:
	- img (np.ndarray): The image array on which to draw the arrowed line.
	- start (tuple): The starting point of the arrowed line (x, y).
	- end (tuple): The ending point of the arrowed line (x, y).
	- color (tuple): The color of the arrowed line and shadow in BGR format.
	- thickness (int): The thickness of the arrowed line and shadow.
	- tipLength (float): The length of the arrow tip as a fraction of the arrow length.
	- shadow_color (tuple): The color of the shadow in BGR format.
	- shadow_offset (int): The offset of the shadow from the arrow.

	Returns:
	- None
	"""
	# Calculate the shadow endpoint
	shadow_end = (end[0] - shadow_offset, end[1] + shadow_offset)
	shadow_start = (start[0]- shadow_offset, start[1] + shadow_offset)
	# Draw the shadow arrow
	cv2.arrowedLine(img, shadow_start, shadow_end, shadow_color, thickness=thickness+2, tipLength=tipLength)

	# Draw a smaller arrow on top to simulate the shadow
	shadow_arrow_end = (end[0] - shadow_offset // 2, end[1] - shadow_offset // 2)
	cv2.arrowedLine(img, start, shadow_arrow_end, color, thickness=thickness-1, tipLength=tipLength-0.1)

	cv2.arrowedLine(img, start, end, color, thickness=thickness, tipLength=tipLength)

class ObjectTrackBase(metaclass=ABCMeta):
	""" Object tracking base class.

	Attributes:
		visualize: bool to visualize tracks.
		names: names of classes/labels.
		class_colors: colors associates with classes/labels.
	"""

	def __init__(self, names: Union[List[str], Dict[str, tuple]]):
		""" Initializes base object trackers.

		Args:
			names: list/dict of classes/labels.
		"""
		# Generate class colors for detection visualization
		self.names = names
		if isinstance(self.names, dict):
			self.class_colors = self.names
			self.names = {key:key for key in self.class_colors.keys()}
		else :
			rng = np.random.default_rng()
			self.class_colors = [
				rng.integers(low=0, high=255, size=3, dtype=np.uint8).tolist()
				for _ in self.names
			]

	@abstractmethod
	def update(self) -> List[Any]:
		""" Updates track states.

		Returns:
			A list of active tracks.
		"""
		raise NotImplementedError

	@staticmethod
	def __compute_directions(trajectories: List[int], limit_shift: int = 2) -> List[float]:
		"""
		Computes the direction vectors between consecutive observations in a list of trajectories.
		If the center of a bounding box shifts more than max_center_shift pixels, it is ignored.

		Parameters:
		- trajectories (list): A list of bounding box coordinates representing the trajectories.
							Each trajectory is a list of bounding box coordinates [(x1, y1, x2, y2), ...].
		- limit_shift (int): The minimum allowed shift of a bounding box.

		Returns:
		- list: A list of direction vectors between consecutive observations in each trajectory.
				Each direction vector is a numpy array [dx, dy].
		"""
		directions = []
		for i in range(len(trajectories) - 1):
			current_box = trajectories[i]
			next_box = trajectories[i + 1]
			box_shift = abs(min(next_box - current_box))

			current_center = np.array([(current_box[0] + current_box[2]) / 2, (current_box[1] + current_box[3]) / 2])
			next_center = np.array([(next_box[0] + next_box[2]) / 2, (next_box[1] + next_box[3]) / 2])

			direction_vector = next_center - current_center
			directions.append(direction_vector if box_shift >= limit_shift else [0, 0])
		return directions

	def plot_directions(self, img: np.ndarray, init_point: list, observations: list, class_id: int) -> None:
		"""
		Plots the main direction of movement based on historical observations.

		Parameters:
		- img (np.ndarray): The image array on which to draw the direction arrow.
		- init_point (list): The initial point (center) of the tracked object. in the format
							  [cx, cy, aspect ratio, height].
		- observations (list): A list of bounding box coordinates representing the historical
								observations of a tracked object. Each observation is in the format
								[(x1, y1, x2, y2), ...].
		- class_id (int): The unique identifier of the tracked object for color consistency in visualization.

		Returns:
		- None
		"""
		lock_count = 5
		directions = self.__compute_directions(observations)
		
		if (len(observations ) > 1 ):
			cx, cy, rate, h = init_point
			w = h*rate

			# Judgment indicator box
			if (len(directions) < lock_count):
				rate_w = (cx - (cx-w//2))/ lock_count
				rate_h = (cy - (cy-h//2))/ lock_count

				sx = int(cx-w//2 + rate_w*len(directions) )
				sy = int(cy-h//2 + rate_h*len(directions) )
				ex = int(cx+w//2 - rate_w*len(directions) )
				ey = int(cy+h//2 - rate_h*len(directions) )
				cv2.rectangle(img, (sx, sy), (ex, ey), tuple( i-10 for i in self.class_colors[class_id]), 2, cv2.LINE_8) 
			else :
				arrow_length = 1000*min(( (h*w) / (img.shape[0]* img.shape[1])), 0.02)
				mean_direction = np.median(directions, axis=0)
				
				end_point = (int(cx + mean_direction[0] * arrow_length), int(cy + mean_direction[1] * arrow_length))
				# cv2.arrowedLine(img, (int(cx), int(cy)), end_point, (255, 255, 255), thickness=3, tipLength=0.3)
				arrowedLine_shadow(img, (int(cx), int(cy)), end_point, (255, 255, 255), thickness=3, tipLength=0.3)

	def plot_trajectories(self, img: np.ndarray, observations: list, class_id: int, track_id: int) -> None:
		"""
		Draws the trajectories of tracked objects based on historical observations. Each point
		in the trajectory is represented by a circle, with the thickness increasing for more
		recent observations to visualize the path of movement.

		Parameters:
		- img (np.ndarray): The image array on which to draw the trajectories.
		- observations (list): A list of bounding box coordinates representing the historical
		observations of a tracked object. Each observation is in the format [(x1, y1, x2, y2), ...].
		- class_id (int): The unique identifier of the tracked object for color consistency in visualization.

		Returns:
		- np.ndarray: The image array with the trajectories drawn on it.
		"""
		if (len(observations ) > 1 ):
			for i, box in enumerate(observations):
				cx, ey = int((box[0] + box[2]) / 2), int(box[3]) # int((box[1] + box[3]) / 2)
				cv2.circle(
					img,
					(cx, ey), 
					int(np.sqrt(float (i + 1)) * 0.5),
					color=self.class_colors[class_id],
					thickness=int(np.sqrt(float (i + 1)) * 1.2)
				)

			foontSize = min(1, sum(box[2:]) * FONT_SCALE)
			# cv2.putText(img, f'ID: {str(track_id)}', (int(box[0]+10*foontSize), int(box[1]+30*foontSize)), 
			#             fontFace = cv2.FONT_HERSHEY_TRIPLEX,
			#             fontScale = min(1, sum(box[2:]) * FONT_SCALE),
			#             thickness = 2,
			#             color = self.class_colors[class_id])
			putText_shadow(img, f'ID: {str(track_id)}', (int(box[0]+10*foontSize), int(box[1]+30*foontSize)), 
							fontFace = cv2.FONT_HERSHEY_TRIPLEX,
							fontScale = min(1, sum(box[2:]) * FONT_SCALE),
							color = self.class_colors[class_id],
							thickness = 1,
							shadow_color=tuple( i-30 for i in self.class_colors[class_id]))

	def plot_bbox(self, img: np.ndarray, observation: np.ndarray, class_id: int, track_id: int) -> None:
		"""
		Plots the bounding box of a tracked object on the image along with the class label and track ID.

		Parameters:
		- img (np.ndarray): The image array on which to draw the bounding box and text.
		- observation (np.ndarray): The bounding box coordinates of the tracked object in the format
									  [x1, y1, width, height].
		- class_id (int): The unique identifier of the class for color and label consistency.
		- track_id (int): The unique identifier of the track.

		Returns:
		- None
		"""
		if (len(observation ) > 1 ):
			tx1, ty1, tw, th = observation.astype(int)

			# reshape bounding box to image
			x1 = max(0, tx1)
			y1 = max(0, ty1)
			x2 = min(img.shape[1], tx1 + tw)
			y2 = min(img.shape[0], ty1 + th)

			cv2.putText(img, f'{self.names[class_id]} : {str(track_id)}', (tx1, ty1 - 10),
						fontFace = cv2.FONT_HERSHEY_TRIPLEX,
						fontScale = min(1, tw*th) * FONT_SCALE,
						thickness = math.ceil(min(*img.shape[:2])* THICKNESS_SCALE),
						color = self.class_colors[class_id])
			cv2.rectangle(img, (x1, y1), (x2, y2), self.class_colors[class_id], thickness=2)

			det = img[y1:y2, x1:x2, :].copy()
			det_mask = np.ones(det.shape, dtype=np.uint8) * np.uint8(self.class_colors[class_id])
			res = cv2.addWeighted(det, 0.6, det_mask, 0.4, 1.0)
			img[y1:y2, x1:x2] = res

""" ObjectTrackBase class """

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Union

import cv2, math
import numpy as np

FONT_SCALE = 6e-4  # Adjust for larger font size in all images
THICKNESS_SCALE = 2e-3  # Adjust for larger thickness in all images

# ObjectDetectBase
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

    def plot_trajectories(self, img: np.ndarray, observations: list, class_id: int, track_id: int) -> None:
        """
        Draws the trajectories of tracked objects based on historical observations. Each point
        in the trajectory is represented by a circle, with the thickness increasing for more
        recent observations to visualize the path of movement.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories.
        - observations (list): A list of bounding box coordinates representing the historical
        observations of a tracked object. Each observation is in the format (x1, y1, x2, y2).
        - class_id (int): The unique identifier of the tracked object for color consistency in visualization.

        Returns:
        - np.ndarray: The image array with the trajectories drawn on it.
        """
        if (len(observations ) > 1 ):
            for i, box in enumerate(observations):
                cx, ey = int((box[0] + box[2]) / 2), int(box[3]) # int((box[1] + box[3]) / 2)
                thickness = int(np.sqrt(float (i + 1)) * 1.2)
                cv2.circle(
                    img,
                    (cx, ey), 
                    2,
                    color=self.class_colors[class_id],
                    thickness=thickness
                )

            foontSize = min(1, sum(box[2:]) * FONT_SCALE)
            cv2.putText(img, f'ID: {str(track_id)}', (int(box[0]+10*foontSize), int(box[1]+30*foontSize)), 
                        fontFace = cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale = min(1, sum(box[2:]) * FONT_SCALE),
                        thickness = 2,
                        color = self.class_colors[class_id])

    def plot_bbox(self, img: np.ndarray, observations: np.ndarray, class_id: int, track_id: int) -> None:
        if (len(observations ) > 1 ):
            tx1, ty1, tw, th = observations.astype(int)

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
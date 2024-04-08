""" STrack class. """

import numpy as np

from .kalman_filter import KalmanFilter
from .base_track import BaseTrack, TrackState

class LimitedList(list):
	def __init__(self, maxlen):
		super().__init__()
		self._maxlen = maxlen
		self._is_full = False

	def full(self):
		return self._is_full
	
	def append(self, element):
		self.__delitem__(slice(0, len(self) == self._maxlen))
		super(LimitedList, self).append(element)
		if len(self) < self._maxlen:
			self._is_full = False
		else :
			self._is_full = True
			
	def extend(self, elements):
		for element in elements:
			self.append(element)
			
	def clear(self):
		super(LimitedList, self).__init__()
		self._is_full = False

class STrack(BaseTrack):

    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, class_id):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False

        self.crops = []

        self.score = score
        self.tracklet_len = 0

        self.class_id = class_id
        self.class_id_history = {class_id: 1}
        self.trajectories = LimitedList(30) # TODO:
        
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.update_class_id(new_track.class_id)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.trajectories.append(new_track.tlbr) # TODO:
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.update_class_id(new_track.class_id)

    def update_class_id(self, class_id: int) -> None:
        """ Update class id to max count of class id history.

        Args:
            class_id: class id.
        """
        self.class_id_history[class_id] = self.class_id_history.get(class_id, 1) + 1
        self.class_id = max(self.class_id_history, key=self.class_id_history.get)

    def update_crops(self, frame: np.ndarray) -> None:
        """ Update crops.

        Args:
            frame: frame.
        """
        tx1, ty1, tw, th = self._tlwh.astype(int)
        x1 = max(0, tx1)
        y1 = max(0, ty1)
        x2 = min(frame.shape[1], tx1 + tw)
        y2 = min(frame.shape[0], ty1 + th)
        crop = frame[y1:y2, x1:x2, :].copy()
        self.crops.append(crop)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"

    def get_track_message(self):
        track_message = super().get_track_message()
        track_message.update(
            {
                "crops": self.crops,
                "class_id": self.class_id,
            }
        )
        return track_message

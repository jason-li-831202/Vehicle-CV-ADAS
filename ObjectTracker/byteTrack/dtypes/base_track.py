from collections import OrderedDict

import numpy as np

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count
    
    @staticmethod
    def reset_counter():
        BaseTrack._count = 0
        BaseTrack.track_id = 0
        BaseTrack.start_frame = 0
        BaseTrack.frame_id = 0
        BaseTrack.time_since_update = 0

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def get_track_message(self):
        return {
            "track_id": self.track_id,
            "count": self._count,
            "is_activated": self.is_activated,
            "state": self.state,
            "score": self.score,
            "start_frame_number": self.start_frame,
            "curr_frame_number": self.frame_id,
            "time_since_update": self.time_since_update,
            "location": str(self.location),
        }

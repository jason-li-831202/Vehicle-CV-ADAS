from typing import Any, Dict, List

import numpy as np

from . import matching
from .dtypes import BaseTrack, STrack, KalmanFilter, TrackState
from .utils import joint_stracks, sub_stracks, remove_duplicate_stracks
from ..core import ObjectTrackBase


class BYTETracker(ObjectTrackBase):
    """
    Initialize the ByteTrack object.

    Parameters:
        track_thresh (float, optional): Detection confidence threshold
            for track activation. Increasing track_thresh improves accuracy
            and stability but might miss true detections. Decreasing it increases
            completeness but risks introducing noise and instability.
        track_buffer (int, optional): Number of frames to buffer when a track is lost.
            Increasing track_buffer enhances occlusion handling, significantly
            reducing the likelihood of track fragmentation or disappearance caused
            by brief detection gaps.
        match_thresh (float, optional): Threshold for matching tracks with detections.
            Increasing match_thresh improves accuracy but risks fragmentation.
            Decreasing it improves completeness but risks false positives and drift.
        frame_rate (int, optional): The frame rate of the video.
        min_box_area (int, optional): Limit the minimum detection size when drawing trajectories.
    """  
    def __init__(self,
                 track_thresh: float = 0.5,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 frame_rate: int = 30,
                 min_box_area: int = 10,
                 **kwargs: Any):
        super().__init__(**kwargs)

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area

        self.frame_id = 0
        self.det_thresh = track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def _get_tracker_messages(self, status=TrackState.Tracked) -> List[Dict[str, Any]]:
        if (status == TrackState.Lost):
            stracks = self.lost_stracks
        elif (status == TrackState.Removed):
            stracks = self.removed_stracks
        else :
            stracks = self.tracked_stracks
        return [t.get_track_message() for t in stracks]
    
    def update(self, bboxes, scores, class_ids, frame: np.ndarray):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        bboxes = np.array(bboxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets = bboxes[remain_inds]
        dets_second = bboxes[inds_second]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        class_ids_keep = class_ids[remain_inds]
        class_ids_second = class_ids[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, class_id) for
                          (tlbr, s, class_id) in zip(dets, scores_keep, class_ids_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        
        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, class_id) for
                          (tlbr, s, class_id) in zip(dets_second, scores_second, class_ids_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            track.update_crops(frame)
            activated_stracks.append(track)
            
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
 
        return self._get_tracker_messages()

    def reset(self):
        """
        Resets the internal state of the ByteTrack tracker.

        This method clears the tracking data, including tracked, lost,
        and removed tracks, as well as resetting the frame counter. It's
        particularly useful when processing multiple videos sequentially,
        ensuring the tracker starts with a clean state for each new video.
        """
        self.frame_id = 0
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        BaseTrack.reset_counter()

    def DrawTrackedOnFrame(self, frame: np.ndarray, show_box: bool = True, show_traject: bool = True) -> None:
        online_targets = [track for track in self.tracked_stracks if track.is_activated]
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            cid = t.class_id
            trajector = t.trajectories
            if tlwh[2] * tlwh[3] > self.min_box_area:
                if show_box: self.plot_bbox(frame, tlwh, cid, tid)
                if show_traject: self.plot_trajectories(frame, trajector, cid, tid)

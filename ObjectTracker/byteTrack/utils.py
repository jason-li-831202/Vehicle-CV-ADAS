""" Utility functions for BYTETracker. """

import numpy as np
from typing import Tuple, List

from . import matching
from .dtypes import STrack

def joint_stracks(track_list_a: List[STrack], track_list_b: List[STrack]) -> List[STrack]:
    """
    Joins two lists of tracks, ensuring that the resulting list does not
    contain tracks with duplicate track_id values.

    Args:
        track_list_a: First list of tracks (with track_id attribute).
        track_list_b: Second list of tracks (with track_id attribute).

    Returns:
        Combined list of tracks from track_list_a and track_list_b
            without duplicate track_id values.
    """
    seen_track_ids = set()
    result = []

    for track in track_list_a + track_list_b:
        if track.track_id not in seen_track_ids:
            seen_track_ids.add(track.track_id)
            result.append(track)

    return result


def sub_stracks(track_list_a: List, track_list_b: List) -> List[int]:
    """
    Returns a list of tracks from track_list_a after removing any tracks
    that share the same track_id with tracks in track_list_b.

    Args:
        track_list_a: List of tracks (with track_id attribute).
        track_list_b: List of tracks (with track_id attribute) to
            be subtracted from track_list_a.
    Returns:
        List of remaining tracks from track_list_a after subtraction.
    """
    tracks = {track.track_id: track for track in track_list_a}
    track_ids_b = {track.track_id for track in track_list_b}

    for track_id in track_ids_b:
        tracks.pop(track_id, None)

    return list(tracks.values())


def remove_duplicate_stracks(track_list_a: list, track_list_b: list) -> Tuple[List, List]:
    pairwise_distance = matching.iou_distance(track_list_a, track_list_b)
    matching_pairs = np.where(pairwise_distance < 0.15)

    duplicates_a, duplicates_b = set(), set()
    for track_index_a, track_index_b in zip(*matching_pairs):
        time_a = track_list_a[track_index_a].frame_id - track_list_a[track_index_a].start_frame
        time_b = track_list_b[track_index_b].frame_id - track_list_b[track_index_b].start_frame
        if time_a > time_b:
            duplicates_b.add(track_index_b)
        else:
            duplicates_a.add(track_index_a)

    resa = [track for index, track in enumerate(track_list_a) if index not in duplicates_a]
    resb = [track for index, track in enumerate(track_list_b) if index not in duplicates_b]
    return resa, resb
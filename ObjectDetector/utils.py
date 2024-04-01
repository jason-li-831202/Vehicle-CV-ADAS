import numpy as np
from enum import Enum
from numba import jit

class CollisionType(Enum):
    UNKNOWN = "Determined ..."
    NORMAL = "Normal Risk"
    PROMPT = "Prompt Risk"
    WARNING = "Warning Risk"


class ObjectModelType(Enum):
    YOLOV5 = 0
    YOLOV5_LITE = 1
    YOLOV6 = 2
    YOLOV7 = 3
    YOLOV8 = 4
    YOLOV9 = 5

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


@jit(nopython=True)
def fast_nms(dets: np.array, scores: np.array, iou_thr: float):
    """
    It's different from original nms because we have float coordinates on range [0; 1]

    Args:
        dets: numpy array of boxes with shape: (N, 4). Order: x1, y1, x2, y2, score. All variables in range [0; 1]
        scores: numpy array of confidence.
        iou_thr: IoU value for boxes.

    Returns:
        Index of boxes to keep
    """
    
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]

    return keep

@jit(nopython=True)
def fast_soft_nms(dets, scores, iou_thr=0.3, sigma=0.5, score_thr=0.001, method='linear'):
    """Pure python implementation of soft NMS as described in the paper
    `Improving Object Detection With One Line of Code`_.

    Args:
        dets (numpy.array): Detection results with shape `(num, 4)`,
            data in second dimension are [x1, y1, x2, y2] respectively.
        scores (numpy.array): scores for boxes
        iou_thr (float): IOU threshold. Only work when method is `linear`
            or 'greedy'.
        sigma (float): Gaussian function parameter. Only work when method
            is `gaussian`.
        score_thr (float): Boxes that score less than the.
        method (str): Rescore method. Only can be `linear`, `gaussian`
            or 'greedy'.
    Returns:
        index of boxes to keep
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.arange(N).reshape(N, 1) # indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes), axis=1) # dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1, x1, y2, x2]
    y1 = dets[:, 1]
    x1 = dets[:, 0]
    y2 = dets[:, 3]
    x2 = dets[:, 2]
    areas = (x2 - x1) * (y2 - y1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :] # tBD = dets[i, :].copy()
        tscore = scores[i]
        tarea = areas[i] # tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N - 1:
            maxscore = np.max(scores[pos:])
            maxpos = np.argmax(scores[pos:])
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(x1[i], x1[pos:])
        yy1 = np.maximum(y1[i], y1[pos:])
        xx2 = np.minimum(x2[i], x2[pos:])
        yy2 = np.minimum(y2[i], y2[pos:])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        if method == "linear":
            weight = np.ones(ovr.shape)
            weight[ovr > iou_thr] = weight[ovr > iou_thr] - ovr[ovr > iou_thr]
        elif method == "gaussian":
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > iou_thr] = 0

        scores[pos:] *= weight

    # select the boxes and keep the corresponding indexes
    keep = np.where(scores >= score_thr)[0]
    return keep
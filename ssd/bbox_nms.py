import numpy as np
from utils import nms


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1] - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.reshape(multi_scores.shape[0], -1, 4)[:, 1:]
    else:
        bboxes = multi_bboxes[:, None]
    scores = multi_scores[:, 1:]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[1]

    if bboxes.size == 0:
        bboxes = np.zeros((0, 5), dtype=multi_bboxes.dtype)
        labels = np.zeros((0, ), dtype=np.long)
        return bboxes, labels

    # Modified from https://github.com/pytorch/vision/blob
    # /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = bboxes.max()
    offsets = labels.astype(bboxes.dtype) * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]

    scores = scores.astype(np.float64)
    bboxes_for_nms = bboxes_for_nms.astype(np.float64)

    nms_cfg_ = nms_cfg.copy()

    keep = nms(bboxes_for_nms, scores, nms_cfg_.get('iou_thr', None))
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if len(keep) > max_num:
        inds = scores.argsort()[::-1]
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]

    return np.concatenate([bboxes, scores[:, None]], 1), labels

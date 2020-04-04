import numpy as np
from utils import sigmoid, softmax, addcmul, topk


def delta2bbox(rois,
               deltas,
               means=None,
               stds=None,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    if stds is None:
        stds = [1, 1, 1, 1]
    if means is None:
        means = [0, 0, 0, 0]
    means = np.tile(np.array(means, dtype=deltas.dtype), (1, deltas.shape[1] // 4))
    stds = np.tile(np.array(stds, dtype=deltas.dtype), (1, deltas.shape[1] // 4))
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = np.clip(dw, -max_ratio, max_ratio)
    dh = np.clip(dh, -max_ratio, max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5)[:, np.newaxis]
    py = ((rois[:, 1] + rois[:, 3]) * 0.5)[:, np.newaxis]
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0] + 1.0)[:, np.newaxis]
    ph = (rois[:, 3] - rois[:, 1] + 1.0)[:, np.newaxis]
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * np.exp(dw)
    gh = ph * np.exp(dh)
    # Use network energy to shift the center of each roi
    gx = addcmul(px, pw, dx)  # gx = px + pw * dx
    gy = addcmul(py, ph, dy)  # gy = py + ph * dy
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1] - 1)
        y1 = np.clip(y1, 0, max_shape[0] - 1)
        x2 = np.clip(x2, 0, max_shape[1] - 1)
        y2 = np.clip(y2, 0, max_shape[0] - 1)
    bboxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(deltas.shape)
    return bboxes


def get_bboxes_single(cls_score_list,
                      bbox_pred_list,
                      mlvl_anchors,
                      img_shape,
                      scale_factor,
                      cfg,
                      rescale=False):

    assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
    mlvl_bboxes = []
    mlvl_scores = []
    # ############# add #############
    use_sigmoid_cls = False
    cls_out_channels = 2
    target_means = (.0, .0, .0, .0)
    target_stds = (0.1, 0.1, 0.2, 0.2)
    # ############# add #############
    for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                             bbox_pred_list, mlvl_anchors):
        assert cls_score.shape[-2:] == bbox_pred.shape[-2:]
        cls_score = np.transpose(cls_score, (1, 2, 0)).reshape(-1, cls_out_channels)
        if use_sigmoid_cls:
            scores = sigmoid(cls_score)
        else:
            scores = softmax(cls_score)
        bbox_pred = np.transpose(bbox_pred, (1, 2, 0)).reshape(-1, 4)
        nms_pre = cfg.get('nms_pre', -1)
        if 0 < nms_pre < scores.shape[0]:
            # Get maximum scores for foreground classes.
            if use_sigmoid_cls:
                max_scores = scores.max(axis=1)
            else:
                max_scores, _ = scores[:, 1:].max(axis=1)
            topk_inds = topk(max_scores, nms_pre, axis=1)
            anchors = anchors[topk_inds, :]
            bbox_pred = bbox_pred[topk_inds, :]
            scores = scores[topk_inds, :]

        bboxes = delta2bbox(anchors, bbox_pred, target_means,
                            target_stds, img_shape)

        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)
    mlvl_bboxes = np.concatenate(mlvl_bboxes)
    mlvl_scores = np.concatenate(mlvl_scores)
    if use_sigmoid_cls:
        # Add a dummy background class to the front when using sigmoid
        padding = np.zeros((mlvl_scores.shape[0], 1), dtype=mlvl_scores.dtype)
        mlvl_scores = np.concatenate([padding, mlvl_scores], axis=1)
    from bbox_nms import multiclass_nms
    det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                            cfg.score_thr, cfg.nms,
                                            cfg.max_per_img)
    if rescale:
        det_bboxes[:, 0] *= scale_factor[0]
        det_bboxes[:, 1] *= scale_factor[1]
        det_bboxes[:, 2] *= scale_factor[0]
        det_bboxes[:, 3] *= scale_factor[1]

    return det_bboxes, det_labels

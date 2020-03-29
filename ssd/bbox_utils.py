import numpy as np
import torch
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
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes


def get_bboxes_single(cls_score_list,
                      bbox_pred_list,
                      mlvl_anchors,
                      img_shape,
                      scale_factor,
                      cfg,
                      rescale=False):
    """
        Transform outputs for a single batch item into labeled boxes.
        """
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
        cls_score = np.transpose(cls_score, (1, 2, 0)).reshape((-1, cls_out_channels))
        if use_sigmoid_cls:
            scores = sigmoid(cls_score)
        else:
            scores = softmax(cls_score)
        bbox_pred = np.transpose(bbox_pred, (1, 2, 0)).reshape((-1, 4))
        nms_pre = cfg.get('nms_pre', -1)
        if 0 < nms_pre < scores.shape[0]:
            # Get maximum scores for foreground classes.
            if use_sigmoid_cls:
                max_scores = np.max(scores, axis=1)
            else:
                max_scores, _ = np.max(scores[:, 1:], axis=1)
            topk_inds = topk(max_scores, nms_pre)
            anchors = anchors[topk_inds, :]
            bbox_pred = bbox_pred[topk_inds, :]
            scores = scores[topk_inds, :]
        bboxes = delta2bbox(anchors, bbox_pred, target_means,
                            target_stds, img_shape)
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)
    mlvl_bboxes = torch.cat(mlvl_bboxes)
    if rescale:
        mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
    mlvl_scores = torch.cat(mlvl_scores)
    if use_sigmoid_cls:
        # Add a dummy background class to the front when using sigmoid
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
    from bbox_nms import multiclass_nms
    det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                            cfg.score_thr, cfg.nms,
                                            cfg.max_per_img)
    return det_bboxes, det_labels

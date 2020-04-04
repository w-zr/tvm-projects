import numpy as np


def sigmoid(x: np.ndarray):
    s = 1 / (1 + np.exp(-x))
    return s


def softmax(x: np.ndarray):
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x


def addcmul(px: np.ndarray, pw: np.ndarray, dx: np.ndarray):
    return px + pw * dx


def topk(x: np.ndarray, k: int, axis=1):
    part = np.argpartition(x, k, axis=axis)
    if axis == 0:
        row_index = np.arange(x.shape[1])
        sort_K = np.argsort(x[part[k + 1:, :], row_index], axis=axis)
        return np.fliplr(part[k + 1:, :][sort_K, row_index])
    else:
        column_index = np.arange(x.shape[1 - axis])[:, None]
        sort_K = np.argsort(x[column_index, part[:, k + 1:]], axis=axis)
        return np.fliplr(part[:, k + 1:][column_index, sort_K])


def nms(dets, scores, prob_threshold):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    score_index = scores.argsort()[::-1]
    keep = []

    while score_index.size > 0:
        i = score_index[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[score_index[1:]])
        yy1 = np.maximum(y1[i], y1[score_index[1:]])
        xx2 = np.minimum(x2[i], x2[score_index[1:]])
        yy2 = np.minimum(y2[i], y2[score_index[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        union = w * h
        iou = union / (areas[i] + areas[score_index[1:]] - union)

        ids = np.where(iou <= prob_threshold)[0]
        score_index = score_index[ids + 1]

    return keep

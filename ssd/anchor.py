import numpy as np
import torch
from anchor_generator import AnchorGenerator

# generate anchor
def gen_anchors():
    basesize_ratio_range = (0.2, 0.9)
    in_channels = (32, 96, 320, 512, 256, 256)
    input_size = 300
    anchor_strides = (8, 16, 32, 64, 100, 300)
    anchor_ratios = ([2], [2, 3], [2, 3], [2, 3], [2], [2])

    min_ratio, max_ratio = basesize_ratio_range
    min_ratio = int(min_ratio * 100)
    max_ratio = int(max_ratio * 100)
    step = int(np.floor(max_ratio - min_ratio) / (len(in_channels) - 2))
    min_sizes = []
    max_sizes = []

    for r in range(int(min_ratio), int(max_ratio) + 1, step):
        min_sizes.append(int(input_size * r / 100))
        max_sizes.append(int(input_size * (r + step) / 100))

    min_sizes.insert(0, int(input_size * 10 / 100))
    max_sizes.insert(0, int(input_size * 20 / 100))

    anchor_generators = []
    for k in range(len(anchor_strides)):
        base_size = min_sizes[k]
        stride = anchor_strides[k]
        ctr = ((stride - 1) / 2., (stride - 1) / 2.)
        scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
        ratios = [1.]
        for r in anchor_ratios[k]:
            ratios += [1 / r, r]  # 4 or 6 ratio
        anchor_generator = AnchorGenerator(
            base_size, scales, ratios, scale_major=False, ctr=ctr)
        indices = list(range(len(ratios)))
        indices.insert(1, len(indices))
        anchor_generator.base_anchors = torch.index_select(
            anchor_generator.base_anchors, 0, torch.LongTensor(indices))
        anchor_generators.append(anchor_generator)

    feature_size = ((38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1))
    mlvl_anchors = []
    for feat_size, stride, anchor_generator in zip(feature_size, anchor_strides, anchor_generators):
        anchor = anchor_generator.grid_anchors(feat_size, stride, device='cpu')
        mlvl_anchors.append(anchor)

    return mlvl_anchors

import time

import cv2
import numpy as np
import onnx
import tvm
import tvm.contrib.graph_runtime as runtime
import tvm.relay as relay

# load image
image_path = '9331584514251_.pic_hd.jpg'
image = cv2.imread(image_path)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pre_start = time.process_time()

resize_shape = (300, 300)
img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
mean = np.array([123.675, 116.28, 103.53]).reshape(1, -1)
std = np.array([1., 1., 1.]).reshape(1, -1)
img = img.astype(np.float32)
img = cv2.subtract(img, mean)
img = cv2.multiply(img, std)
img = img.transpose(2, 0, 1)
pre_end = time.process_time()

# load onnx model and build tvm runtime
target = 'llvm'
ctx = tvm.context(target)
dtype = 'float32'
mssd = onnx.load('mssd.onnx')
input_blob = mssd.graph.input[0]
input_shape = tuple(map(lambda x: getattr(x, 'dim_value'), input_blob.type.tensor_type.shape.dim))
shape_dict = {input_blob.name: input_shape}
mod, params = relay.frontend.from_onnx(mssd, shape_dict)
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod, target, params=params)

######## export lib ########
# path = 'model/'
# path_lib = path+"deploy_lib.tar"
# path_graph = path+"deploy_graph.json"
# path_params = path+"deploy_param.params"
# lib.export_library(path_lib)
# with open(path_graph, "w") as fo:
#     fo.write(graph)
# with open(path_params, "wb") as fo:
#     fo.write(relay.save_param_dict(params))

######## load lib ########
# load the module back.
# graph = open(path_graph).read()
# lib = tvm.runtime.load_module(path_lib)
# params = bytearray(open(path_params, "rb").read())

module = runtime.create(graph, lib, ctx)

# run
# module.load_params(bytearray(params))
module.set_input(**params)
module.set_input('input.1', tvm.nd.array(img))
time_list = []
post_time_list = []

start = time.process_time()
module.run()
end = time.process_time()
time_list.append(end - start)

# get output
post_start = time.process_time()
cls_score_list = [module.get_output(i).asnumpy()[0] for i in range(6)]
bbox_pred_list = [module.get_output(i + 6).asnumpy()[0] for i in range(6)]

# generate anchor
from anchor import gen_anchors

mlvl_anchors = gen_anchors()
# recover bbox
from bbox_utils import get_bboxes_single

img_shape = image.shape
scale_factor = [img_shape[1] / resize_shape[1], img_shape[0] / resize_shape[0]] # x_scale, y_scale

cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200
)
from easydict import EasyDict

cfg = EasyDict(cfg)
proposals = get_bboxes_single(cls_score_list, bbox_pred_list, mlvl_anchors, resize_shape, scale_factor, cfg,
                              rescale=True)
post_end = time.process_time()
post_time_list.append(post_end - post_start)

from heapq import nlargest

print(sum(nlargest(20, time_list)) / 20)
print(sum(nlargest(20, post_time_list)) / 20)


from vis_bbox import imshow_det_bboxes

bboxes = proposals[0]
labels = proposals[1]
imshow_det_bboxes(image, bboxes, labels, score_thr=0.9, out_file='out.png')

print("pre: {}".format(pre_end - pre_start))
print("run: {}".format(end - start))
print("post: {}".format(post_end - post_start))

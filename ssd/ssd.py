import time
import cv2
import onnx
import tvm
import tvm.contrib.graph_runtime as runtime
import tvm.relay as relay
from PIL import Image
from torchvision import transforms

# load image
image_path = '9331584514251_.pic_hd.jpg'

pre_start = time.process_time()
img = Image.open(image_path)

resize_shape = (300, 300)
transform = transforms.Compose([
    transforms.Resize(resize_shape, interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[123.675 / 255, 116.28 / 255, 103.53 / 255], std=[1, 1, 1]),
])

img = transform(img) * 255
pre_end = time.process_time()

# load onnx model and build tvm runtime
target = 'cuda'
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
module.set_input('input.1', tvm.nd.array(img.numpy().astype('float32')))
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

scale_factor = 1.

cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200
)
from easydict import EasyDict

cfg = EasyDict(cfg)
proposals = get_bboxes_single(cls_score_list, bbox_pred_list, mlvl_anchors, resize_shape, scale_factor, cfg,
                              rescale=False)
post_end = time.process_time()
post_time_list.append(post_end - post_start)

from heapq import nlargest

print(sum(nlargest(20, time_list)) / 20)
print(sum(nlargest(20, post_time_list)) / 20)

image = cv2.imread(image_path)

img_shape = image.shape
y_scale = img_shape[0] / resize_shape[0]
x_scale = img_shape[1] / resize_shape[1]

bboxes = proposals[0]
labels = proposals[1]
bboxes[:, 0] = bboxes[:, 0] * x_scale
bboxes[:, 1] = bboxes[:, 1] * y_scale
bboxes[:, 2] = bboxes[:, 2] * x_scale
bboxes[:, 3] = bboxes[:, 3] * y_scale

from vis_bbox import imshow_det_bboxes

imshow_det_bboxes(image, bboxes, labels, score_thr=0.9, out_file='out.png')

print("pre: {}".format(pre_end - pre_start))
print("run: {}".format(end - start))
print("post: {}".format(post_end - post_start))

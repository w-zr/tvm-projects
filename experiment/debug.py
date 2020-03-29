import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib.debugger import debug_runtime as graph_runtime


def get_val_data():
    root = '/home/ziran/文档/ILSVRC2012_img_val/'
    meta = '/home/ziran/文档/meta/val.txt'
    from torchvision.transforms import transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    from PIL import Image
    with open(meta, 'r') as f:
        content = f.readlines()
        for i in range(0, len(content), batch_size):
            batch = content[i:i+batch_size]
            imgs = []
            categories = []
            for item in batch:
                img_name, category = item.strip().split()
                category = int(category)

                with Image.open(root + img_name) as img:
                    img = img.convert('RGB')
                    img = preprocess(img)
                imgs.append(img.detach().numpy())
                categories.append(category)
                yield {'data': np.stack(imgs), 'label': categories}


calibration_samples = 100


def calibrate_dataset():
    generator = get_val_data()
    for i, batch in enumerate(generator):
        if i * batch_size >= calibration_samples:
            break
        yield {'data': batch['data']}


def quantize(mod, params):
    with relay.quantize.qconfig(calibrate_mode='kl_divergence', weight_scale='max'):
        mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset())
    return mod


batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

mod, params = relay.testing.resnet.get_workload(
    num_layers=18, batch_size=batch_size, image_shape=image_shape)

# set show_meta_data=True if you want to show meta data
print(mod.astext(show_meta_data=False))

opt_level = 3
target = tvm.target.cuda()
with relay.build_config(opt_level=opt_level):
    graph, lib, params = relay.build_module.build(
        mod, target, params=params)


# create random input
ctx = tvm.gpu()
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
# create module
module = graph_runtime.create(graph, lib, ctx)
# set input and parameters
module.set_input("data", data)
module.set_input(**params)
# run
module.run()
# get output
out = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()

# Print first 10 elements of output
print(out.flatten()[0:10])
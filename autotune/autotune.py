import os

import cv2
import numpy as np
import onnx
import tvm
import tvm.contrib.graph_runtime as runtime
import tvm.relay.testing
from tvm import autotvm
from tvm import relay
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir

#### DEVICE CONFIG ####

# Replace "aarch64-linux-gnu" with the correct target of your board.
# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
target = tvm.target.create('llvm -target=aarch64-linux-gnu -mattr=+neon')

# Also replace this with the device key in your tracker
device_key = 'arm'

# Set this to True if you use android phone
use_android = False

#### TUNING OPTION ####
network = 'mssd_quantize'
log_file = "%s.%s.log" % (device_key, network)
dtype = 'float32'


tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb_knob',
    'n_trial': 1500,
    'early_stopping': 800,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(
            build_func='ndk' if use_android else 'default'),
        runner=autotvm.RPCRunner(
            device_key, host='0.0.0.0', port=9000,
            number=5,
            timeout=10,
        ),
        #runner=autotvm.LocalRunner()
    ),
}


def get_val_data(image_path):
    filenames = os.listdir(image_path)
    images = []
    imgs = []
    for filename in filenames:
        image = cv2.imread(image_path + filename)
        images.append(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resize_shape = (300, 300)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
        mean = np.array([123.675, 116.28, 103.53]).reshape(1, -1)
        std = np.array([1., 1., 1.]).reshape(1, -1)
        img = img.astype(np.float32)
        img = cv2.subtract(img, mean)
        img = cv2.multiply(img, std)
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, :]
        imgs.append(img)
    return imgs, images


def calibrate_dataset():
    val_data, _ = get_val_data('/home/ziran/repositories/mobilenet/data/widerface/val/images/0--Parade/')
    for i in range(10):
        yield {'input.1': val_data[i]}


# Quantize the Model
def quantize(mod, params, data_aware):
    if data_aware:
        with relay.quantize.qconfig(calibrate_mode='kl_divergence', weight_scale='max'):
            mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset())
    else:
        with relay.quantize.qconfig(calibrate_mode='global_scale', global_scale=8.0):
            mod = relay.quantize.quantize(mod, params)
    return mod


def get_network():
    """Get the symbol definition and random weight of a network"""
    mssd = onnx.load("/home/ziran/repositories/tvm-projects/ssd/mssd.onnx")
    input_blob = mssd.graph.input[0]
    input_shape = tuple(map(lambda x: getattr(x, 'dim_value'), input_blob.type.tensor_type.shape.dim))
    shape_dict = {input_blob.name: input_shape}
    mod, params = relay.frontend.from_onnx(mssd, shape_dict)

    mod = quantize(mod, params, data_aware=True)

    return mod, params, input_shape


# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank', num_threads=12)
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob', num_threads=12)
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape = get_network()
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

    # run tuning tasks
    print("Tuning...")
    #tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

    # export library
    tmp = tempdir()
    if use_android:
        from tvm.contrib import ndk
        filename = "net.so"
        lib.export_library(tmp.relpath(filename), ndk.create_shared)
    else:
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

    # upload module to device
    print("Upload...")
    remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9000,
                                            timeout=10000)
    remote.upload(tmp.relpath(filename))
    rlib = remote.load_module(filename)

    ctx = remote.context(str(target), 0)
    module = runtime.create(graph, rlib, ctx)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input('input.1', data_tvm)
    module.set_input(**params)

    # evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=100)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
          (np.mean(prof_res), np.std(prof_res)))

# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.


tune_and_evaluate(tuning_option)

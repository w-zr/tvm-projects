import os

import numpy as np
import onnx
import tvm
import tvm.relay as relay
from tvm.contrib.debugger import debug_runtime

from tvm import te
from tvm import autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime

batch_size = 1

target = 'cuda'
ctx = tvm.context(target)

log_file = 'resnet-18.log'
dtype = 'float32'

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 2000,
    'early_stopping': 600,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        # runner=autotvm.RPCRunner(
        #    '1070',
        #    'localhost',
        #    9191,
        #    number=20, repeat=3, timeout=4, min_repeat_ms=150)
    ),
}


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
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
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


def tune_and_evaluate(tuning_opt, mod, params, input_shape):
    # extract workloads from relay program
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(mod["main"], target=tvm.target.cuda(),
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # export library
        tmp = tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

# tune_and_evaluate(tuning_option)


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
            batch = content[i:i + batch_size]
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


def bind_params(func, params):
    """Bind the params to the expression.
    """
    name_dict = {}
    for arg in func.params:
        name = arg.name_hint
        if name in name_dict:
            name_dict[name] = None
        else:
            name_dict[name] = arg
    bind_dict = {}
    for k, v in params.items():
        if k not in name_dict:
            continue
        arg = name_dict[k]
        if arg is None:
            raise ValueError("Multiple args in the function have name %s" % k)
        bind_dict[arg] = relay.expr.const(v)
    return relay.expr.bind(func, bind_dict)


def run_inference(mod):
    top1_correct = 0
    top5_correct = 0
    total = 0
    executor = relay.create_executor('vm', mod, ctx, target)
    val_data = get_val_data()
    import time
    start = time.process_time()
    for i, batch in enumerate(val_data):
        data, categories = batch['data'], batch['label']
        prediction = executor.evaluate()(data).asnumpy()
        top1_correct += (prediction.argmax(1) == categories).sum()
        top5_correct += sum(map(lambda x: x[0] in x[1], zip(categories, prediction.argsort()[:, -5:])))
        total += len(data)
        print(prediction)
        print('Top1 Acc: {}, {}/{}'.format(float(top1_correct) / total, top1_correct, total))
        print('Top5 Acc: {}, {}/{}'.format(float(top5_correct) / total, top5_correct, total))
    end = time.process_time()
    print('Time: {}'.format(end - start))
    print('Top1 Acc: {}, {}/{}'.format(float(top1_correct) / total, top1_correct, total))
    print('Top5 Acc: {}, {}/{}'.format(float(top5_correct) / total, top5_correct, total))


def main():
    resnetv1 = onnx.load('models/resnet18v1.onnx')
    input_blob = resnetv1.graph.input[0]
    input_shape = tuple(map(lambda x: getattr(x, 'dim_value'), input_blob.type.tensor_type.shape.dim))
    shape_dict = {input_blob.name: input_shape}
    mod_resnetv1, params_resnetv1 = relay.frontend.from_onnx(resnetv1, shape_dict)

    # resnetv2 = onnx.load('models/resnet18v2.onnx')
    # input_blob = resnetv2.graph.input[0]
    # input_shape = tuple(map(lambda x: getattr(x, 'dim_value'), input_blob.type.tensor_type.shape.dim))
    # shape_dict = {input_blob.name: input_shape}
    # mod_resnetv2, params_resnetv2 = relay.frontend.from_onnx(resnetv2, shape_dict)

    mod_q_resnetv1 = quantize(mod_resnetv1, params_resnetv1)
    # mod_q_resnetv2 = quantize(mod_resnetv2, params_resnetv2)

    # mod_resnetv1['main'] = bind_params(mod_resnetv1['main'], params_resnetv1)

    # f = open('graphs/resnetv1_q.log.new', 'w+')
    # f.write(str(mod_q_resnetv1))
    # f.close()

    # f = open('graphs/resnetv2_q.log', 'w+')
    # f.write(str(mod_q_resnetv2))
    # f.close()

    # run_inference(mod_resnetv1)
    # run_inference(mod_q_resnetv1)
    # run_inference(mod_q_resnetv2)

    with autotvm.apply_history_best(log_file):
        #print("Compile...")
        #with relay.build_config(opt_level=3):
        #graph, lib, params = relay.build_module.build(
        #mod_q_resnetv1, target=target, params=params_resnetv1)

        #export library
        #tmp = tempdir()
        #filename = "net.tar"
        #lib.export_library(tmp.relpath(filename))

        # load parameters
        #ctx = tvm.context(str(target), 0)
        #module = runtime.create(graph, lib, ctx)
        #module.set_input(**params)

        #val_data = get_val_data()
        #top1_correct = 0
        #top5_correct = 0
        #total = 0
        #import time
        #start = time.process_time()
        #for i, batch in enumerate(val_data):
        #    data, categories = batch['data'], batch['label']
        #    module.set_input('data', data)
        #    module.run()
        #    prediction = module.get_output(0).asnumpy()
        #    top1_correct += (prediction.argmax(1) == categories).sum()
        #    top5_correct += sum(map(lambda x: x[0] in x[1], zip(categories, prediction.argsort()[:, -5:])))
        #    total += len(data)
        #    print(prediction)
        #    print('Top1 Acc: {}, {}/{}'.format(float(top1_correct) / total, top1_correct, total))
        #    print('Top5 Acc: {}, {}/{}'.format(float(top5_correct) / total, top5_correct, total))
        #end = time.process_time()
        #print('Time: {}'.format(end - start))
        #print('Top1 Acc: {}, {}/{}'.format(float(top1_correct) / total, top1_correct, total))
        #print('Top5 Acc: {}, {}/{}'.format(float(top5_correct) / total, top5_correct, total))

        # evaluate
        #print("Evaluate inference time cost...")
        #ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        #prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        #print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
        #      (np.mean(prof_res), np.std(prof_res)))

        graph, mod, params = relay.build_module.build(mod_q_resnetv1['main'], target=target, params=params_resnetv1)

        val_data = get_val_data()
        for i, batch in enumerate(val_data):
            if i > 0:
                break
            data, categories = batch['data'], batch['label']
            m = debug_runtime.create(graph, mod, ctx, dump_root='tvmdbg')
            m.set_input('data', tvm.nd.array(data.astype('float32')))
            m.run()
            tvm_out = m.get_output(0, tvm.nd.empty(tuple([1, 1000]), 'float32')).asnumpy()
    ## tune_and_evaluate(tuning_option, mod_q_resnetv1, params_resnetv1, input_shape)


if __name__ == '__main__':
    main()

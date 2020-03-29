import onnx.optimizer

model = onnx.load('models/resnet18v2.onnx')

opt_model = onnx.optimizer.optimize(model, passes=['fuse_bn_into_conv'])

onnx.save(opt_model, 'models/merged_resnet18v2.onnx')

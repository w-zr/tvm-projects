import os
import tarfile

from flask import Flask, request, flash, redirect, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

import onnx
import tvm.relay as relay
from tvm.contrib.util import tempdir

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = '123456'


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/model', methods=['POST', 'OPTIONS'])
def upload_model():
    if request.method == 'OPTIONS':
        return ''
    # check if the post request has the file part
    if 'model' not in request.files:
        flash('No model part')
        return redirect(request.url)
    model = request.files['model']
    # if user does not select file, browser also
    # submit an empty part without filename
    if model.filename == '':
        flash('No selected file')
        return redirect(request.url)

    filename = secure_filename(model.filename)
    if filename.endswith('.onnx'):
        onnx_model = onnx.load(model)
        model_graph = onnx_model.graph
        target = request.form.get('opts')
        input_blob = model_graph.input[0]
        input_shape = tuple(map(lambda x: getattr(x, 'dim_value'), input_blob.type.tensor_type.shape.dim))
        input_name = input_blob.name
        shape_dict = {input_name: input_shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, target=target, params=params)

        tmp = tempdir()
        lib.export_library(tmp.relpath('net.tar'))
        with open(tmp.relpath('deploy_graph.json'), 'w') as fo:
            fo.write(graph)
        with open(tmp.relpath('deploy_param.params'), 'wb') as fo:
            fo.write(relay.save_param_dict(params))

        if os.path.isfile('./download/deploy.tar.gz'):
            os.remove('./download/deploy.tar.gz')

        tar = tarfile.open('./download/deploy.tar.gz', 'w:gz')
        tar.add(tmp.relpath('net.tar'), arcname='lib.tar')
        tar.add(tmp.relpath('deploy_graph.json'), arcname='deploy_graph.json')
        tar.add(tmp.relpath('deploy_param.params'), arcname='deploy_param.params')
        tar.close()
        tmp.remove()
        return 'convert successful'
    return 'not support'


@app.route('/download', methods=['GET'])
def download_model():
    return send_file('./download/deploy.tar.gz')

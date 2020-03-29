import React, { useState } from 'react';
import {UploadOutlined} from '@ant-design/icons'
import { Upload, message, Button, Modal, Spin, Layout, Input} from 'antd'
import './App.css';

const {Header, Content} = Layout


function App() {

	const [uploading, setUploading] = useState(false);
	const [opts, setOpts] = useState('');

	const props = {
		name: 'model',
		action: 'http://127.0.0.1:5000/model',
		accept: '.onnx',
		data: {opts},
	
		onChange(info) {
			if (info.file.status !== 'uploading') {
				console.log(info.file, info.fileList);
			}
			if (info.file.status === 'uploading') {
				setUploading(true); 
			}
			else if (info.file.status === 'done') {
				setUploading(false);
				Modal.info({
					title: '模型转换成功，点击下载',
					onOk() {},
				});
			} else if (info.file.status === 'error') {
				setUploading(false);
				message.error(`${info.file.name} 上传失败`);
			}
		},

		onDownload() {
			fetch('http://127.0.0.1:5000/download')
			.then(response => response.blob())
			.then(blob => {
				const url = window.URL.createObjectURL(blob);
				const a = document.createElement('a');
				a.style.display = 'none';
				a.href = url;
				a.download = 'deploy.tar.gz';
				document.body.appendChild(a);
				a.click();
				window.URL.revokeObjectURL(url);
			})
			.catch(() => console.log('error'))
		}
	};

	return(
		<div className="App">
			<Spin spinning={uploading} tip='上传并转换中'>
			<Layout>
				<Header className="App-header"><h1 style={{color: 'white'}}>模型转换</h1></Header>
			<Content className="App-body">
				<div className="Upload">
					<Input placeholder='输入编译属性' onChange={e => {setOpts(e.target.value); console.log(opts);}}/>
					<Upload {... props}>
						<Button onClick={() => {console.log(opts)}}>
							<UploadOutlined/> 点击上传
						</Button>
					</Upload>
				</div>
			</Content>
			</Layout>
			</Spin>
		</div>
	)
}

export default App;

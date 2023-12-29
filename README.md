## 示例模型链接
<https://pan.baidu.com/s/1sJNz9_zTqAXJhvBeN3tFkQ?pwd=k49b> 提取码：k49b 
提供的模型需要使用tensorrt-23.04以及tritonserver-23.04容器运行，并将其下载至/workspace/code目录下
## docker镜像搭建
1) 镜像拉取
```sh
docker pull nvcr.io/nvidia/tritonserver:23.04-py3
```
在拉取triton镜像的时候，需要检查镜像版本和NVIDIA驱动是否匹配，镜像的版本号和驱动版本存在对应关系,参考文档：<https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html>

2) 启动镜像
```sh
docker run -dt --gpus=all -p 1237:22 --shm-size 32g --name triton -v /home/xxiao/code:/workspace/code nvcr.io/nvidia/tritonserver:23.04-py3
```
shm-size根据自己的设备实际大小进行调整

3) 进入镜像
```
docker exec -it triton /bin/bash
cd /workspace/code
git clone https://github.com/yuxiaoranyu/stable_diffusion_trt_triton.git
cd stable_diffusion_trt_triton
pip install -r requirements.txt
```
安装tensorrt
```
cp -r /workspace/code/stable_diffusion_inpainting/tensorrt /usr/local/lib/python3.8/dist-packages/tensorrt
```
该tensorrt文件夹是从tensorrt-23.04镜像中拷贝出来

## 将模型转换为tensorrt
```
python3 inpaint2trt.py
```
等待转换完成，模型较大，所需时间较长，耐心等候
转换完成后将在/workspace/code/stable_diffusion_inpainting目录下生成onnx和engine文件夹
## 使用原生tensorrt执行推理
```
python3 inpaint_trt_infer.py
```
## 使用tritonserver执行推理
```
tritonserver --model-repository=inpaint_model
```
等待triton启动后
```
python3 inpaint_triton_infer.py
```
在unet的推理过程中，需要循环执行推理，使用triton client访问server端会非常耗时，故unet部分仍旧使用tensorrt的推理。在triton的代码仓库中提供了python_backend的方法，该方法已写到代码中，但是需要换一种triton的方法才能使用，该方法暂时先卖个关子，后续将会更新上来。
使用triton推理，需要建一个类似于inpaint_model的目录
```
./inpaint_model/
|-- text_encoder
|   |-- 1
|   |   `-- model.plan
|   `-- config.pbtxt
|-- vae_decoder
|   |-- 1
|   |   `-- model.plan
|   `-- config.pbtxt
`-- vae_encoder
    |-- 1
    |   `-- model.plan
    `-- config.pbtxt
``` 
需要将stable_diffusion_inpainting/engine下的tensorrt模型拷贝至inpaint_model相应目录下，并配置相应的config.pbtxt文件（已提供）
后续将会陆续更新controlnet，ip_adapter的inpaint模型代码以及文生图tensorrt推理
## 相关代码参考
<https://github.com/huggingface/diffusers>  
<https://github.com/triton-inference-server/server>  
<https://github.com/triton-inference-server/python_backend>

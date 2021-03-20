# Example of Pytorch -> ONNX -> TensorRT
Pytorch로 구현된 모델을 ONNX로 바꾸고 ONNX에서 TensorRT로 엔진을 다시 빌드하여 inference time을 최적화 시키는것이 목적

Pytorch 공식 홈페이지에서 제공하는 ONNX Tutorial을 base로 진행


## Pytorch -> ONNX
환경 세팅은 공식 tutorial 참조

Pytorch 공식 tutorial : https://tutorials.pytorch.kr/advanced/super_resolution_with_onnxruntime.html

코드는 onnx.ipynb, pytorch2onnx.py 참조

### example output
```python
# pytorch model
tensor([[[[0.7651, 0.8741, 0.9048,  ..., 0.8559, 0.7837, 0.7129],
          [0.8241, 1.0140, 1.1038,  ..., 1.1016, 1.0106, 0.7954],
          [0.8494, 1.0746, 1.1395,  ..., 1.1076, 1.0816, 0.8434],
          ...,
          [0.9251, 1.0277, 1.0539,  ..., 1.0218, 1.0811, 0.9037],
          [0.8515, 0.9133, 0.9483,  ..., 0.9819, 1.0057, 0.8684],
          [0.7569, 0.7720, 0.7775,  ..., 0.7539, 0.8017, 0.7727]]]],
       grad_fn=<UnsafeViewBackward>)
```
```python
# onnx
[array([[[[0.76511073, 0.8740847 , 0.90481126, ..., 0.85588545,
           0.7836736 , 0.7129    ],
          [0.82408434, 1.0140268 , 1.1037575 , ..., 1.1016412 ,
           1.0106119 , 0.79541755],
          [0.84935516, 1.0745931 , 1.1394703 , ..., 1.1076125 ,
           1.0815643 , 0.843422  ],
          ...,
          [0.92509943, 1.0276617 , 1.0539078 , ..., 1.0218285 ,
           1.081129  , 0.9036602 ],
          [0.851483  , 0.91334385, 0.9482872 , ..., 0.9819409 ,
           1.005691  , 0.86839116],
          [0.7569455 , 0.772007  , 0.77752113, ..., 0.75390404,
           0.8017341 , 0.7727227 ]]]], dtype=float32)]
```

## ONNX -> TensorRT
환경 세팅은
1. Nvidia docker container 사용
https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt
```
# container 안에서

$ trtexec --onnx=super_resolution.onnx --explicitBatch --saveEngine=sample.trt --workspace=1024 --fp16
```
2. Jetson 사용시 (jetson jetpack이라는 가정 하에) models의 build_trt.sh 참고
```
$/usr/src/tensorrt/bin/trtexec \
  --onnx=ONNX_MODEL.onnx \
  --explicitBatch \
  --saveEngine=TRT_ENGINE.trt \
  --workspace=2048 \
  --fp16 # fp32시 해당 옵션 제거
```


## TensorRT Demo

### Python
Python TensorRT Inference에 필요한 function은 trt_utils.py에 정리되어있음

코드는 demo_trt.py 참조

TensorRT 데모는 다음 레포를 참조함

https://github.com/Tianxiaomo/pytorch-YOLOv4


### example output
```python
array([[[0.76464844, 0.87353516, 0.9038086 , ..., 0.85595703,
         0.7841797 , 0.7133789 ],
        [0.82421875, 1.0136719 , 1.1044922 , ..., 1.1005859 ,
         1.0097656 , 0.79589844],
        [0.8491211 , 1.0742188 , 1.140625  , ..., 1.1074219 ,
         1.0810547 , 0.8432617 ],
        ...,
        [0.92529297, 1.0283203 , 1.0556641 , ..., 1.0234375 ,
         1.0800781 , 0.9033203 ],
        [0.8520508 , 0.9140625 , 0.9482422 , ..., 0.9824219 ,
         1.0068359 , 0.86816406],
        [0.75683594, 0.7729492 , 0.77734375, ..., 0.75390625,
         0.8017578 , 0.77246094]]], dtype=float32)
```

### C++
trt_cpp 디렉토리 참고
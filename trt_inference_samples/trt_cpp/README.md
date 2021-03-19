Download TensorRT tar

NVIDA Download Web Page: https://developer.nvidia.com/nvidia-tensorrt-7x-download

TensorRT 7.2.2.3: https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.2/tars/TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz

작업 디렉토리로 TensrotRT.tar.gz를 옮긴 후 untar
```
tar -zxvf TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz
```

Sample Build
```
cd TensorRT-7.2.2.3/samples/sampleMNIST
make

# 실행파일은 TensorRT-7.2.2.3/bin 하위에 있음
../../bin/sample_mnist
```

디렉토리 tree가 다음과 같을때
```
trt_cpp
├── sampleTRT
│   └── cp_run_TRT.sh
│   └── Makefile
│   └── sampleTRT.cpp
├── TensorRT-7.2.2.3
│   └── ....
│   └── samples
│   └── ....
```
sampleTRT 디렉토리 안에서
```
./cp_run_TRT.sh
```
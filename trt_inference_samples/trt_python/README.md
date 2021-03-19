Python TensorRT Inference에 필요한 function은 trt.utils.py에 정리되어있음

Pytorch -> ONNX pytorch2onnx.py 참고

ONNX -> TensorRT 는 (jetson jetpack이라는 가정 하에) trt_inference_samples/models의 build_trt.sh 참고

```
$/usr/src/tensorrt/bin/trtexec \
  --onnx=ONNX_MODEL.onnx \
  --explicitBatch \
  --saveEngine=TRT_ENGINE.trt \
  --workspace=2048 \
  --fp16 # fp32시 해당 옵션 제거
  ```
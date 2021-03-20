#!/bin/bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=super_resolution.onnx \
  --explicitBatch \
  --saveEngine=super_resolution.onnx \
  --workspace=2048 \
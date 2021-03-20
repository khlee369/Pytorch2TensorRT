#!/bin/bash
DIR="$( cd "$( dirname "$0" )" && pwd -P )"

echo "-----------------------------------------"
echo "COPY to under TRT samples"
echo "-----------------------------------------"
cp -r ../sampleTRT/ ../../trt_cpp/TensorRT-7.2.2.3/samples/
cd ../../trt_cpp/TensorRT-7.2.2.3/samples/sampleTRT

echo "-----------------------------------------"
echo "Build MakeFile"
echo "-----------------------------------------"
make

echo "-----------------------------------------"
echo "Run"
echo "-----------------------------------------"
TRT="${DIR}/../../models/super_resolution.trt"
echo $TRT

./../../bin/sample_TRT \
    $TRT \

echo "-----------------------------------------"
echo "Delete Built Binary"
echo "-----------------------------------------"
rm ./../../bin/sample_TRT
rm ./../../bin/sample_TRT_debug
/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! \file sampleMNIST.cpp
//! \brief This file contains the implementation of the MNIST sample.
//!
//! It builds a TensorRT engine by importing a trained MNIST Caffe model. It uses the engine to run
//! inference on an input image of a digit.
//! It can be run with the following command line:
//! Command: ./sample_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvUffParser.h"
#include "NvUtils.h"
#include "NvOnnxParser.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include <memory>
#include <string>
#include <vector>

// #include <opencv4/opencv2/opencv.hpp>

// using namespace nvinfer1;
using namespace std;
// using namespace cudawrapper;

// TensorRT Looger
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) override
    {
        if(severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;


// destroy TensorRT objects if something goes wrong
struct TRTDestroy
{
    template< class T >
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

// Calculate size of tensor if we have all dimensions
size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

int main(int argc, char** argv)
{
    // cudaSetDevice(0);

    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " enigne.trt ";
        return -1;
    }
    std::string model_path(argv[1]);
    int batch_size = 1;

    cout << model_path << endl;

    cout << "Hello TRT" << endl;

    // https://forums.developer.nvidia.com/t/how-to-load-and-deserialize-the-engine-file/79117
    // Read trt engine
    std::vector<char> trtModelStream_;
    size_t size{ 0 };

    std::ifstream file(model_path, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream_.resize(size);
        std::cout << "size: " << trtModelStream_.size() << std::endl;
        file.read(trtModelStream_.data(), size);
        std::cout << "data: " << trtModelStream_.data() << std::endl;
        std::cout << "data type: " << typeid(trtModelStream_.data()).name() << std::endl;
        file.close();
    }
    std::cout << "size: " << size << std::endl;;

    // Deserialize engine and create context
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
    IExecutionContext *context = engine->createExecutionContext();

    // Netwokr intput output binding
    std::string INPUT_BLOB_NAME = "input";
    std::string OUTPUT_BLOB_NAME = "output";

    int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME.c_str());
    int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME.c_str());

    cout << "engine->getNbBindings(): " << engine->getNbBindings() << endl;
    cout << "inputIndex: " << inputIndex << endl;
    cout << "outputIndex: " << outputIndex << endl;
    cout << "--------------------------" << endl;

    // input output place holder
    float a_in[4] = {1,1,1,1};
    float a_out[36];
    
    // CUDA memory allocation
    std::vector< nvinfer1::Dims > input_dims; // we expect only one input
    std::vector< nvinfer1::Dims > output_dims; // and one output
    // std::vector< void* > buffers(engine->getNbBindings()); // buffers for input and output data
    void* buffers[2];
    for (size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);
        cout << engine->getBindingDimensions(i) << endl;
        cout << typeid(engine->getBindingDimensions(i)).name() << endl;
        cout << binding_size << endl;
        cout << typeid(binding_size).name() << endl;
        cudaMalloc(&buffers[i], binding_size);
        if (i==0)
        {
            cout << "intput binindg" << endl;
            input_dims.emplace_back(engine->getBindingDimensions(i));
        }
        else
        {
            cout << "output binindg" << endl;
            output_dims.emplace_back(engine->getBindingDimensions(i));
        }
    }
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Expect at least one input and one output for networkn";
        return -1;
    }

    cout << "Execute TensorRT --------------------------" << endl;

    //https://zenoahn.tistory.com/85
    cudaMemcpy(buffers[0], a_in, 16, cudaMemcpyHostToDevice);
    cudaMemcpy(buffers[1], a_out, 144, cudaMemcpyHostToDevice);

    context->executeV2(buffers);

    //https://zenoahn.tistory.com/85
    cout << "Before Memcopy --------------------------" << endl;
    cout << "show input --------------------------" << endl;
    for (int i=0; i<4; ++i){
        cout << a_in[i] << ", ";
    }
    cout << endl;
    cout << "show output --------------------------" << endl;
    for (int i=0; i<6; ++i){
        for(int j=0; j<6; ++j){
            cout << a_out[6*i+j] << ", ";
        }
        cout << endl;
    }

    cudaMemcpy(a_out, buffers[1], 144, cudaMemcpyDeviceToHost);
    cudaMemcpy(a_in, buffers[0], 16, cudaMemcpyDeviceToHost);
    cout << "After Memcopy --------------------------" << endl;
    cout << "show input --------------------------" << endl;
    for (int i=0; i<4; ++i){
        cout << a_in[i] << ", ";
    }
    cout << endl;
    cout << "show output --------------------------" << endl;
    for (int i=0; i<6; ++i){
        for(int j=0; j<6; ++j){
            cout << a_out[6*i+j] << ", ";
        }
        cout << endl;
    }
    

    // free cuda
    for (void* buf : buffers)
    {
        cudaFree(buf);
    }

    return 0;
}

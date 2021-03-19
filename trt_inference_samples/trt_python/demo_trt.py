import sys
import os
import time
import argparse
import numpy as np
import cv2
# from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from trt_tuils import HostDeviceMem, allocate_buffers, get_engine, do_inference, TRT_LOGGER, GiB

from tqdm import tqdm

if __name__ == '__main__':
    engine_path = sys.argv[1]
    
    if len(sys.argv) != 2:
        print("Usage : python3 evaluate_trt.py engine.trt")

    # engine built precision
    precision = '32'
    # input shape
    batch, ch_in, h_in, w_in = 1, 1, 3, 3

    if precision == '16':
        img = np.ones([batch, ch_in, h_in, w_in]).astype(np.float16)
    elif precision == '32':
        img = np.ones([batch, ch_in, h_in, w_in]).astype(np.float32)
    else:
        exit("Precision muste be fp16 for fp32")

    with get_engine(engine_path) as engine, engine.create_execution_context() as context:
        
        buffers = allocate_buffers(engine, batch_size=1)
        # binding input shape
        context.set_binding_shape(0, (batch, ch_in, h_in, w_in))

        inputs, outputs, bindings, stream = buffers
        inputs[0].host = np.ascontiguousarray(img)

        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        print(len(trt_outputs))

        # reshape outputs
        ch_out, h_out, w_out = 1, 9, 9
        trt_outputs[0] = trt_outputs[0].reshape(batch, ch_out, h_out, w_out)
        # trt_outputs[1] = trt_outputs[1].reshape(1, -1, num_classes)
        print(trt_outputs[0])
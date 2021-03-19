# https://tutorials.pytorch.kr/advanced/super_resolution_with_onnxruntime.html

import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init

# sample model
class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

if __name__ == '__main__':
    torch_model = SuperResolutionNet(upscale_factor=3)
    # 미리 학습된 가중치를 읽어옵니다
    model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    batch_size = 1    # 임의의 수, 보통 inference시에는 1로 고정

    # 모델을 미리 학습된 가중치로 초기화합니다
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

    # 모델을 추론 모드로 전환합니다
    torch_model.eval()

    # 모델에 대한 입력값
    # x = torch.ones(1,1,224,224)
    x = torch.ones(1,1,3,3)
    print("INPUT")
    print(x.shape)
    print(x)

    # 모델의 출력
    torch_out = torch_model(x)
    print("OUTPUT")
    print(torch_out.shape)
    print(torch_out)

    # 모델 변환
    # input_names와 output_names는
    # TensorRT 변환 후에도 사용됨으로 기억해야함
    torch.onnx.export(torch_model,               # 실행될 모델
                      x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                      "./../models/super_resolution.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                      export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                      opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                      do_constant_folding=True,  # 최적하시 상수폴딩을 사용할지의 여부
                      input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                      output_names = ['output'], # 모델의 출력값을 가리키는 이름
                      dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원, 없으면 None
                                    'output' : {0 : 'batch_size'}})
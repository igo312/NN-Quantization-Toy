from torch import quantization as Q
import torch
from torch import nn
from mmcls.apis import init_model

from utils.test import getDataLoader
from utils.fileCheck import modelExist, getCkptPath
from utils.load import imgsFromFile
from config.classifier_config import *
import os
import argparse
import pdb
import cv2

from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
from mqbench.convert_deploy import convert_deploy             # remove quant nodes for deploy

ckpt_save_path = './ckpt_path/'
if not os.path.exists(ckpt_save_path):
    os.makedirs(ckpt_save_path)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='The device option for model and data', default='cuda')
    return parser.parse_args()



if __name__ == '__main__':
    args = arg_parse()
    model_name = resnet50_8xb32_in1k
    if not modelExist(ckpt_save_path, model_name):
        os.system('mim download mmcls --config {} --dest {}'.format(model_name, ckpt_save_path))


    model = init_model(os.path.join(ckpt_save_path, model_name+'.py'),
                          getCkptPath(ckpt_save_path, model_name),
                          device=args.device)
    data_loader = getDataLoader()

    pdb.set_trace()
    backend = BackendType.Tensorrt
    model.eval()
    model = prepare_by_platform(model.backbone, backend) # with mmcls, must use model.backbone
    enable_calibration(model)
    for i, batch_data in enumerate(data_loader):
        ... # do calibration

    enable_quantization(model)
    for i, batch_data in enumerate(data_loader):
        ... # do quantization

    input_shape = {'data': [10, 3, 1333, 800]}
    convert_deploy(model, backend, input_shape)



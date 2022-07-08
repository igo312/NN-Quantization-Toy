# use pip install mmdet and use mim to quickly get detector
# mim error: solution founded in https://issuehint.com/issue/open-mmlab/mmdetection/8122
# 0706 遇到的问题： 测试精度过低
# 1. 可视化发现还不错
# 2. 使用完整的数据集进行测试看看是否有问题
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

from utils.fileCheck import modelExist, getCkptPath
from utils.test import modelEval, getDataLoader
from config.detector_config import *
import os
import argparse

ckpt_save_path = './ckpt_path/'
if not os.path.exists(ckpt_save_path):
    os.makedirs(ckpt_save_path)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='The device option for model and data', default='cuda')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    model_name = faster_rcnn_r50_fpn_1x_coco
    if not modelExist(ckpt_save_path, model_name):
        os.system('mim download mmdet --config {} --dest {}'.format(model_name, ckpt_save_path))


    model = init_detector(os.path.join(ckpt_save_path, model_name+'.py'),
                          getCkptPath(ckpt_save_path, model_name),
                          device=args.device)
    model.eval()


    #demo_result = inference_detector(model, './demo.jpg')
    #show_result_pyplot(model, './demo.jpg', demo_result)

    data_loader = getDataLoader('./config/coco_detection.py', mode='det')
    modelEval(model, data_loader, args.device, mode='det')



from torch import quantization as Q
import torch
from torch import nn
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

from utils.fileCheck import modelExist, getCkptPath
from utils.load import imgsFromFile
from config.detector_config import *
import os
import argparse
import pdb
import cv2

ckpt_save_path = './ckpt_path/'
if not os.path.exists(ckpt_save_path):
    os.makedirs(ckpt_save_path)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='The device option for model and data', default='cuda')
    return parser.parse_args()

class QModel(nn.Module):
    def __init__(self, model):
        super(QModel, self).__init__()
        self.model = model
        self.quant = Q.QuantStub()
        self.dequant = Q.DeQuantStub()

    def forward(self, img, img_metas):
        x = self.quant(img)
        x = self.model(return_loss=False, rescale=True, img=x, img_metas=img_metas)
        x = self.dequant(x)
        return x

if __name__ == '__main__':
    args = arg_parse()
    model_name = faster_rcnn_r50_fpn_1x_coco
    if not modelExist(ckpt_save_path, model_name):
        os.system('mim download mmdet --config {} --dest {}'.format(model_name, ckpt_save_path))


    model = init_detector(os.path.join(ckpt_save_path, model_name+'.py'),
                          getCkptPath(ckpt_save_path, model_name),
                          device=args.device)

    qmodel = QModel(model)


    data = imgsFromFile(model.cfg, './demo.jpg')
    img, img_metas = data['img'], data['img_metas']
    img[0] = img[0].to(args.device)
    results = qmodel(img=img, img_metas=img_metas)
    show_result_pyplot(model, './demo.jpg', results[0])
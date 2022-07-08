# Note
# some model is not uploaded to mim model store. So use different cfg may raise error
# and the downloaded ckpt may get different name compare to cfg such as resnet18_8xb16_cifar10 will get resnet18_b16x8...pth

from mmcls.apis import init_model
from utils.fileCheck import modelExist, getCkptPath
from utils.test import modelEval, getDataLoader
from config.classifier_config import *
import os
import argparse
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization import get_default_qconfig
from torch import nn
import pdb
import copy
from tqdm import tqdm

ckpt_save_path = './ckpt_path/'
if not os.path.exists(ckpt_save_path):
    os.makedirs(ckpt_save_path)

class preModel(nn.Module):
    '''
    About why creating a new module
    I have tried using mmcls.model with prepare_fx directly and it will raise `model is not subscriptable`
    At first, prepared_fx says it need nn.Module so I split the module
    and there is a notImplementError, I was really confused about that
    By coincidence, I tried prepare_fx(model.backbone), prepare_fx(model.head)
    And it raised notImplementError when tring prepare_fx(model.head).
    In result, using `self.head = new_model.head.fc` works although I do not know why.
    '''
    def __init__(self, model):
        super(preModel, self).__init__()
        new_model = copy.deepcopy(model)
        self.backbone = new_model.backbone
        self.neck = new_model.neck
        self.head = new_model.head.fc #

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)[0]
        return self.head(x)

def load_model(model_name):
    if not modelExist(ckpt_save_path, model_name):
        os.system('mim download mmcls --config {} --dest {}'.format(model_name, ckpt_save_path))
    model = init_model(os.path.join(ckpt_save_path, model_name + '.py'),
                       getCkptPath(ckpt_save_path, model_name),
                       device=args.device)
    model.eval()
    return model

def quant_fx(model, carloader=None):
    qconfig = get_default_qconfig('fbgemm') # 使用fx2trt 下述任务功能实现
    qconfig_dict = {"":qconfig}
    new_model = preModel(model)
    new_model.eval()
    prepared_model = prepare_fx(new_model, qconfig_dict)
    if carloader:
        iterator = iter(carloader)
        for _ in tqdm(range(200)):
            batchdata = next(iterator)
            img = batchdata['img']
            prepared_model(img)
    quantized_model = convert_fx(prepared_model)
    return quantized_model

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='The device option for model and data', default='cpu')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    data_loader = getDataLoader('./config/cifar10.py', mode='cls')

    model_name = resnet18_8xb16_cifar10
    model = load_model(model_name)

    quantitizaed_model = quant_fx(model, data_loader)
    print('Quantization Complete')


    modelEval(quantitizaed_model, data_loader, args.device, mode='cls_int')



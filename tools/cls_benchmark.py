import argparse
import sys

from mmcls.apis import init_model
from utils.test import getDataLoader, modelEval
from config.classifier_config import *
import torch
import pdb
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization import get_default_qconfig
from torch import nn
import copy

config_path = './ckpt_path/'
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('quant_model')
    parser.add_argument('--device', default='cpu')
    return parser.parse_args()


class preModel(nn.Module):
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

if __name__ == '__main__':
    args = arg_parse()
    data_loader = getDataLoader('./config/cifar10.py', mode='cls')
    #pdb.set_trace()

    dummy_input = torch.randn([1,3,32,32]).to(args.device)
    model_name = resnet18_8xb16_cifar10
    # load unquantitized model
    model = init_model(config_path+model_name+'.py', args.model, device=args.device)
    model.eval()

    # load quantitized model, first use unquant model to gen a quantitized arch with random weight, then load the weight
    default_qconfig = get_default_qconfig('fbgemm')
    qconfig_dict = {"":default_qconfig}
    premodel = preModel(model)
    premodel.eval()
    model_prepared = prepare_fx(premodel, qconfig_dict)
    qmodel = convert_fx(model_prepared)
    qmodel.load_state_dict(torch.load(args.quant_model))

    # test
    print('Quantitized model testing...')
    modelEval(qmodel, data_loader, args.device, mode='cls_int')
    print('Unquantitized model testing...')
    modelEval(model, data_loader, args.device, mode='cls')


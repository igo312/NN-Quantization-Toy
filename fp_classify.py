# Note
# some model is not uploaded to mim model store. So use different cfg may raise error
# and the downloaded ckpt may get different name compare to cfg such as resnet18_8xb16_cifar10 will get resnet18_b16x8...pth
# the cifar based model only include resnet series
from mmcls.apis import init_model
from utils.fileCheck import modelExist, getCkptPath
from utils.test import modelEval, getDataLoader
from config.classifier_config import *
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
    model_name = resnet18_8xb16_cifar10
    if not modelExist(ckpt_save_path, model_name):
        os.system('mim download mmcls --config {} --dest {}'.format(model_name, ckpt_save_path))

    model = init_model(os.path.join(ckpt_save_path, model_name + '.py'),
                       getCkptPath(ckpt_save_path, model_name),
                       device=args.device)
    model.eval()


    #demo_result = inference_detector(model, './demo.jpg')
    #show_result_pyplot(model, './demo.jpg', demo_result)

    data_loader = getDataLoader('./config/cifar10.py', mode='cls')
    modelEval(model, data_loader, args.device, mode='cls')



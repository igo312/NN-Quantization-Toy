import copy
import pdb

from fuseModules import fuse_conv_bn
from weightEqualization import weight_equalization
from DFQ.utils import test, QuantConv2d, QuantLinear, load_model

import torchvision
import torch
from torch import fx
from torch import nn
from tqdm import tqdm

def _parent_name(target) :
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

def replace_node(graph, node, modules):
    def _replace_modules(node, new_node, modules):
        # create new module
        old_module = modules[node.target]
        if isinstance(modules[node.target], nn.Conv2d):
            inc, outc, k, stride, pad, dia, groups = old_module.in_channels, old_module.out_channels,\
            old_module.kernel_size, old_module.stride, old_module.padding, old_module.dilation, old_module.groups
            new_module = QuantConv2d(inc, outc, k, stride=stride, padding=pad, dilation=dia, groups=groups)
        elif isinstance(modules[node.target], nn.Linear):
            in_features, out_features = old_module.in_features, old_module.out_features
            new_module = QuantLinear(in_features, out_features)

        new_module.weight = old_module.weight
        new_module.bias = old_module.bias
        modules[new_node.target] = new_module
        del modules[node.target]

    if node.op != 'call_module': return False
    if 'quant' in node.target: return False
    if isinstance(modules[node.target], nn.Conv2d):
        with graph.inserting_after(node):
            new_node = graph.call_module(node.target+'_quant', args=node.args)
    elif isinstance(modules[node.target], nn.Linear):
        with graph.inserting_after(node):
            new_node = graph.call_module(node.target+'_quant', args=node.args)
    else:
        return False

    _replace_modules(node, new_node, modules)
    node.replace_all_uses_with(new_node)
    graph.erase_node(node)

    return True

def quant_model(model):
    # NOTE please copy graph and modules rather than copy model directly
    graph = copy.deepcopy(model.graph)
    modules = copy.deepcopy(dict(model.named_modules()))
    for node in tqdm(graph.nodes):
        replace_node(graph, node, modules)
    #model.recompile()
    return fx.GraphModule(modules, graph)

if __name__ == '__main__':
    # model load
    model = torchvision.models.resnet18(pretrained=10)
    model.fc = torch.nn.Linear(512, 10)
    model = load_model(model, ckpt='./ckpt/resnet18_cifar10_best.pth')
    # model = torchvision.models.mobilenet_v2()
    # model.classifier[1] = torch.nn.Linear(1280, 10)
    # model = load_model(model, ckpt='./ckpt/mobilenet_cifar10_best.pth')

    ori_model = copy.deepcopy(model)
    model.eval()
    # Do the (conv2d, BN) fusion and cross layer equalization
    # TODO if use copy.deepcopy, it cannot add new module, why?
    model = fuse_conv_bn(model)
    fuse_model = quant_model(model)
    model = weight_equalization(model)
    cle_model = quant_model(model)

    # replace the module
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Resize((128, 128)),  # mobilenet
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    data = torchvision.datasets.CIFAR10(root='./data/', train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=32, num_workers=8, shuffle=False)


    test(ori_model, loader, prefix='Model Ori', device="cuda:0") # resnet18 acc 72.24 mobilnet 75.27
    test(fuse_model, loader, prefix='Model FUSE BN', device="cuda:0") # resnet18 acc 71.98 mobilenet 75.16
    test(cle_model, loader, prefix='Model CLE', device="cuda:0") # resnet18 acc 72.06 mobilenet 73.86




# Used cifar10
import copy
import pdb

from fuseModules import fuse_conv_bn
import torchvision
import torch
from tqdm import tqdm
from torch.fx import symbolic_trace
import copy
from torch import nn
from torch import fx
from torch.quantization import quantize_fx
from DFQ.utils import load_model, test

eps = 1e-8
# TODO: there should be bias absortion process, will do it later.

# weight function
def weight_equalization(model):
    assert not model.training
    model = copy.deepcopy(model)
    traced = symbolic_trace(model)
    graph = traced.graph

    modules = dict(traced.named_modules())
    patterns = ((nn.Conv2d, nn.ReLU, nn.Conv2d),
                (nn.Conv2d, nn.ReLU6, nn.Conv2d))
    prev, cur = None, None
    for node in graph.nodes:
        prev = cur
        cur = node
        if not isinstance(prev, fx.Node): continue
        if not isinstance(cur, fx.Node): continue
        for pattern in patterns:
            # the prev should be a instance of nn.Relu and the cur should be a instance of nn.Conv2d
            if pattern_match(prev, cur, pattern, modules):
                equalization(prev, cur, modules)
    return fx.GraphModule(traced, graph)

def pattern_match(prev, cur, patterns, modules):
    def check(node):
        # TODO sometimes relu can be a call_function, be attention about that
        if node.op != 'call_module': return False
        if not isinstance(node, fx.Node): return False
        if node.target not in modules: return False
        if node.args[0].target not in modules: return False
        return True
    if check(prev) and check(cur):
        query = (prev.args[0].target, prev.target, cur.target)
        for current_pattern, current_node in zip(patterns, query):
            if not isinstance(modules[current_node], current_pattern): return False
    else:
        return False
    return True

def equalization(prev, cur, modules):
    # conv weight: [cout, cin, w, h]
    w1 = modules[prev.args[0].target]
    w2 = modules[cur.target]
    # NOTE: weight eq of mobilenet,
    # hint from https://github.com/jakc4103/DFQ/blob/6f15805cfdbf2769275defd54728df0a5d30dbc6/dfq.py#L30
    w1_weight, w1_bias, w2_weight = w1.weight.clone(), w1.bias.clone(), w2.weight.clone()

    num_group = 1
    if w1_weight.shape[0] != w2_weight.shape[1]:
        num_group = w1_weight.shape[0] // w2_weight.shape[1]
    group_channel_i = w1_weight.shape[0] // num_group # the weight num per group
    group_channel_o = w2_weight.shape[0] // num_group

    for g in range(num_group):
        c_start_i = g*group_channel_i
        c_end_i = (g+1)*group_channel_i
        w1_weight_sub = w1_weight[c_start_i:c_end_i]

        c_start_o = g * group_channel_o
        c_end_o = (g + 1) * group_channel_o
        w2_weight_sub = w2_weight[c_start_o:c_end_o]

        cout, cin = w1_weight_sub.size(0), w1_weight_sub.size(1)
        for c_idx in range(cout):
            range1 = torch.max(w1_weight_sub[c_idx]) - torch.min(w1_weight_sub[c_idx])
            range2 = torch.max(w2_weight_sub[:, c_idx]) - torch.min(w2_weight_sub[:, c_idx])
            s = (1/(range1+eps)) * torch.sqrt(range1*range2+eps)
            s.clamp_(1e-8, 1e8)
            w1_weight[c_start_i+c_idx].mul_(s)
            if w1_bias is not None:
                w1_bias[c_start_i+c_idx].mul_(s)
            w2_weight[c_start_o:c_end_o, c_idx].mul_(1/s)

    w1.weight, w1.bias = nn.Parameter(w1_weight), nn.Parameter(w1_bias)
    w2.weight = nn.Parameter(w2_weight)

# TODO there can be statical function to analyze the range from different channel
def equalizationv1(prev, cur, modules):
    # conv weight: [cout, cin, w, h]
    w1 = modules[prev.args[0].target]
    w2 = modules[cur.target]
    # NOTE: weight eq of mobilenet,
    # hint from https://github.com/jakc4103/DFQ/blob/6f15805cfdbf2769275defd54728df0a5d30dbc6/dfq.py#L30
    w1_weight, w1_bias, w2_weight = w1.weight.clone(), w1.bias.clone(), w2.weight.clone()

    num_group = 1
    if w1_weight.shape[0] != w2_weight.shape[1]:
        num_group = w1_weight.shape[0] // w2_weight.shape[1]
    group_channel_i = w1_weight.shape[0] // num_group # the weight num per group
    group_channel_o = w2_weight.shape[0] // num_group

    for g in range(num_group):
        c_start_i = g*group_channel_i
        c_end_i = (g+1)*group_channel_i
        w1_weight_sub = w1_weight[c_start_i:c_end_i]

        c_start_o = g * group_channel_o
        c_end_o = (g + 1) * group_channel_o
        w2_weight_sub = w2_weight[c_start_o:c_end_o]

        cout, cin = w1_weight_sub.size(0), w1_weight_sub.size(1)
        w1_weight_sub = w1_weight_sub.view(cout, -1)
        w2_weight_sub = w2_weight_sub.permute(1,0,2,3).contiguous().view(cout, -1)

        w1_max, w1_min = w1_weight_sub.max(dim=1)[0], w1_weight_sub.min(dim=1)[0]
        w2_max, w2_min = w2_weight_sub.max(dim=1)[0], w2_weight_sub.min(dim=1)[0]
        range1, range2 = w1_max-w1_min, w2_max-w2_min

        scale = torch.sqrt(range1*range2)/(range1+1e-8)
        scale.clamp_(1e-8, 1e8)
        w1_s = scale.reshape(cout, 1, 1, 1)

        w1_weight[c_start_i:c_end_i] = nn.Parameter(w1_weight[c_start_i:c_end_i]*w1_s)
        if isinstance(w1.bias, nn.Parameter):
            w1_bias[c_start_i:c_end_i] = nn.Parameter(w1_bias[c_start_i:c_end_i]*scale)

        w2_s = scale.reshape(1, cout, 1, 1)
        w2_weight[c_start_o:c_end_o] = nn.Parameter(w2_weight[c_start_o:c_end_o]/w2_s)
    #pdb.set_trace()
    w1.weight, w1.bias = nn.Parameter(w1_weight), nn.Parameter(w1_bias)
    w2.weight = nn.Parameter(w2_weight)


if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((128,128)), # mobilenet
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),

    ])
    data = torchvision.datasets.CIFAR10(root='./data/', train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=32, num_workers=8, shuffle=False)

    model = torchvision.models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(1280, 10)
    model = load_model(model, ckpt='./ckpt/mobilenet_cifar10_best.pth')
    # model = torchvision.models.resnet18(pretrained=10)
    # model.fc = torch.nn.Linear(512, 10)
    # model = load_model(model, ckpt='./ckpt/resnet18_cifar10_best.pth')
    print('Float32 Testing')
    model.eval()

    test(model, loader, 'Origin Model') # mobilenet 75
    model_fuse = fuse_conv_bn(model)
    model_fuse.eval()
    test(model_fuse, loader, 'Fuse_conv_bn Model')

    # weight equalization
    #pdb.set_trace()
    model_eqa = weight_equalization(model_fuse)
    model_eqa.eval()
    test(model_eqa, loader, 'Weight_Equalization Model') # mobilenet 74


    qconfig = {"": torch.quantization.get_default_qconfig("gpu")}

    print('Int8 Testing')
    rand_inp = torch.randn((1,3,128,128))
    model_int = copy.deepcopy(model)
    model_int = model_int.to('cpu')
    model_int.eval()
    model_int = quantize_fx.prepare_fx(model_int, qconfig)
    model_int(rand_inp)
    model_int = quantize_fx.convert_fx(model_int)
    test(model_int, loader, 'Quant Origin Model', device="cpu")

    model_fuse_int = copy.deepcopy(model_fuse)
    model_fuse_int = model_fuse_int.to('cpu')
    model_fuse_int.eval()
    model_fuse_int = quantize_fx.prepare_fx(model_fuse_int, qconfig)
    model_fuse_int(rand_inp)
    model_fuse_int = quantize_fx.convert_fx(model_fuse_int)
    test(model_fuse_int, loader, 'Quant Fuse_conv_bn Model', device="cpu")


    model_eqa_int = copy.deepcopy(model_eqa)
    model_eqa_int = model_eqa_int.to('cpu')
    model_eqa_int.eval()
    model_eqa_int = quantize_fx.prepare_fx(model_eqa_int, qconfig)
    model_eqa_int(rand_inp)
    model_eqa_int = quantize_fx.convert_fx(model_eqa_int)
    test(model_eqa_int, loader, 'Quant Weight_Equalization Model',device="cpu")





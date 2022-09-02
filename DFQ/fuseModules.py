# The code is almost from https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L50
import pdb

import torchvision
from torch import fx
from torch.fx import symbolic_trace
from torch import nn
import copy
import torch
# fuse conv bn

def matches_module_pattern(pattern, node, modules):
    if len(node.args) == 0:
        return False
    nodes = (node.args[0], node)
    for expected_pattern, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False

        if current_node.op != 'call_module':
            # target 是实例名称
            # name 似乎是torch自己给的一个名字
            return False

        if current_node.target not in modules:
            return False

        #Todo 怎么根据node判断是否是conv, 如下所示
        if not isinstance(modules[current_node.target], expected_pattern):
            return False

    return True

def _fuse_conv_bn_eval(conv, bn):
    assert(not (conv.training or bn.training))
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = fused_weight_bias(conv.weight, conv.bias,
                                            bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    return fused_conv

def fused_weight_bias(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    '''
    :param conv_weight: (cout, cin, s, s)
    :param conv_bias: (cout)
    :param bn_mean: (cout)
    :param bn_var: (cout)
    :param bn_eps: scale
    :param bn_weight: (cout)
    :param bn_bias: (cout)
    :return:
    '''
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps) # 加r表示倒数

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)

def _parent_name(target) :
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

def replace_node_module(node: fx.Node, modules, new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)

def fuse_conv_bn(model):
    '''
    :param model: a traind model
    :return: the default mode is un-inplaced, which means the function will copy a model then fuse the BN_CONV
    '''
    assert not model.training
    model = copy.deepcopy(model)
    symbolic_traced = symbolic_trace(model)
    graph = symbolic_traced.graph
    modules = dict(symbolic_traced.named_modules()) # it should use traced modules not model
    #new_graph = copy.deepcopy(graph)

    pattern = (nn.Conv2d, nn.BatchNorm2d)
    #[old line] for node in graph.nodes: (bug, should use the node from new graph )
    for node in graph.nodes:
        if matches_module_pattern(pattern, node, modules):
            if len(node.args[0].users) > 1: continue
            # 如果为true，则当前node为bn，且前一层是conv
            fused_conv = _fuse_conv_bn_eval(modules[node.args[0].target], modules[node.target])
            replace_node_module(node.args[0], modules, fused_conv)
            # 融合删除节点的一个方法
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)

    return fx.GraphModule(symbolic_traced, graph)


# TODO not done yet
def fuse_conv_relu(model):
    assert not model.training
    model = copy.deepcopy(model)
    symbolic_traced = symbolic_trace(model)
    graph = symbolic_traced.graph
    modules = dict(model.named_modules())
    new_graph = copy.deepcopy(graph)
    pattern = (nn.Conv2d, nn.ReLU)



if __name__ == '__main__':
    '''
    The process of fusing conv and bn 
    1. using graph to find the matched pair (conv, bn)
    2. fused the weight and return a good conv 
    3. replace the conv node 
    4. erase the bn node
    '''

    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    symbolic_traced = symbolic_trace(model)
    graph = symbolic_traced.graph
    modules = dict(model.named_modules())
    new_graph = copy.deepcopy(graph)

    pattern = (nn.Conv2d, nn.BatchNorm2d)
    for node in new_graph.nodes:
        if matches_module_pattern(pattern, node, modules):
            # 如果为true，则当前node为bn，且前一层是conv
            fused_conv = fuse_conv_bn(modules[node.args[0].target], modules[node.target])
            replace_node_module(node.args[0], modules, fused_conv)
            # 融合删除节点的一个方法
            node.replace_all_uses_with(node.args[0])
            new_graph.erase_node(node)
    new_graph.print_tabular()
    print('done')


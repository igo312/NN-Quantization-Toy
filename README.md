# Quantization Toy

基于[mmdetection](https://github.com/open-mmlab/mmdetection)与[mmclassification](https://github.com/open-mmlab/mmclassification),实现无需模型训练，专注于对模型量化的学习。

This project is just a naive quantization toy which is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [mmclassification](https://github.com/open-mmlab/mmclassification). The best thing we can focus on quantization, the model can be downloaded from openmmlab.


## Installation
**Step 1**: create a conda environment, activate it and install pytorch
```
# The code is based on CUDA10.2 pytorch1.8.1
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
```

**Step 2**: install mmcv
```
pip install -U openmim
mim install mmcv-full

# code change, if you do not do this step, the process of downloading trained model will raise error.
# assuming envs is your python install path
vim envs/lib/python3.7/site-packages/mim/commands/search.py
# about line:392, there is a code as
`if collection_name :`
# we need rewrite it as
`if collection_name and collection_name in name2collection.keys():`
# and done
```

**Step 3**: install the necessary library
```
pip install -r requirements.txt
```

**Step 4**: install `MQBench`
```
git clone https://github.com/ModelTC/MQBench.git
cd MQBench
python setup.py install
```

## Demo

1. If you want to test a fp model you can just run
```
python fp_classify
```

2. If you want to quantize a model you can just run
```
python fp2int_cls.py
```

3. if you want to compare the speed of origin model and quanted model, you can run
you also can find out how to load a quantitized model here which follows [link](https://discuss.pytorch.org/t/how-to-load-quantized-model-for-inference/140283)
```
python ./tools/cls_benchmark ...your_cifar10_resnet18.pth ...your_quanted_cifar10_resnet18.pth
```
**0708 Attention**: Right now we only support classification test Although you can find `fp_detection.py` work as well if you sample the coco data and give the right to the `config/coco_detection.py`



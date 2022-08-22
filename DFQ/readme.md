[Pytorch fx]A complement about "Data-Free Quantization Through Weight Equalization and Bias Correction" [Paper Url](https://ieeexplore.ieee.org/document/9008784/)

# Cifar 10 quantization

We cannot download a ckpt of cifar10 directly, so I train a model with Cifar10 in `train.py`.

```
python train.py # it will download cifar10 and save to ./data, and only save the best model in ./ckpt
python weightEqualization.py # it will fuse the BN and conv2d first, and do weight equalization/ layer equalization
```

It really confuse me that quantitize a origin mobilenet and give it a random data, the quantitized model still get high
accuray. And after weight equalization, it will cause accuracy decrease, which is more obvious in imagenet test.

# ImageNet 1k
## Data Download
you can download the two following files, change the data path in `imagenet_format.py` and run it.
```
#BaiDu disk
##validation data
url: https://pan.baidu.com/s/1rVPHWu4LzP7D73ZtcHIWJw
code: f6th

##devkit
url: https://pan.baidu.com/s/1mmFYTFoYAUpi5v7jgOM78A
code: m5q2
```

## test
you should change the `root` in `weightEqualization_ImageNet.py` then run, you can find after weight equalization, the
accuracy has big decrease.
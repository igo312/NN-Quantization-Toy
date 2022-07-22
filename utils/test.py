import torch
import mmcv
from mmdet.datasets import build_dataloader as build_det_dataloader, build_dataset as build_det_dataset
from mmcls.datasets import build_dataloader as build_cls_dataloader, build_dataset as build_cls_dataset
from mmcv import Config
from mmdet.apis import show_result_pyplot
import pdb

def modelEval(model, data_loader, device, mode):
    assert mode in ['cls', 'det', 'cls_int', 'det_int']
    model.eval()
    def evaldet(model, batch_data):
        img, img_metas = batch_data['img'], batch_data['img_metas']
        pdb.set_trace()
        img[0] = img[0].to(device)
        img_metas = img_metas[0].data
        with torch.no_grad():
            if 'int' in mode: result = model(img)
            else:result = model(return_loss=False,  img=img, img_metas=img_metas)
        return result

    def evalcls(model, batch_data):
        img, img_metas = batch_data['img'], batch_data['img_metas']
        img = img.to(device)
        with torch.no_grad():
            if 'int' in mode: result = model(img)
            else:result = model(return_loss=False,  img=img, img_metas=img_metas)
        return result

    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, batch_data in enumerate(data_loader):
        if 'cls' in mode: result = evalcls(model, batch_data)
        else: result = evaldet(model, batch_data)

        # show detection result
        #show_result_pyplot(model, img_metas[0][0]['filename'], result[0])
        batch_size = len(result)
        results.extend(result)
        for _ in range(batch_size):
            prog_bar.update()

    if 'cls' in mode:
        acc_result = dataset.evaluate(results, **dict(metric='accuracy'))
        print('\n cifar10 top-1 acc:{}, top-5 acc:{}'.format(acc_result['accuracy_top-1'], acc_result['accuracy_top-5']))
    else: dataset.evaluate(results, **dict(metric='bbox'))

def getDataLoader(cfg_path, mode):
    assert mode in ['det', 'cls']
    #### dataloader #####
    data_cfg = Config.fromfile(cfg_path)
    if mode=='cls':
        dataset = build_cls_dataset(data_cfg.data.test)
    else:
        dataset = build_det_dataset(data_cfg.data.test)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False,)
        #num_gpus=1,
        #persistent_workers=False)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **data_cfg.data.get('test_dataloader', {})
    }

    if mode == 'cls':
        data_loader = build_cls_dataloader(dataset, **test_loader_cfg)
    else:
        data_loader = build_det_dataloader(dataset, **test_loader_cfg)
    return data_loader

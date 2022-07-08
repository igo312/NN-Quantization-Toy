# To sample coco json with a sample rate
import json
from pycocotools.coco import COCO
import random
import argparse
import os
random.seed(22)

from tqdm import tqdm



def sample(coco_path, save_path, sample_rate=0.1):
    coco = COCO(coco_path)
    imgIds = list(coco.imgs.keys())
    random.shuffle(imgIds)
    sample_num = int(len(imgIds)*sample_rate)
    imgIds = imgIds[:sample_num]

    with open(coco_path, 'r') as f:
        coco_data = json.load(f)

    data = dict(
        images = [],
        annotations = [],
        categories= coco_data['categories'],
        info=coco_data['info'],
        licenses=coco_data['licenses']
    )
    #import pdb; pdb.set_trace()
    for img_meta in tqdm(coco.loadImgs(imgIds)):
        data['images'].append(img_meta)
        anno_id = coco.getAnnIds(img_meta['id'])
        anno = coco.loadAnns(anno_id)
        data['annotations'].extend(anno)
    with open(save_path, 'w') as f:
        json.dump(data, f)

def argParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='the source of coco')
    parser.add_argument('--sample_rate', default=0.1, type=float)
    parser.add_argument('--save_path', help='the save dir path')
    return parser.parse_args()

if __name__ == '__main__':
    args = argParse()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    sample(os.path.join(args.path, 'instances_val2017.json'), \
           os.path.join(args.save_path, 'instances_val2017_sample{}.json').format(args.sample_rate), \
           args.sample_rate)
    print('[Sample Done] annotations is saved to {}'.format(args.save_path))

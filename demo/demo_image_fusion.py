import argparse
import cv2
import mmcv
import torchvision
import os
import os.path as osp
from mmcv import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import get_dataset

def save_fusion_result(config_file, out_dir, checkpoint_file=None, img_dir=None):
    cfg = Config.fromfile(config_file)
    data_test = cfg.data.test
    dataset = get_dataset(data_test)
    classnames = [dataset.CLASSES]
    # import ipdb;ipdb.set_trace()
    # use checkpoint path in cfg
    if not checkpoint_file:
        checkpoint_file = osp.join(cfg.work_dir,'latest.pth')
    # use testset in cfg
    if not img_dir:
        img_dir = data_test.img_prefix
        ann_file = osp.join(img_dir, 'test.txt')  # Replace 'test.txt' with YOUR TEXT containing the list of test images
        newimg_dir = osp.join(img_dir)
        img_list = mmcv.list_from_file(ann_file)

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    out_filepath = '/root/MoE-Fusion/demo/Fusionresults'
    if not os.path.exists(out_filepath):
        os.mkdir(out_filepath)

    for img_name in img_list:
        img_name_tir = img_name + '.png'
        img_name_rgb = img_name + '.png'
        img_name_fusion = img_name + '.png'
        img_path_i = osp.join(newimg_dir, 'Ir', img_name_tir)   # Replace 'Ir' with YOUR INFRARED DIR
        img_path_r = osp.join(newimg_dir, 'Vis', img_name_rgb)  # Replace 'Vis' with YOUR VISIBLE DIR
        img_out_path = osp.join(out_filepath, img_name_fusion)
        image_fusion = inference_detector(model, img_path_r, img_path_i)
        torchvision.utils.save_image(image_fusion, img_out_path)
        print(img_out_path)

if __name__ == '__main__':
    save_fusion_result('/root/MoE-Fusion/configs/MoE_Fusion.py', '/root/MoE-Fusion/demo/', '/root/MoE-Fusion/work_dirs/MoE_Fusion/epoch_24.pth', img_dir=None)
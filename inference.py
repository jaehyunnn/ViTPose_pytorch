import argparse
import os
import os.path as osp
from loguru import logger
import torch
from torch import Tensor
from glob import glob
from pathlib import Path
import cv2
import numpy as np

from time import time
from PIL import Image
from torchvision.transforms import transforms
from typing import Tuple
from vitpose.models.model import ViTPose
from vitpose.utils.visualization import draw_points_and_skeleton, joints_dict
from vitpose.utils.dist_util import get_dist_info, init_dist
from vitpose.utils.top_down_eval import keypoints_from_heatmaps

__all__ = ['inference']


@torch.no_grad()
def vitpose_inference_model(
              model_cfg: dict, ckpt_path: Path,
              device: torch.device,
              out_dir: None,
                input_size: Tuple[int, int] = (256, 192),
              # save_result: bool=True
              ) -> np.ndarray:
    # Prepare model
    vit_pose = ViTPose(model_cfg)

    ckpt = torch.load(ckpt_path)
    if 'state_dict' in ckpt:
        vit_pose.load_state_dict(ckpt['state_dict'])
    else:
        vit_pose.load_state_dict(ckpt)
    vit_pose.to(device)
    logger.info(f"ViTPose model loaded from {ckpt_path}")

    def run_once(img_path):
        if isinstance(img_path, str):
            img = Image.open(img_path)
        elif isinstance(img_path, np.ndarray):
            img = Image.fromarray(img_path)
        else:
            raise TypeError(f"Unknown input type {type(img_path)}")

        org_w, org_h = img.size
        logger.info(f">>> Original image size: {org_h} X {org_w} (height X width)")
        logger.info(f">>> Resized image size: {input_size[1]} X {input_size[0]} (height X width)")
        logger.info(f">>> Scale change: {org_h / input_size[1]}, {org_w / input_size[0]}")
        img_tensor = transforms.Compose(
            [transforms.Resize((input_size[1], input_size[0])),
             transforms.ToTensor()]
        )(img).unsqueeze(0).to(device)

        # Feed to model
        tic = time()

        heatmaps = vit_pose(img_tensor).detach().cpu().numpy()  # N, 17, h/4, w/4
        elapsed_time = time() - tic
        logger.info(f">>> Output size: {heatmaps.shape} ---> {elapsed_time:.4f} sec. elapsed [{elapsed_time ** -1: .1f} fps]\n")

        # points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w // 2, org_h // 2]]),
                                               scale=np.array([[org_w, org_h]]),
                                               unbiased=True, use_udp=True)
        points = np.concatenate([points[:, :, ::-1], prob], axis=2)



        return points
    return run_once

if __name__ == "__main__":
    # from vitpose.configs.ViTPose_base_coco_256x192 import model as model_cfg
    # from vitpose.configs.ViTPose_base_coco_256x192 import data_cfg
    from vitpose.configs.ViTPose_huge_coco_256x192 import model as model_cfg
    from vitpose.configs.ViTPose_huge_coco_256x192 import data_cfg

    img_paths = glob('/mnt/x/data_repos/volleyball/HierVolley/player_crops/*.jpg')
    out_dir = '/mnt/x/data_repos/volleyball/HierVolley/player_crops_vitpose'

    CUR_DIR = "/mnt/x/data_repos/models/ViTPose"  # osp.dirname(__file__)
    CKPT_PATH = f"{CUR_DIR}/vitpose-h-multi-coco.pth"

    img_size = data_cfg['image_size']

    kpt_model = vitpose_inference_model(input_size=img_size, model_cfg=model_cfg, ckpt_path=CKPT_PATH,
                                        device=torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'),
                                        out_dir=out_dir)


    for img_path in img_paths:
        image = cv2.imread(img_path)
        # convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        keypoints = kpt_model(img_path=image)

        for pid, point in enumerate(keypoints):
            img = np.array(image)[:, :, ::-1]  # RGB to BGR for cv2 modules
            img = draw_points_and_skeleton(img.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
                                           points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                           points_palette_samples=10, confidence_threshold=0.4)
            cv2.imshow('result', img)
            cv2.waitKey(0)
            # os.makedirs(out_dir, exist_ok=True)
            # out_image_path = osp.join(out_dir,osp.basename(img_path))
            # cv2.imwrite(out_image_path, img)
            # logger.success(f"created {out_image_path}")
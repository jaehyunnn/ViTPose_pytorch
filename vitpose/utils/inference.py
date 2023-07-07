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


from vitpose.utils.dist_util import get_dist_info, init_dist
from vitpose.utils.top_down_eval import keypoints_from_heatmaps
from vitpose.utils.commons import get_support_dir, create_list_chunks

from typing import Optional

import importlib

def dynamic_import(module_name, variable_name):
    module = importlib.import_module(module_name)
    variable = getattr(module, variable_name)
    return variable


__all__ = ['vitpose_inference_model']

@torch.no_grad()
def vitpose_inference_model(device=None,
                            onnx=False,
                            half=True,
                            verbostiy=0,
                            batch_size=32,
                            model_spec='large-body25',
                            weights_dir:Optional[str]=None,
                            ) -> np.ndarray:
    # Prepare model

    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if weights_dir is None: weights_dir = osp.join(get_support_dir(),'vitpose')

    kpt_mode, model_mode = model_spec.split('-')
    assert model_mode in ['base', 'large', 'huge'], f"Unknown model mode {model_mode}"
    assert kpt_mode in ['coco17', 'body25'], f"Unknown number of keypoints {kpt_mode}"

    model_cfg = dynamic_import(f'vitpose.configs.{kpt_mode}.ViTPose_{model_mode}_coco_256x192', 'model')
    data_cfg = dynamic_import(f'vitpose.configs.{kpt_mode}.ViTPose_{model_mode}_coco_256x192', 'data_cfg')

    vitpose_ckpt_path = osp.join(weights_dir, kpt_mode, f"vitpose-{model_mode}.{'onnx' if onnx else 'pth'}")
    if onnx and not osp.exists(vitpose_ckpt_path):
        export_onnx(device=device, half=half, weights_dir=weights_dir, model_spec=model_spec)

    if device == 'cpu':
        half = False

    vit_pose = ViTPose(model_cfg)
    if half:
        vit_pose.half()

    assert osp.exists(vitpose_ckpt_path), f"ViTPose ckpt not found at {vitpose_ckpt_path}"

    input_size = data_cfg['image_size']

    if onnx:
       # assert 'cuda' in device, f"ONNX Runtime only supports CUDA device, got {device}"
        logger.info(f'Loading {vitpose_ckpt_path} for ONNX Runtime inference...')
        import onnxruntime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(vitpose_ckpt_path, providers=providers)
        output_names = [x.name for x in session.get_outputs()]
        # meta = session.get_modelmeta().custom_metadata_map  # metadata
        # if 'stride' in meta:
        #     stride, names = int(meta['stride']), eval(meta['names'])
    else:
        ckpt = torch.load(vitpose_ckpt_path)
        if 'state_dict' in ckpt:
            vit_pose.load_state_dict(ckpt['state_dict'])
        else:
            vit_pose.load_state_dict(ckpt)

        vit_pose.eval()
        vit_pose.to(device)

    logger.info(f"ViTPose model loaded from {vitpose_ckpt_path}")

    def from_numpy(x):
        return torch.from_numpy(x).to(device) if isinstance(x, np.ndarray) else x

    def c2c(x):
        return x.cpu().numpy() if isinstance(x, Tensor) else x

    def run_once(images):
        assert isinstance(images, list), f"Input must be a list of images, got {type(images)}"
        img_tensors = []
        orig_sizes = []
        for img_path in images:
            if isinstance(img_path, str):
                img = Image.open(img_path)
            elif isinstance(img_path, np.ndarray):
                img = Image.fromarray(img_path)

            else:
                raise TypeError(f"Unknown input type {type(img_path)}")

            org_w, org_h = img.size
            orig_sizes.append([org_w, org_h])
            if verbostiy > 0:
                logger.info(f">>> Original image size: {org_h} X {org_w} (height X width)")
                logger.info(f">>> Resized image size: {input_size[1]} X {input_size[0]} (height X width)")
                logger.info(f">>> Scale change: {org_h / input_size[1]}, {org_w / input_size[0]}")
            img_tensors.append(transforms.Compose(
                [transforms.Resize((input_size[1], input_size[0])),
                 transforms.ToTensor()]
            )(img).unsqueeze(0))

        ids_batches = create_list_chunks(list(range(len(img_tensors))), group_size=batch_size, overlap_size=0,
                                         drop_smaller_batches=False)
        batch_points = []
        for ids_batch in ids_batches:
            cur_img_tensors = torch.cat([img_tensors[i] for i in ids_batch], dim=0)
            if half:
                cur_img_tensors = cur_img_tensors.half()
            cur_img_tensors = cur_img_tensors.to(device)
            cur_orig_sizes = np.array([orig_sizes[i] for i in ids_batch])

            # Feed to model
            tic = time()

            if onnx:
                cur_img_tensors = cur_img_tensors.cpu().numpy()  # torch to numpy
                heatmaps = session.run(output_names, {'input_0': cur_img_tensors})
                # if isinstance(heatmaps, (list, tuple)):
                #     heatmaps = from_numpy(heatmaps[0]) if len(heatmaps) == 1 else [from_numpy(x) for x in heatmaps]
                # else:
                #     heatmaps = from_numpy(heatmaps)
                heatmaps = c2c(heatmaps[0]) if len(heatmaps) == 1 else np.array([c2c(x) for x in heatmaps])

                # heatmaps = heatmaps.astype(np.float32)
                # heatmaps = session.run(output_names, {session.get_inputs()[0].name: cur_img_tensors})
            else:
                heatmaps = vit_pose(cur_img_tensors)  # N, 17, h/4, w/4
                if half:
                    heatmaps = heatmaps.type(torch.float)
                heatmaps = heatmaps.detach().cpu().numpy()

            #breakpoint()
            elapsed_time = time() - tic
            if verbostiy > 0:
                logger.info(
                    f">>> Output size: {heatmaps.shape} ---> {elapsed_time:.4f} sec. elapsed [{len(cur_img_tensors) / elapsed_time: .1f} fps]\n")

            # points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
            points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=(cur_orig_sizes / 2).astype(int),
                                                   scale=cur_orig_sizes,
                                                   unbiased=True, use_udp=True)
            batch_points.append(np.concatenate([points[:, :, ::-1], prob], axis=2))

        if len(batch_points) >0:
            return np.concatenate(batch_points, axis=0)
        else:
            return None

    return run_once


def export_onnx(device=None,
                half=False,
                model_spec='large-body25',
                weights_dir: Optional[str] = None) -> np.ndarray:
    # Prepare model

    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if weights_dir is None: weights_dir = osp.join(get_support_dir(), 'vitpose')

    kpt_mode, model_mode = model_spec.split('-')
    assert model_mode in ['base', 'large', 'huge'], f"Unknown model mode {model_mode}"
    assert kpt_mode in ['coco17', 'body25'], f"Unknown number of keypoints {kpt_mode}"

    model_cfg = dynamic_import(f'vitpose.configs.{kpt_mode}.ViTPose_{model_mode}_coco_256x192', 'model')
    data_cfg = dynamic_import(f'vitpose.configs.{kpt_mode}.ViTPose_{model_mode}_coco_256x192', 'data_cfg')

    vitpose_ckpt_path = osp.join(weights_dir, kpt_mode, f"vitpose-{model_mode}.pth")
    vitpose_onnx_path = vitpose_ckpt_path.replace('.pth','.onnx')

    vit_pose = ViTPose(model_cfg).to(device)

    assert osp.exists(vitpose_ckpt_path), f"ViTPose ckpt not found at {vitpose_ckpt_path}"

    C, H, W = (3, 256, 192)

    ckpt = torch.load(vitpose_ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        vit_pose.load_state_dict(ckpt['state_dict'])
    else:
        vit_pose.load_state_dict(ckpt)
    if device == 'cpu':
        half = False

    vit_pose.eval()

    input_names = ["input_0"]
    output_names = ["output_0"]

    inputs = torch.randn(1, C, H, W).to(device)
    if half:
        vit_pose = vit_pose.half()
        inputs = inputs.half()
    dynamic_axes = {'input_0' : {0 : 'batch_size'},
                    'output_0' : {0 : 'batch_size'}}

    torch_out = torch.onnx.export(vit_pose, inputs, Path(vitpose_onnx_path), export_params=True, verbose=True,
                                  input_names=input_names, output_names=output_names,
                                  opset_version=12, dynamic_axes = dynamic_axes)
    logger.success(f">>> Saved at: {vitpose_onnx_path}")



if __name__ == "__main__":
    export_onnx(device='cpu',
                weights_dir='',
                model_spec='body25-huge', half=False)

    # # from vitpose.configs.ViTPose_base_coco_256x192 import model as model_cfg
    # # from vitpose.configs.ViTPose_base_coco_256x192 import data_cfg
    # from vitpose.configs.ViTPose_huge_coco_256x192 import model as model_cfg
    # from vitpose.configs.ViTPose_huge_coco_256x192 import data_cfg
    #
    # img_paths = glob('/mnt/x/data_repos/volleyball/HierVolley/player_crops/*.jpg')
    # out_dir = '/mnt/x/data_repos/volleyball/HierVolley/player_crops_vitpose'
    #
    # CUR_DIR = "/mnt/x/data_repos/models/ViTPose"  # osp.dirname(__file__)
    # CKPT_PATH = f"{CUR_DIR}/vitpose-h-multi-coco.pth"
    #
    # img_size = data_cfg['image_size']
    #
    # kpt_model = vitpose_inference_model(input_size=img_size, model_cfg=model_cfg, ckpt_path=CKPT_PATH,
    #                                     device=torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'),
    #                                     out_dir=out_dir)
    #
    #
    # for img_path in img_paths:
    #     image = cv2.imread(img_path)
    #     # convert BGR to RGB
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    #     keypoints = kpt_model(img_path=image)
    #
    #     for pid, point in enumerate(keypoints):
    #         img = np.array(image)[:, :, ::-1]  # RGB to BGR for cv2 modules
    #         img = draw_points_and_skeleton(img.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
    #                                        points_color_palette='gist_rainbow', skeleton_color_palette='jet',
    #                                        points_palette_samples=10, confidence_threshold=0.4)
    #         cv2.imshow('result', img)
    #         cv2.waitKey(0)
    #         # os.makedirs(out_dir, exist_ok=True)
    #         # out_image_path = osp.join(out_dir,osp.basename(img_path))
    #         # cv2.imwrite(out_image_path, img)
    #         # logger.success(f"created {out_image_path}")
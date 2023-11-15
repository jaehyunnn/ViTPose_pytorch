import torch.nn as nn

from .backbone.vit import ViT
from .backbone.vit_moe import ViTMoE
from .head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
from .head.topdown_heatmap_base_head import TopdownHeatmapBaseHead


__all__ = ['ViTPose']


class ViTPose(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super(ViTPose, self).__init__()
        
        backbone_cfg = {k: v for k, v in cfg['backbone'].items() if k != 'type'}
        head_cfg = {k: v for k, v in cfg['keypoint_head'].items() if k != 'type'}
        if backbone_cfg['type'].lower() == "ViT".lower():
            self.backbone = ViT(**backbone_cfg)
        elif backbone_cfg['type'].lower() == "ViTMoE".lower():
            self.backbone = ViTMoE(**backbone_cfg)
        else:
            print(f"cfgs model.backbone.type shoulde be ViT or ViTMoE but got {backbone_cfg['type']}")
            raise
        if head_cfg['type'].lower() == 'TopdownHeatmapSimpleHead'.lower():
            self.keypoint_head = TopdownHeatmapSimpleHead(**head_cfg)
        elif head_cfg['type'].lower() == "TopdownHeatmapBaseHead".lower():
            self.keypoint_head = TopdownHeatmapBaseHead(**head_cfg)
        else :
            print(f"cfgs model.keypoint_head.type shoulde be TopdownHeatmapBaseHead or TopdownHeatmapSimpleHead but got {head_cfg['type']}")
            raise
    
    def forward_features(self, x):
        return self.backbone(x)
    
    def forward(self, x):
        return self.keypoint_head(self.backbone(x))

import importlib
from os import path as osp

import numpy as np
from importlib_resources import files
from omegaconf import DictConfig
from omegaconf import DictConfig, OmegaConf
def prepare_cfg(base_cfg_path=None, **kwargs) -> DictConfig:

    app_support_dir = get_support_dir()
    if base_cfg_path is None:
        base_cfg_path = osp.join(app_support_dir, "../conf/conf.yaml")

    assert osp.exists(base_cfg_path), f"conf file not found: {base_cfg_path}"
    base_cfg = OmegaConf.load(base_cfg_path)

    if not OmegaConf.has_resolver("get_support_dir"):
        OmegaConf.register_new_resolver('get_support_dir',lambda : get_support_dir())
    if not OmegaConf.has_resolver("upper"):
        OmegaConf.register_new_resolver("upper", lambda a: a.upper())

    override_cfg_dotlist = [f"{k}={v}" for k, v in kwargs.items()]
    override_cfg = OmegaConf.from_dotlist(override_cfg_dotlist)

    return OmegaConf.merge(base_cfg, override_cfg)

def create_list_chunks(list_, group_size, overlap_size, drop_smaller_batches=True):
    assert isinstance(list_, list) or isinstance(list_, range), f"first argument should be a list type: {type(list_)}"
    if drop_smaller_batches:
        return [list_[i:i + group_size] for i in range(0, len(list_), group_size - overlap_size) if
                len(list_[i:i + group_size]) == group_size]
    else:
        return [list_[i:i + group_size] for i in range(0, len(list_), group_size - overlap_size)]

def get_support_dir(module_name="vitpose"):
    support_data_dir = str(files(f"{module_name}.support_data")._paths[0])
    assert osp.exists(support_data_dir)
    return support_data_dir


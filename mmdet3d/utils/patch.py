import glob
import os
import os.path as osp
import shutil
import types

from mmcv.runner import BaseRunner, EpochBasedRunner, IterBasedRunner
from mmcv.utils import Config

import re
from typing import Union

import inspect


def parse_method_info(method):
    sig = inspect.signature(method)
    params = sig.parameters
    return params


pattern = re.compile("\$\{[a-zA-Z\d_.]*\}")

def get_value(cfg: dict, chained_key: str):
    keys = chained_key.split(".")
    if len(keys) == 1:
        return cfg[keys[0]]
    else:
        return get_value(cfg[keys[0]], ".".join(keys[1:]))


def resolve(cfg: Union[dict, list], base=None):
    if base is None:
        base = cfg
    if isinstance(cfg, dict):
        return {k: resolve(v, base) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [resolve(v, base) for v in cfg]
    elif isinstance(cfg, tuple):
        return tuple([resolve(v, base) for v in cfg])
    elif isinstance(cfg, str):
        # process
        var_names = pattern.findall(cfg)
        if len(var_names) == 1 and len(cfg) == len(var_names[0]):
            return get_value(base, var_names[0][2:-1])
        else:
            vars = [get_value(base, name[2:-1]) for name in var_names]
            for name, var in zip(var_names, vars):
                cfg = cfg.replace(name, str(var))
            return cfg
    else:
        return cfg



def find_latest_checkpoint(path, ext="pth"):
    if not osp.exists(path):
        return None
    if osp.exists(osp.join(path, f"latest.{ext}")):
        return osp.join(path, f"latest.{ext}")

    checkpoints = glob.glob(osp.join(path, f"*.{ext}"))
    if len(checkpoints) == 0:
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split("_")[-1].split(".")[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path


def patch_checkpoint(runner: BaseRunner):
    # patch save_checkpoint
    old_save_checkpoint = runner.save_checkpoint
    params = parse_method_info(old_save_checkpoint)
    default_tmpl = params["filename_tmpl"].default

    def save_checkpoint(self, out_dir, **kwargs):
        create_symlink = kwargs.get("create_symlink", True)
        filename_tmpl = kwargs.get("filename_tmpl", default_tmpl)
        # create_symlink
        kwargs.update(create_symlink=False)
        old_save_checkpoint(out_dir, **kwargs)
        if create_symlink:
            dst_file = osp.join(out_dir, "latest.pth")
            if isinstance(self, EpochBasedRunner):
                filename = filename_tmpl.format(self.epoch + 1)
            elif isinstance(self, IterBasedRunner):
                filename = filename_tmpl.format(self.iter + 1)
            else:
                raise NotImplementedError()
            filepath = osp.join(out_dir, filename)
            shutil.copy(filepath, dst_file)

    runner.save_checkpoint = types.MethodType(save_checkpoint, runner)
    return runner


def patch_runner(runner):
    runner = patch_checkpoint(runner)
    return runner


def setup_env(cfg):
    os.environ["WORK_DIR"] = cfg.work_dir


def patch_config(cfg):

    cfg_dict = super(Config, cfg).__getattribute__("_cfg_dict").to_dict()
    cfg_dict["cfg_name"] = osp.splitext(osp.basename(cfg.filename))[0]
    cfg_dict = resolve(cfg_dict)
    cfg = Config(cfg_dict, filename=cfg.filename)
    # wrap for semi
    if cfg.get("model_warpper", None) is not None:
        cfg.model = cfg.model_warpper
        cfg.pop("model_warpper")
    # enable environment variables
    # setup_env(cfg)
    return cfg

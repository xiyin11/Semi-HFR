import argparse
from yacs.config import CfgNode as CN
from .logger import setup_logger
from .file_io import PathManager
import os
import numpy as np
import random
import torch
from datetime import datetime
import logging

def argument_parser():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument("--config-file",'--c',default="config/Tufts_val.yml",help="path to config file")
    return parser



def get_config(yaml_path):
    yaml_file = open(yaml_path,'r',encoding='utf-8')
    config=CN.load_cfg(yaml_file)
    config.freeze()
    yaml_file.close()

    return config

def setup(cfg,args=''):

    output_dir=cfg.OUTPUT_PATH
    setup_logger(output=output_dir, name=__name__)
    logger = setup_logger(output_dir)
    logger.info("Command line arguments: " + str(args))
    if args:
        if hasattr(args, "config_file") and args.config_file != "":
            logger.info(
                "Contents of args.config_file={}:\n{}".format(
                    args.config_file, PathManager.open(args.config_file, "r").read()
                )
            )

    logger.info("Running with full config:\n{}".format(cfg))
    path = os.path.join(output_dir, "config.yaml")
    with PathManager.open(path, "w") as f:
        f.write(cfg.dump())
    logger.info("Full config saved to {}".format(os.path.abspath(path)))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng()

def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.
    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger('Train')
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)

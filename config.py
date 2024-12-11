import os
from datetime import datetime
import argparse
import yaml
from dotmap import DotMap


def get_config(sysv):
    parser = argparse.ArgumentParser(description='Training variables.')
    parser.add_argument('--local_rank', type=int, default=1, metavar='N', help='Local rank process')
    parser.add_argument('--config_file', help="path to yaml config file")
    parser.add_argument('--exp_name', default=datetime.now().strftime("%Y_%m_%d-%H%M%S"))
    parser.add_argument('--logroot', default='')
    args, _ = parser.parse_known_args(sysv)
    with open(args.config_file, 'r') as file:
        cfg = yaml.safe_load(file)
    cfg = DotMap(cfg)
    cfg.exp_name = "{}_{}_{}".format(args.exp_name, cfg.wandb.model_basename, cfg.dataset.dataset_name)
    cfg.local_rank = args.local_rank
    cfg.wandb.log_dir = os.path.join(args.logroot, cfg.wandb.log_dir) if args.logroot else cfg.wandb.log_dir
    with open(cfg.model.prompts, 'r') as file:
        cfg.prompts = yaml.safe_load(file)
        cfg.prompts = DotMap(cfg.prompts)
    return cfg

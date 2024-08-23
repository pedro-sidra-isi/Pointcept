"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
import pointcept.utils.comm as comm
import wandb
import os
from pathlib import Path
from pointcept.utils import ForkedPdb


def wandb_tracking_main_worker(cfg):
    # == Original code
    cfg = default_setup(cfg)
    # == 

    exp_path = Path(cfg.save_path)
    exp_path.mkdir(exist_ok=True, parents=True)
    exp_name = exp_path.name
    exp_project = exp_path.parent.name

    if comm.is_main_process():
        # Wandb tracking
        wandb.init(
            name=exp_name,
            project=exp_project,
            config=cfg,
            sync_tensorboard=True,
        )
        if (cfg.weight) and (not Path(cfg.weight).is_file()):
            cfg.weight = wandb.use_model(cfg.weight)
    
    try:
        # Original code
        trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
        trainer.train()
        # == 
    finally:
        if comm.is_main_process():
            # Wandb model save
            filename = exp_path / "model" / "model_last.pth"

            if filename.is_file():
                wandb.log_model(path=str(filename))


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()

    return trainer


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        wandb_tracking_main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()

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
import wandb
import os


def main_worker(cfg):
    cfg = default_setup(cfg)

    # Wandb tracking
    wandb.init(project="msc", config=cfg)

    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


    filename = os.path.join(
        trainer.cfg.save_path, "model", "model_last.pth"
    )
    wandb.log_model(path=filename, name="model_last.pth")


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()

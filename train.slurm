#!/bin/bash
#SBATCH --job-name=pcept_train
#SBATCH --partition=tupi
#SBATCH --exclude=tupi[1-2]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x/%j_%N.out
#SBATCH --error=slurm/%x/%j_%N.err

DIST_URL='auto'
DATASET=scannet
CONFIG=pretrain-sculpting_sched-spunet2
EXP_NAME="wacv/sculpting_scheduling"
WEIGHT="None"
MACHINES=1
GPUS=1

srun mkdir -p $SCRATCH/data && rsync -azP data/scannet $SCRATCH/data
srun docker compose run -e DIST_URL=$DIST_URL --rm --remove-orphans dev \
    bash scripts/train.sh -g $GPUS -d $DATASET  -m $MACHINES -c $CONFIG -n $EXP_NAME -w $WEIGHT

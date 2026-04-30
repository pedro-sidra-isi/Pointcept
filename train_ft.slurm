#!/bin/bash
#SBATCH --job-name=pcept_train
#SBATCH --partition=tupi
#SBATCH --exclude=tupi[1-2]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x/%j_%N.out
#SBATCH --error=slurm/%x/%j_%N.err

# mkdir -p $SCRATCH/exp
DIST_URL='auto'
DATASET=scannet
CONFIG=pipeline-trimming-spunet
EXP_NAME="wacv/pretrain_trimming_spunet_less"
WEIGHT="None"
MACHINES=1
GPUS=1
srun mkdir -p $SCRATCH/data && rsync -azP data/scannet $SCRATCH/data
srun docker compose run \
    --rm --remove-orphans \
    -v $HOME/Pointcept:/workspaces/Pointcept -v $HOME/.netrc:/root/.netrc \
    -v $SCRATCH/exp:/workspaces/Pointcept/exp -v $SCRATCH/data/scannet:/workspaces/Pointcept/data/scannet \
    -e GLOO_SOCKET_IFNAME=enp4s0 -e NCCL_SOCKET_IFNAME=enp4s0 \
    -e SLURM_NODELIST -e SLURM_NODEID  -e DIST_URL=$DIST_URL  -e WORLD_SIZE  -e MASTER_PORT  -e MASTER_ADDR \
    dev \
    bash scripts/train.sh -g $GPUS -d $DATASET  -m $MACHINES -c $CONFIG -n $EXP_NAME -w $WEIGHT

# CONFIG=semseg-spunet-sidra-efficient-lr10
# WEIGHT="exp/scannet/$EXP_NAME/model/model_last.pth"
# srun docker compose run \
#     --rm --remove-orphans \
#     -v $HOME/Pointcept:/workspaces/Pointcept -v $HOME/.netrc:/root/.netrc \
#     -v $SCRATCH/exp:/workspaces/Pointcept/exp -v $SCRATCH/data/scannet:/workspaces/Pointcept/data/scannet \
#     -e GLOO_SOCKET_IFNAME=enp4s0 -e NCCL_SOCKET_IFNAME=enp4s0 \
#     -e SLURM_NODELIST -e SLURM_NODEID  -e DIST_URL=$DIST_URL  -e WORLD_SIZE  -e MASTER_PORT  -e MASTER_ADDR \
#     dev \
#     bash scripts/train.sh -g $GPUS -d $DATASET  -m $MACHINES -c $CONFIG -n $EXP_NAME -w $WEIGHT
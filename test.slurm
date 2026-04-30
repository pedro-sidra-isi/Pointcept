#!/bin/bash
#SBATCH --job-name=test_all_models
#SBATCH --partition=tupi
#SBATCH --exclude=tupi[1-2]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x/%j_%N.out
#SBATCH --error=slurm/%x/%j_%N.err

# mkdir -p $SCRATCH/exp
DIST_URL='auto'
DATASET=scannet
CONFIG=semseg-spunet-sidra-efficient-lr10
EXP_NAME_BASE="exp/wacv_tests/"
MACHINES=1
GPUS=1

# Done:
# --
srun mkdir -p $SCRATCH/data && rsync -azP data/scannet $SCRATCH/data

for model in test_models/*.pth
do
	WEIGHT=$model
	EXP_NAME=$(basename -- $model)
	srun docker compose run --rm --remove-orphans dev \
	    python tools/test.py \
	    --config-file "configs/scannet/semseg-spunet-sidra-efficient-lr10.py" \
	    --num-gpus "1" \
	    --options save_path="$EXPNAME_BASE/$EXP_NAME" weight=$WEIGHT
done


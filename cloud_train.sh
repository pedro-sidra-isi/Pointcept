set -e

cd /workspace/Pointcept
# bash setup_datasets.sh
bash scripts/train.sh -g 2 -d scannet -c pipeline-sculpting-spunet -n debug/slurm


DIST_URL='auto'
DATASET=scannet
CONFIG=semseg-spunet-sidra-efficient-lr10
EXP_NAME_BASE="exp/wacv_tests/"
MACHINES=1
GPUS=1

for model in test_models/FT_msc*.pth
do
	WEIGHT=$model
	EXP_NAME=$(basename -- $model)
	echo srun docker compose run --rm --remove-orphans dev \
	    python tools/test.py \
	    --config-file "/workspace/Pointcept/configs/scannet/semseg-spunet-sidra-efficient-lr10.py" \
	    --num-gpus "1" \
	    --options save_path="$EXP_NAME_BASE$EXP_NAME" weight=$WEIGHT
done


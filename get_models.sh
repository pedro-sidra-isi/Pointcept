for m in $SCRATCH/exp/scannet/wacv/*/model/model_last.pth; do echo $m | sed -E -e 's/\/scratch\/psdfreitas\/\/exp\/scannet\/wacv\/(.*)\/model\/model_last/\1/' | xargs -I {} cp $m {}  ; done

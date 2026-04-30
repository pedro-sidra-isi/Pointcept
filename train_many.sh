python -m pip install -r requirements.txt
bash scripts/train.sh -g 1 -c pipeline-trimming-spunet -n debug/trimmingv0
bash scripts/train.sh -g 1 -c semseg-spunet-sidra-efficient-lr1 -n debug/trimmingv0_FT -w $PWD/exp/scannet/debug/trimmingv0/model/model_last.pth
# bash scripts/train.sh -g 1 -c semseg-ptv3-sidra-efficient-lr1 -n icip/LOCAL_REPRO_ptv3_lr1 
# bash scripts/train.sh -g 1 -c semseg-ptv3-sidra-efficient-lr1 -n icip/LOCAL_REPRO_ptv3_lr1_sculpt -w '/workspaces/Pointcept/ptv3_sculpting.pth'
# bash scripts/train.sh -g 1 -c semseg-ptv3-sidra-efficient-lr1 -n icip/LOCAL_REPRO_ptv3_lr1 
# bash scripts/train.sh -g 2 -c semseg-ptv3-sidra-efficient-lr5 -o -n icip/ptv3_lr5_sculpt -w /root/Pointcept/artifacts/ptv3_sculpt:v0/ptv3_sculpting.pth
# bash scripts/train.sh -g 2 -c semseg-ptv3-sidra-efficient-lr5 -o -n icip/ptv3_lr5 
# bash scripts/train.sh -g 2 -c semseg-ptv3-sidra-efficient-lr10 -o -n icip/ptv3_lr10_sculpt -w /root/Pointcept/artifacts/ptv3_sculpt:v0/ptv3_sculpting.pth
# bash scripts/train.sh -g 2 -c semseg-ptv3-sidra-efficient-lr10 -o -n icip/ptv3_lr10
# bash scripts/train.sh -g 2 -c semseg-ptv3-sidra-efficient-lr20 -o -n icip/ptv3_lr20_sculpt -w /root/Pointcept/artifacts/ptv3_sculpt:v0/ptv3_sculpting.pth
#ptv3 bash scripts/train.sh -g 2 -c semseg-ptv3-sidra-efficient-lr20 -o -n icip/ptv3_lr20
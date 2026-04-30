python run.py "sh scripts/train.sh -g 2 -d scannet -c semseg-spunet-v1m1-2-efficient-lr10 -n msc/LR10FT_Adam_RepoMSC2 -w pedrosidra/msc/pretrain_repo:v0"
python run.py "sh scripts/train.sh -g 2 -d scannet -c semseg-spunet-v1m1-2-efficient-lr10 -n msc/LR10FT_Adam_MyMSC2 -w pedrosidra/msc/pretrain_mine:v0"
python run.py "sh scripts/train.sh -g 2 -d scannet -c semseg-spunet-v1m1-2-efficient-lr10 -n msc/LR10FT_Adam_Sculpting2 -w pedrosidra/sculpting/run-3uk103f5-model_last.pth:v0"
python run.py "sh scripts/train.sh -g 2 -d scannet -c semseg-spunet-v1m1-2-efficient-lr10 -n msc/LR10FT_Adam_Scratch2"
set -e

pip install -U "huggingface_hub[cli]" wandb
mkdir -p /workspace/Pointcept/data/
cd /workspace/Pointcept/data/
wandb artifact get --type dataset --root ./scannet pedrosidra/dataset/scannet:v0&
huggingface-cli download --repo-type dataset  --local-dir . --  Pointcept/arkitscenes-compressed
for i in 1 2 3 4 5 6 7 8; do tar -xzvf arkitscenes_$i.tar.gz --no-same-owner & done
wait
mv Training/ scannet/arkit/
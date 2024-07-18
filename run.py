import clearml
import clearml.datasets
import subprocess
from pathlib import Path
import os

task : clearml.Task = clearml.Task.init(
    project_name="mestre_pedro",
    task_name="msc-train"
)
task.set_base_docker(docker_image="pedrosidra0/pointcept:v0")
task.execute_remotely()

data_path = clearml.datasets.Dataset.get(dataset_id="a38dbe849a49415f99196916d4f947c5").get_local_copy()

dataset_path = "./data/scannet"
if not Path(dataset_path).exists():
    print(f"Symlinking {data_path} to {dataset_path}")
    os.symlink(src=data_path, dst=dataset_path)

subprocess.run("sh scripts/train.sh -g 2 -d scannet -c pretrain-msc-v1m1-0-spunet-base -n pretrain-msc-v1m1-0-spunet-base")
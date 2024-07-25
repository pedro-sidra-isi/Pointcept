import clearml
import requests
import clearml.datasets
import subprocess
from pathlib import Path
import os

task: clearml.Task = clearml.Task.init(
    project_name="mestre_pedro", task_name="msc-train"
)
task.set_base_docker(
    docker_image="pedrosidra0/pointcept:v0",
    # TODO: fix this to be more generic than this workaround for credentials
    docker_arguments="--shm-size=64000mb -e MKL_SERVICE_FORCE_INTEL=1 -v /home/freitas/.netrc:/root/.netrc",
)
task.execute_remotely(queue_name="default")

data_path = clearml.datasets.Dataset.get(
    dataset_id="a38dbe849a49415f99196916d4f947c5"
).get_local_copy()

dataset_path = Path("./data/scannet")
if not dataset_path.exists():
    print(f"Symlinking {data_path} to {dataset_path}")
    dataset_path.parent.mkdir(exist_ok=True, parents=True)
    os.symlink(src=data_path, dst=dataset_path)

pretrain_link = "http://10.167.1.54:8081/mestre_pedro/msc-train.3ae6e83760764352bf3f2a62a51f138b/artifacts/model/model_last.pth"
pretrain_path = "pretrain_model.pth"

r = requests.get(pretrain_link)
open(pretrain_path , 'wb').write(r.content)

# subprocess.run(
#     "sh scripts/train.sh -g 2 -d scannet -c pretrain-msc-v1m1-0-spunet-base -n pretrain-msc-v1m1-0-spunet-base".split()
# )
subprocess.run(
    f"sh scripts/train.sh -g 1 -d scannet -w {pretrain_path} -c semseg-spunet-v1m1-4-ft -n semseg-msc-v1m1-0f-spunet-base".split()
)

model_path = next(Path("exp").glob("**/model/*.pth"))
task.upload_artifact(name="model", artifact_object=str(model_path))

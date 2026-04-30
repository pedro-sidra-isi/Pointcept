#%%
import pandas as pd
import wandb
from pathlib import Path
import shutil
import os
# %%
df = pd.read_csv("/workspaces/Pointcept/data/wandb_export_2025-07-20T14_56_06.735-03_00.csv")
#%%
to_eval = Path(f"./models_to_eval/{os.getenv('SLURM_NODEID')}")
to_eval.mkdir(parents=True, exist_ok=True)
# %%
root = "pedrosidra/wacv"
runs=wandb.Api().runs(path=root, filters={"config.model.type":"DefaultSegmentor"})
models = []
for run in runs:
    model=None
    artifacts = run.logged_artifacts()
    data = [(a, a.created_at) for a in artifacts if a.type=="model"]
    if len(data)>0:
        last_artifact = max(data, key=lambda a:a[1])
        p = Path(last_artifact[0].download())
    else:
        p = Path(f"/scratch/exp/scannet/wacv/{run.name}/model/")
    local= list(p.glob("*last.pth"))
    if local:
        model=local[0]

    if model:
        print(run.name, end="-->")
        print(model)
        shutil.copy(model, to_eval/run.name)
        models.append(model)
    else:
        print(f"no model found at {p}")

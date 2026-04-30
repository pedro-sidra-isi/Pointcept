_base_ = ["../_base_/default_runtime.py"]

# wandb_off = 1
enable_wandb = False

# No precise evaluator because it breaks sculpting
hooks = [
    dict(type="CheckpointLoaderAllowMismatch"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    # dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaverWandb", save_freq=5),
    # dict(type="PreciseEvaluator", test_last=False),
]

# Sculpting params
sculpting_transform = dict(
    type="SculptingOcclude",
    cube_size_min=0.2,
    cube_size_max=0.3,
    npoint_frac=None,
    npoints=1,
    cell_size=0.02,
    density_factor=1.0,
    kill_color_proba=0.0,
    sampling="dense random",
)
update_index_keys = dict(
    type="Update",
    keys_dict={
        "index_valid_keys": [
            "coord",
            "grid_coord",
            "color",
            "normal",
            "superpoint",
            "strength",
            "segment",
            "instance",
        ]
    },
)


voxelize_transform = dict(
    type="VoxelizeAgg",
    grid_size=0.02,
    hash_type="fnv",
    mode="train",
    return_grid_coord=True,
    how_to_agg_feats=dict(
        coord="mean",
        color="mean",
        segment="rand_choice",
        normal="first",
        instance="first",
    ),
)

# test = dict(type="SemSegPredictor", verbose=True)

tta_identity = [
    [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
]

sculpting_data_base_configs = dict(
    num_classes=2,
    ignore_index=-1,
    names=[
        "occluded",
        "original",
    ],
)

# FT_config = "configs/scannet/semseg-spunet-sidra-efficient-lr100.py"

## ===== MODEL DEFINITION

# misc custom setting
batch_size = 6  # bs: total bs in all gpus
num_worker = 6
mix_prob = 0
clip_grad = 3.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
evaluate = False
find_unused_parameters = False


model = dict(
    type="DefaultSegmentorV2",
    num_classes=2,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m3",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        enc_mode=True,
        mask_token=True,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 200
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# dataset settings
dataset_type = "ScanNetDataset"
data_root = "data/scannet"

data = dict(
    **sculpting_data_base_configs,
    train=dict(
        type=dataset_type,
        split=[
            "train",
            "val",
            "test",
            # "arkit",
        ],
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            # dict(
            #    type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            # ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=1.0),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.2),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.2),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(type="SphereCrop", point_max=150000, mode="random"),
            update_index_keys,
            sculpting_transform,
            voxelize_transform,
            dict(type="SphereCrop", point_max=120000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color",),
            ),
        ],
        test_mode=False,
    ),
)

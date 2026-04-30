_base_ = ["../_base_/default_runtime.py"]

# wandb_off = 1

# No precise evaluator because it breaks sculpting
hooks = [
    dict(type="CheckpointLoaderAllowMismatch"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaverWandb", save_freq=None),
    # dict(type="PreciseEvaluator", test_last=False),
]

# Sculpting params
sculpting_transform = dict(
    type="SculptingOcclude",
    cube_size_min=0.1,
    cube_size_max=0.5,
    npoint_frac=0.002,
    npoints=None,
    cell_size=0.02,
    density_factor=0.25,
    kill_color_proba=0.5,
    sampling="dense random",
)

# Original VoxelizeAgg - erro dimensional
# voxelize_transform = dict(
#     type="VoxelizeAgg",
#     grid_size=0.02,
#     hash_type="fnv",
#     mode="train",
#     return_grid_coord=True,
#     how_to_agg_feats=dict(
#         coord="mean",
#         color="mean",
#         normal="mean",
#         segment="mode",
#         instance="rand_choice",
#     ),
# )

# GridSample compativelm OACNN:
voxelize_transform = dict(
    type="GridSample",
    grid_size=0.02,
    hash_type="fnv",
    mode="train",
    return_grid_coord=True,
)

test = dict(type="SemSegPredictor", verbose=True)

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

FT_config = "configs/scannet/semseg-spunet-sidra-efficient-lr10.py"

## ===== MODEL DEFINITION

# misc custom setting
batch_size = 8  # reduzido 
num_worker = 64  # reduzido 
mix_prob = 0.8
empty_cache = True  # Liberar cache pra OACNN
enable_amp = True

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="OACNNs", # /workspaces/Pointcept/Pointcept/configs/scannetpp/semseg-oacnn-v1m1-0-base.
        in_channels=3,
        num_classes=2,
        embed_channels=64,
        enc_channels=[64, 64, 128, 256],
        groups=[4, 4, 8, 16],
        enc_depth=[3, 3, 9, 8],
        dec_channels=[256, 256, 256, 256],
        point_grid_size=[[8, 12, 16, 16], [6, 9, 12, 12], [4, 6, 8, 8], [3, 4, 6, 6]],
        dec_depth=[2, 2, 2, 2],
        enc_num_ref=[16, 16, 16, 16],
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-100)],
)


# scheduler settings
epoch = 800
optimizer = dict(type="SGD", lr=0.05, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=10000.0,
)

# dataset settings
dataset_type = "ScanNetDataset"
data_root = "data/scannet"

data = dict(
    **sculpting_data_base_configs,
    train=dict(
        type=dataset_type,
        split=["train", "val", "test", "arkit"],
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            sculpting_transform,
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            voxelize_transform,
            dict(type="SphereCrop", point_max=80000, mode="random"),  # Reduzido pra OACNN
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color",),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        lr_file="data/scannet/tasks/scenes/10.txt",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            sculpting_transform,
            voxelize_transform,
            # dict(type="SphereCrop", point_max=1000000, mode="center"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color",),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        lr_file="data/scannet/tasks/scenes/10.txt",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            sculpting_transform,
            voxelize_transform,
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "name",
                    "coord",
                    "grid_coord",
                    "segment",
                ),
                feat_keys=("color",),
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            fragment=False,
            voxelize=voxelize_transform,
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=(
                        "name",
                        "coord",
                        "grid_coord",
                        "segment",
                    ),
                    feat_keys=("color",),
                ),
            ],
            aug_transform=tta_identity,
            # [
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[0],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         )
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[1 / 2],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         )
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[1],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         )
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[3 / 2],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         )
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[0],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         ),
            #         dict(type="RandomScale", scale=[0.95, 0.95]),
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[1 / 2],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         ),
            #         dict(type="RandomScale", scale=[0.95, 0.95]),
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[1],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         ),
            #         dict(type="RandomScale", scale=[0.95, 0.95]),
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[3 / 2],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         ),
            #         dict(type="RandomScale", scale=[0.95, 0.95]),
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[0],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         ),
            #         dict(type="RandomScale", scale=[1.05, 1.05]),
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[1 / 2],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         ),
            #         dict(type="RandomScale", scale=[1.05, 1.05]),
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[1],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         ),
            #         dict(type="RandomScale", scale=[1.05, 1.05]),
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[3 / 2],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         ),
            #         dict(type="RandomScale", scale=[1.05, 1.05]),
            #     ],
            #     [dict(type="RandomFlip", p=1)],
            # ],
        ),
    ),
)

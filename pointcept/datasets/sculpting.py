"""
3D Point Cloud Sculpting

Author: Pedro Sidra (pedrosidra0@gmail.com)
"""
import torch
import time
from scipy.stats import zscore
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import pandas as pd

import random
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
import copy
from collections.abc import Sequence, Mapping

from pointcept.utils.registry import Registry

TRANSFORMS = Registry("transforms")


@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


def add_random_cubes(self, xyz, rgb, semantic_label, instance_label):
    cube_size_min = 0.1
    cube_size_max = 0.5
    np.random.seed(0)
    cubes = get_random_cubes_random_sampled_point_references(
        cube_size_min,
        cube_size_max,
        xyz,
        npoints=int(0.005 * len(xyz)),
        cell_size=0.02,
        actual_cube=False,
        sphere=False,
        point_sampling="random",
        density_factor=0.1,
    )

    xyz = np.vstack([xyz, cubes])

    rand_colors = 2 * np.random.rand(*cubes.shape) - 1
    rgb = np.vstack([rgb, rand_colors])

    # Randomly turn colors off
    if np.random.rand() < 0.5:
        rgb = rgb * 0.0

    semantic_label = np.hstack(
        [np.ones_like(semantic_label), np.zeros(cubes.shape[0])]
    )

    instance_label = np.hstack(
        [-1 * np.ones(instance_label.shape[0]), -1 * np.ones(cubes.shape[0])]
    )

    return xyz, rgb, semantic_label, instance_label

def normalize(arr):
    return arr / np.linalg.norm(arr)


def get_random_cube(
    cube_size_min=np.array([0.1, 0.1, 0.1]),
    cube_size_max=np.array([0.5, 0.5, 0.5]),
    cell_size=0.01,
    actual_cube=False,
    sphere=None,
    point_sampling="dense",
    density_factor=1.0,
):
    # random rotation in z
    rotation = R.from_euler("z", np.random.rand() * np.pi, degrees=False).as_matrix()

    # uniform random between size_min and size_max
    cube_size = cube_size_min + np.random.rand(3) * (cube_size_max - cube_size_min)
    if actual_cube:
        cube_size[0] = cube_size[1]
        cube_size[2] = cube_size[1]

    if "dense" in point_sampling:
        points_x = np.arange(0, cube_size[0], cell_size)
        points_y = np.arange(0, cube_size[1], cell_size)
        points_z = np.arange(0, cube_size[2], cell_size)
        if "random" in point_sampling:
            # sample from each dimension to total dimension factor (cube root)
            choice_factor = density_factor ** (1 / 3)
            points_x = np.random.choice(
                points_x, int(choice_factor * len(points_x)), replace=False
            )
            points_y = np.random.choice(
                points_y, int(choice_factor * len(points_y)), replace=False
            )
            points_z = np.random.choice(
                points_z, int(choice_factor * len(points_z)), replace=False
            )
        x, y, z = np.meshgrid(points_x, points_y, points_z)
    if point_sampling == "random":
        npoints = (density_factor * (cube_size / cell_size).prod()).astype(int)
        x = np.random.rand(npoints) * cube_size[0]
        y = np.random.rand(npoints) * cube_size[1]
        z = np.random.rand(npoints) * cube_size[2]

    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    z = z - z.mean()

    cube = np.stack([x, y, z])
    if sphere:
        cube = cube[:, (x**2 + y**2 + z**2) < cube_size.min() ** 2]

    cube = rotation @ cube

    return cube.T


def get_random_cube_random_point_reference(points, *args, **kwargs):
    point = np.random.randint(0, len(points))

    point = points[point]
    return get_random_cube(*args, **kwargs) + point.reshape((1, 3))


def get_random_cube_average_heighted_point_reference(points, *args, **kwargs):
    z_coordinate_zscores = zscore(points[2, :])

    # high absolute value of zscore -> low proba
    sample_probas = normalize(1 / (0.001 + np.abs(z_coordinate_zscores)))
    point = np.random.choice(np.arange(len(points)), p=sample_probas)

    point = points[point]
    return get_random_cube(*args, **kwargs) + point.reshape((1, 3))


def get_random_cubes_random_sampled_point_references(
    cube_size_min, cube_size_max, points, npoints=10, rand_aspect=False, *args, **kwargs
):
    idxs = np.random.randint(0, len(points), size=npoints)

    cubes = []

    for point in points[idxs]:
        # aspect = np.random.randint(0, 3)
        _cube_size_min = np.ones(3) * cube_size_min
        _cube_size_max = np.ones(3) * cube_size_max
        # _cube_size_min[aspect] = cube_size_max / 5
        # _cube_size_max[aspect] = cube_size_max

        cubes.append(
            get_random_cube(_cube_size_min, _cube_size_max, *args, **kwargs)
            + point.reshape((1, 3))
        )

    return np.vstack(cubes)


if __name__ == "__main__":
    f = Path("./dataset/scannetv2/train/scene0000_00_inst_nostuff.pth")
    xyz, rgb, dummy_sem_label, dummy_inst_label = torch.load(f)

    pc = pd.DataFrame(
        dict(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
        )
    )

    cube_size_min = 0.1
    cube_size_max = 0.5
    cubes = get_random_cubes_random_sampled_point_references(
        cube_size_min,
        cube_size_max,
        pc[["x", "y", "z"]].to_numpy(),
        npoints=int(0.005 * len(pc)),
        cell_size=0.02,
        actual_cube=False,
        sphere=True,
        point_sampling="random",
        density_factor=0.1,
    )
    cubes = pd.DataFrame(
        cubes,
        columns=["x", "y", "z"],
    )
    cubes["label"] = 0
    pc["label"] = 1
    # cubes = []
    # for i in range(5):
    #     cubes.append(
    #         pd.DataFrame(
    #             get_random_cube_random_point_reference(
    #                 pc[["x", "y", "z"]].to_numpy(),
    #                 cell_size=0.01,
    #                 actual_cube=True,
    #                 cube_size_min=np.array([0.1, 0.1, 0.1]),
    #                 cube_size_max=np.array([1.0, 1.0, 1.0]),
    #             ),
    #             columns=["x", "y", "z"],
    #         )
    #     )
    pc = pd.concat([pc, cubes])

    pc.to_csv("output.txt")
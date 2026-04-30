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
from copy import deepcopy

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


def add_random_cubes(xyz, rgb, semantic_label, instance_label, normal=None):
    cube_size_min = 0.1
    cube_size_max = 0.5
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

    if normal is not None:
        rand_normals = np.random.rand(*cubes.shape)
        normal = np.vstack([normal, rand_normals])

    # Randomly turn colors off
    if np.random.rand() < 0.5:
        rgb = rgb * 0.0

    semantic_label = np.hstack(
        [np.ones_like(semantic_label), np.zeros(cubes.shape[0])]
    ).astype(np.int32)

    instance_label = np.hstack(
        [-1 * np.ones(instance_label.shape[0]), -1 * np.ones(cubes.shape[0])]
    ).astype(np.int32)

    return xyz, rgb, semantic_label, instance_label, normal


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
    rand_rotate=True,
):
    # random rotation in z
    if rand_rotate:
        rotation = R.from_euler(
            "z", np.random.rand() * np.pi, degrees=False
        ).as_matrix()
    else:
        rotation = R.from_euler("z", 0, degrees=False).as_matrix()

    # uniform random between size_min and size_max
    cube_size = cube_size_min + np.random.rand(3) * (cube_size_max - cube_size_min)
    if actual_cube:
        cube_size[0] = cube_size[1]
        cube_size[2] = cube_size[1]

    if "dense" in point_sampling:
        points_x = np.arange(0, cube_size[0], cell_size)
        points_y = np.arange(0, cube_size[1], cell_size)
        points_z = np.arange(0, cube_size[2], cell_size)
        x, y, z = np.meshgrid(points_x, points_y, points_z)
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

        if "random" in point_sampling:
            choices = np.random.choice(
                np.arange(len(x)),
                int(density_factor * len(x)),
            )

            x = x[choices]
            y = y[choices]
            z = z[choices]

    if point_sampling == "random":
        npoints = (density_factor * (cube_size / cell_size).prod()).astype(int)
        x = np.random.rand(npoints) * cube_size[0]
        y = np.random.rand(npoints) * cube_size[1]
        z = np.random.rand(npoints) * cube_size[2]
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

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
    cube_size_min, cube_size_max, points, npoints=10, return_idxs=False, *args, **kwargs
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


def get_random_colored_cubes_on_pts(
    cube_size_min, cube_size_max, points, feats, npoints=10, *args, **kwargs
):
    idxs = np.random.randint(0, len(points), size=npoints)

    cubes = []
    cube_feats = []

    for idx in idxs:
        # aspect = np.random.randint(0, 3)
        _cube_size_min = np.ones(3) * cube_size_min
        _cube_size_max = np.ones(3) * cube_size_max
        # _cube_size_min[aspect] = cube_size_max / 5
        # _cube_size_max[aspect] = cube_size_max
        point = points[idx]
        f = feats[idx]

        cube = get_random_cube(
            _cube_size_min, _cube_size_max, *args, **kwargs
        ) + point.reshape((1, 3))

        feat = np.ones_like(cube) * f

        cubes.append(cube)
        cube_feats.append(feat)

    return np.vstack(cubes), np.vstack(cube_feats)


def array_choice(ndarray, choices, axis=0):
    ndarray = np.asarray(ndarray)
    ndim = ndarray.ndim

    if ndim == 1:
        return ndarray[choices]

    choice_idxs = np.arange(0, ndarray.shape[axis - 1])
    locs = (*(np.s_[:] for _ in range(axis - 1)), choice_idxs, choices)

    return ndarray[locs]


def array_rand_choice(ndarray, axis=0):
    ndarray = np.asarray(ndarray)
    ndim = ndarray.ndim

    if ndim == 1:
        return np.random.choice(ndarray)

    choices = np.random.randint(0, ndarray.shape[axis], ndarray.shape[axis - 1])
    return array_choice(ndarray, choices, axis=axis)


def array_mode(ndarray, axis=0, return_details=False):
    # Check inputs
    ndarray = np.asarray(ndarray)
    ndim = ndarray.ndim
    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception("Cannot compute mode on empty array")
    try:
        axis = range(ndarray.ndim)[axis]
    except:
        raise Exception(
            'Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim)
        )

    # mode of 1 or 2 elements is whatever
    if ndarray.shape[axis] == 1 or ndarray.shape[axis] == 2:
        return np.take(ndarray, 0, axis=axis)

    # If array is 1-D and np version is > 1.9 np.unique will suffice
    if all(
        [
            ndim == 1,
            int(np.__version__.split(".")[0]) >= 1,
            int(np.__version__.split(".")[1]) >= 9,
        ]
    ):
        modals, counts = np.unique(ndarray, return_counts=True)
        index = np.argmax(counts)
        return modals[index], counts[index]

    # Sort array
    sort_idx = np.argsort(ndarray, axis=axis)
    sort = np.take_along_axis(ndarray, sort_idx, axis=axis)
    inverse_sort_idx = np.take_along_axis(
        np.indices(ndarray.shape)[axis], sort_idx, axis=axis
    )
    # Create array to transpose along the axis and get padding shape
    transpose = np.roll(np.arange(ndim)[::-1], axis)
    shape = list(sort.shape)
    shape[axis] = 1
    # Create a boolean array along strides of unique values
    strides = (
        np.concatenate(
            [
                np.zeros(shape=shape, dtype="bool"),
                np.diff(sort, axis=axis) == 0,
                np.zeros(shape=shape, dtype="bool"),
            ],
            axis=axis,
        )
        .transpose(transpose)
        .ravel()
    )
    # Count the stride lengths
    counts = np.cumsum(strides)
    counts[~strides] = np.concatenate([[0], np.diff(counts[~strides])])
    counts[strides] = 0
    # Get shape of padded counts and slice to return to the original shape
    shape = np.array(sort.shape)
    shape[axis] += 1
    shape = shape[transpose]
    slices = [slice(None)] * ndim
    slices[axis] = slice(1, None)
    # Reshape and compute final counts
    counts = counts.reshape(shape).transpose(transpose)[tuple(slices)] + 1

    # Find maximum counts and return modals/counts
    slices = [slice(None, i) for i in sort.shape]
    del slices[axis]
    index = np.ogrid[slices]
    index.insert(axis, np.argmax(counts, axis=axis))

    reverse_index = deepcopy(index)
    reverse_index[axis] = inverse_sort_idx[tuple(index)]

    index = tuple(index)
    reverse_index = tuple(reverse_index)
    if return_details:
        return ndarray[reverse_index], counts[index], reverse_index
    else:
        return sort[index]


def get_pointgrid(ncells=[10, 10, 10]):

    if isinstance(ncells, int):
        ncells = [ncells, ncells, ncells]

    points_x = np.arange(0, ncells[0])
    points_y = np.arange(0, ncells[1])
    points_z = np.arange(0, ncells[2])

    x, y, z = np.meshgrid(points_x, points_y, points_z)

    return np.stack(
        [
            x.flatten(),
            y.flatten(),
            z.flatten(),
        ]
    ).T


def get_cube(
    cube_size,
    cell_size=0.01,
    point_sampling="dense",
    density_factor=1.0,
):
    if "dense" in point_sampling:
        points_x = np.arange(0, cube_size, cell_size)
        points_y = np.arange(0, cube_size, cell_size)
        points_z = np.arange(0, cube_size, cell_size)
        x, y, z = np.meshgrid(points_x, points_y, points_z)
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

        if "random" in point_sampling:
            choices = np.random.choice(
                np.arange(len(x)),
                int(density_factor * len(x)),
            )

            x = x[choices]
            y = y[choices]
            z = z[choices]

    if point_sampling == "random":
        npoints = (density_factor * (cube_size / cell_size).prod()).astype(int)
        x = np.random.rand(npoints) * cube_size[0]
        y = np.random.rand(npoints) * cube_size[1]
        z = np.random.rand(npoints) * cube_size[2]
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

    x = x - x.mean()
    y = y - y.mean()
    z = z - z.mean()

    cube = np.stack([x, y, z])

    return cube.T


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
    pc = pd.concat([pc, cubes])

    pc.to_csv("output.txt")

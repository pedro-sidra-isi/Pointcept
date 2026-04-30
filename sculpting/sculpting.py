import numpy as np
import torch
from .sculpting_ops import (
    add_random_cubes,
    get_random_cubes_random_sampled_point_references,
    get_random_colored_cubes_on_pts,
    array_mode,
    array_rand_choice,
    array_choice,
    get_pointgrid,
)
from copy import deepcopy

import pointcept.datasets.transform as transform
from pointcept.utils.registry import Registry
from pointcept.datasets.transform import TRANSFORMS

# TRANSFORMS = Registry("transforms")


@TRANSFORMS.register_module()
class VoxelizeAgg(object):

    # agg_funcs = dict(
    #     mean=np.mean,
    #     mode=array_mode,
    #     max=np.max,
    #     min=np.min,
    #     rand_choice=array_rand_choice,
    #     first=lambda x, axis: array_choice(x, 0, axis=axis),
    # )

    def __init__(
        self,
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        how_to_agg_feats=dict(
            coord="mean",
            color="mean",
            normal="mean",
            segment="mode",
        ),
    ):
        self.grid_size = grid_size
        self.hash = (
            transform.GridSample.fnv_hash_vec
            if hash_type == "fnv"
            else transform.GridSample.ravel_hash_vec
        )
        assert mode in ["train", "test"]
        self.mode = mode

        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord

        self.how_to_agg_feats = how_to_agg_feats
        self.agg_func_names = deepcopy(how_to_agg_feats)

        # for key, agg_func_name in self.how_to_agg_feats.items():
        #     self.how_to_agg_feats[key] = VoxelizeAgg.agg_funcs[agg_func_name]

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()

        # To voxel indexes
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord

        # Save the min coord in original values
        min_coord = min_coord * np.array(self.grid_size)

        # Hash of the grid coords -> to group the unique voxel coords
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]

        # unique values of the key
        # inverse: mapping from points to voxels (p2v_map)
        # count: points per voxel
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

        # mapping from voxels to a single point (v2p_map)
        first_point_idx = idx_sort[np.cumsum(np.insert(count, 0, 0)[0:-1])]

        for var_name, agg_func in self.agg_func_names.items():
            if agg_func == "first":
                data_dict[var_name] = data_dict[var_name][first_point_idx]
            elif agg_func == "rand_choice":
                idx_select = idx_sort[
                    np.cumsum(np.insert(count, 0, 0)[0:-1])
                    + np.random.randint(0, count.max(), count.size) % count
                ]
                data_dict[var_name] = data_dict[var_name][idx_select]
            elif agg_func == "mean":
                data_dict[var_name] = (
                    np.add.reduceat(
                        data_dict[var_name][idx_sort],
                        np.cumsum(np.insert(count, 0, 0)[0:-1]),
                    )
                    / count[:, np.newaxis]
                )
            elif agg_func == "max":
                data_dict[var_name] = np.maximum.reduceat(
                    data_dict[var_name][idx_sort],
                    np.cumsum(np.insert(count, 0, 0)[0:-1]),
                )
            elif agg_func == "min":
                data_dict[var_name] = np.minimum.reduceat(
                    data_dict[var_name][idx_sort],
                    np.cumsum(np.insert(count, 0, 0)[0:-1]),
                )

        if self.return_inverse:
            data_dict["inverse"] = np.zeros_like(inverse)
            data_dict["inverse"][idx_sort] = inverse
        if self.return_grid_coord:
            data_dict["grid_coord"] = grid_coord[first_point_idx]
        if self.return_min_coord:
            data_dict["min_coord"] = min_coord.reshape([1, 3])
        return data_dict


@TRANSFORMS.register_module()
class SculptingOcclude(object):
    def __init__(
        self,
        cube_size_min=0.1,
        cube_size_max=0.5,
        npoint_frac=0.005,
        npoints=None,
        cell_size=0.02,
        density_factor=0.1,
        kill_color_proba=0.5,
        sampling="random",
    ):
        self.cube_size_min = cube_size_min
        self.cube_size_max = cube_size_max
        self.npoint_frac = npoint_frac
        self.npoints = npoints
        self.cell_size = cell_size
        self.density_factor = density_factor
        self.kill_color_proba = kill_color_proba
        self.sampling = sampling

    def get_random_colors(self, size, low=0, high=255):
        return np.random.randint(low, high, size).astype(np.float32)

    def get_random_normals(self, size):
        n = np.random.rand(*size).astype(np.float32) * 2 - 1
        n = n / np.linalg.norm(n, axis=-1)[:, np.newaxis]
        return n

    def add_random_cubes(self, data_dict):

        xyz = data_dict["coord"]
        rgb = data_dict.get("color", self.get_random_colors(xyz.shape))
        normal = data_dict.get("normal", self.get_random_normals(xyz.shape))

        semantic_label = data_dict.get("segment", np.ones(len(xyz), dtype=int))

        if self.npoints is None:
            ncubes = max(int(self.npoint_frac * len(xyz)), 1)
        else:
            ncubes = self.npoints

        cubes, cube_feats = get_random_colored_cubes_on_pts(
            self.cube_size_min,
            self.cube_size_max,
            xyz,
            feats=rgb,
            npoints=ncubes,
            cell_size=self.cell_size,
            actual_cube=False,
            sphere=False,
            point_sampling=self.sampling,
            density_factor=self.density_factor,
        )

        xyz = np.vstack([xyz, cubes])

        # rand_colors = self.get_random_colors(cubes.shape)
        rgb = np.vstack([rgb, cube_feats])

        if normal is not None:
            rand_normals = self.get_random_normals(cubes.shape)
            normal = np.vstack([normal, rand_normals])

        # Randomly turn colors off
        # if np.random.rand() < self.kill_color_proba:
        #     rgb = rgb * 0.0 + np.random.rand() * 255

        dummy_cube = np.ones(len(cubes), dtype=np.int32)
        dummy_pc = np.ones_like(semantic_label, dtype=np.int32)

        semantic_label = np.hstack([dummy_pc, 0 * dummy_cube])
        instance_label = np.hstack([-1 * dummy_pc, -1 * dummy_cube])

        return (
            xyz.astype(np.float32),
            rgb.astype(np.float32),
            semantic_label.astype(np.int32),
            normal.astype(np.float32),
            instance_label.astype(np.int32),
        )

    def __call__(self, data_dict):
        """
        for semseg models,
        data_dict.keys() = ['coord', 'color', 'normal', 'name', 'segment', 'instance']
        """

        (
            data_dict["coord"],
            data_dict["color"],
            data_dict["segment"],
            data_dict["normal"],
            data_dict["instance"],
        ) = self.add_random_cubes(data_dict)
        # from pointcept.utils import ForkedPdb; ForkedPdb().set_trace()
        return data_dict


@TRANSFORMS.register_module()
class SculptingMaskOcclude(object):
    def __init__(
        self,
        enable_feat_masking=True,
        mask_size=0.2,
        mask_ratio=0.5,
        cell_size=0.02,
        density_factor=0.1,
    ):
        self.enable_feat_masking = enable_feat_masking
        self.mask_size = mask_size
        self.mask_ratio = mask_ratio
        self.cell_size = cell_size
        self.density_factor = density_factor

    def hash(self, arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

    def get_sculpting_blocks_and_mask(
        self,
        coord,
    ):
        # Ratio of masked voxels
        MASK_RATIO = self.mask_ratio
        # Size of each masked voxel
        MASK_SIZE = self.mask_size
        # Size of each point in the cube-grid for sculpting
        SCULPT_CELL_SIZE = self.cell_size
        SCULPT_CELL_DENSITY = self.density_factor

        # Offset to start at origin and have positive indexes
        min_coord = np.min(coord, axis=0)
        grid_coord = ((coord - min_coord) // MASK_SIZE).astype(np.int32)

        # get voxel ids(torch impl)
        unique_cells, clusters = torch.unique(
            torch.tensor(grid_coord), dim=0, return_inverse=True
        )
        unique_cells=unique_cells.numpy()
        clusters=clusters.numpy()

        # Pick cells for masking
        ncells = unique_cells.shape[0]
        ncubes = int(ncells * MASK_RATIO)
        picked_cells = np.random.randint(low=0, high=ncells, size=(ncubes,))

        # Voxel coordinates of picked cells
        p0s = unique_cells[picked_cells]
        p0s = (
            p0s * MASK_SIZE + min_coord  # cell coordinates  # min_coord per batch index
        )

        # Place cubes at each picked cell
        c = get_pointgrid(int(MASK_SIZE // SCULPT_CELL_SIZE)) * SCULPT_CELL_SIZE
        # trick to do outer addition with broadcasting
        offsetted = p0s[None, ...] + c[:, None, :]
        offsetted = offsetted.reshape(-1, 3)

        # subsample cubes randomly
        rand_picks = np.arange(0, len(offsetted))
        np.random.shuffle(rand_picks)
        offsetted = offsetted[rand_picks[: int(SCULPT_CELL_DENSITY * len(offsetted))]]

        mask = np.isin(clusters, picked_cells).astype(int)
        return offsetted, mask

    def get_random_colors(self, size, low=0, high=255):
        return np.random.randint(low, high, size).astype(np.float32)

    def get_random_normals(self, size):
        n = np.random.rand(*size).astype(np.float32) * 2 - 1
        n = n / np.linalg.norm(n, axis=-1)[:, np.newaxis]
        return n

    def __call__(self, data_dict):
        # Input PC data
        xyz = data_dict["coord"]
        rgb = data_dict.get("color", self.get_random_colors(xyz.shape))
        normal = data_dict.get("normal", self.get_random_normals(xyz.shape))

        # Will be passed through SonataLikeSculptor.before_train
        self.mask_ratio = data_dict.get("mask_ratio", self.mask_ratio)
        self.mask_size = data_dict.get("mask_size", self.mask_size)

        cubes, mask = self.get_sculpting_blocks_and_mask(xyz)

        # mask will be 0 for original points, 1 for sculpted points, 2 for masked points
        mask = np.hstack((mask * 2, torch.full((len(cubes),), 1))).astype(np.int32)

        xyz = np.vstack([xyz, cubes])

        rgb = np.vstack([rgb, 0.0 * self.get_random_colors(cubes.shape)])

        if normal is not None:
            rand_normals = self.get_random_normals(cubes.shape)
            normal = np.vstack([normal, 0.0 * rand_normals])

        if self.enable_feat_masking:
            rgb[mask == 2] = [0.0, 0.0, 0.0]
            normal[mask == 2] = [0.0, 0.0, 0.0]

        data_dict["coord"] = xyz.astype(np.float32)
        data_dict["color"] = rgb.astype(np.float32)
        data_dict["normal"] = normal.astype(np.float32)
        data_dict["mask"] = mask.astype(np.int32)
        data_dict.pop("segment")
        data_dict.pop("instance")

        return data_dict

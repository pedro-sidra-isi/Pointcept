import numpy as np
import torch
from .sculpting_ops import (
    get_random_colored_cubes_on_pts,
    array_mode,
    array_rand_choice,
    array_choice,
    get_pointgrid,
)
from copy import deepcopy


from perlyn import (
    generate_perlin_noise,
)

# from perlin_numpy import (
#     generate_fractal_noise_2d,
#     generate_fractal_noise_3d,
#     generate_perlin_noise_2d,
#     generate_perlin_noise_3d,
# )

import pointcept.datasets.transform as transform
from pointcept.utils.registry import Registry
from pointcept.datasets.transform import TRANSFORMS
from .sculpting import SculptingOcclude


# TRANSFORMS = Registry("transforms")
def get_perlin(noise_num_cells, noise_cell_size):
    noise = generate_perlin_noise(
        noise_num_cells, # need to get multiple of 2
          (2, 2, 2), tileable=(False, False, False)
    )
    i, j, k = np.indices(noise.shape)

    noise = noise.flatten()

    threshold = 0.8/noise_num_cells[0]
    if np.random.rand()>0.5:
        locs = np.bitwise_and(noise < threshold, noise > -threshold)
    else:
        locs = np.bitwise_and(noise > 1e-6, noise < 2*threshold)

    i = noise_cell_size * i.flatten()[locs]
    j = noise_cell_size * j.flatten()[locs]
    k = noise_cell_size * k.flatten()[locs]
    return np.vstack([i, j, k])


def get_perlin_noise_on_pts(
    cube_size_min, cube_size_max, points, feats, npoints=10, *args, **kwargs
):
    idxs = np.random.randint(0, len(points), size=npoints)
    cell_size = kwargs.get("cell_size")

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

        cube_size = np.random.randint( _cube_size_min // cell_size, _cube_size_max // cell_size, (3,))
        # brain fart
        cube_size[1:]=cube_size[0]
        cube = get_perlin(
            noise_num_cells=(cube_size//2)*2,
            noise_cell_size=cell_size
        ).T + point.reshape((1, 3))

        feat = np.ones_like(cube) * f

        cubes.append(cube)
        cube_feats.append(feat)

    return np.vstack(cubes), np.vstack(cube_feats)


@TRANSFORMS.register_module()
class TrimmingOcclude(SculptingOcclude):
    def add_random_cubes(self, data_dict):
        xyz = data_dict["coord"]
        rgb = data_dict.get("color", self.get_random_colors(xyz.shape))
        normal = data_dict.get("normal", self.get_random_normals(xyz.shape))

        semantic_label = data_dict.get("segment", np.ones(len(xyz), dtype=int))

        if self.npoints is None:
            ncubes = int(self.npoint_frac * len(xyz))
        else:
            ncubes = self.npoints

        cubes, cube_feats = get_perlin_noise_on_pts(
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

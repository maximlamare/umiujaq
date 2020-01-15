#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processing functions.

This file is part of the Umiujaq processing toolbox
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
from osgeo import gdal
import numpy as np


def open_large_raster(inpath):
    """
    Open a raster with gdal, cutting the raster into chunks.

    Not necessaraly faster, but avoids reading full raster to memory.
    """
    ds = gdal.Open(str(inpath))
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)

    rows = band.YSize
    cols = band.XSize

    array = np.zeros((rows, cols))

    def optimal_tile_size(rast_obj, n, aspect=0):
        """
        Calculate optimal tile size.

        https://stackoverflow.com/questions/41742162/gdal-readasarray-for-vrt
        -extremely-slow/45745848#45745848

        Returns a tile size that optimizes reading a raster by considering the
        blocksize of the raster. The raster is divided into (roughly) N tiles.
        If the shape of the tiles is unimportant (aspect=0), optimization
        considers only the blocksize. If an aspect ratio is provided,
        optimization tries to respect it as much as possible.

        INPUTS:
            rast_obj - obj: gdal raster object created by gdal.Open()
                   N - int: number of tiles to split raster into
             aspect  - float or str: (optional) - If no value is provided, the
                       aspect ratio is set only by the blocksize. If aspect is
                       set to 'use_raster', aspect is obtained from the aspect
                       of the given raster. Optionally, an aspect may be
                       provided where aspect = dx/dy.

        OUTPUTS:
                  dx -  np.int: optimized number of columns of each tile
                  dy -  np.int: optimized number of rows of each tile

        """
        blocksize = rast_obj.GetRasterBand(1).GetBlockSize()

        ncols = rast_obj.RasterXSize
        nrows = rast_obj.RasterYSize

        # Compute ratios for sizing
        totalpix = ncols * nrows
        pix_per_block = blocksize[0] * blocksize[1]
        pix_per_tile = totalpix / n

        if aspect == 0:  # optimize tile size for fastest I/O

            n_blocks_per_tile = np.round(pix_per_tile / pix_per_block)

            if n_blocks_per_tile >= 1:
                # This assumes the larger dimension of the block size should
                # be retained for sizing tiles
                if blocksize[0] > blocksize[1] or blocksize[0] == blocksize[1]:
                    dx = blocksize[0]
                    dy = np.round(pix_per_tile / dx)
                    ndy = dy / nrows
                    if ndy > 1.5:
                        dx = dx * np.round(ndy)
                    dy = (
                        np.round((pix_per_tile / dx) / blocksize[1]) *
                        blocksize[1]
                    )
                    dy = np.min((dy, nrows))
                    if dy == 0:
                        dy = blocksize[1]
                else:
                    dy = blocksize[1]
                    dx = np.round(pix_per_tile / dy)
                    ndx = dx / ncols
                    if ndx > 1.5:
                        dy = dy * np.round(ndx)
                    dx = (
                        np.round((pix_per_tile / dy) / blocksize[0]) *
                        blocksize[0]
                    )
                    dx = np.min((dx, ncols))
                    if dx == 0:
                        dx = blocksize[0]

            else:
                print(
                    "Block size is smaller than tile size;"
                    "setting tile size to block size."
                )
                dy = blocksize[0]
                dx = blocksize[1]

        else:  # optimize but respect the aspect ratio as much as possible

            if aspect == "use_raster":
                aspect = ncols / nrows

            dya = np.round(np.sqrt(pix_per_tile / aspect))
            dxa = np.round(aspect * dya)

            dx = np.round(dxa / blocksize[0]) * blocksize[0]
            dx = np.min((dx, ncols))

            dy = np.round(dya / blocksize[1]) * blocksize[1]
            dy = np.min((dy, nrows))

            # Set dx,dy to blocksize if they're zero
            if dx == 0:
                dx = blocksize[0]
            if dy == 0:
                dy = blocksize[1]

        return dx, dy

    xb_size = int(optimal_tile_size(ds, 8, aspect=1)[0])
    yb_size = int(optimal_tile_size(ds, 8, aspect=1)[1])

    for i in range(0, rows, yb_size):
        if i + yb_size < rows:
            num_rows = yb_size
        else:
            num_rows = rows - i
        for j in range(0, cols, xb_size):
            if j + xb_size < cols:
                num_cols = xb_size
            else:
                num_cols = cols - j

            # Write chunk to numpy
            array[i: i + num_rows, j: j + num_cols] = band.ReadAsArray(
                j, i, num_cols, num_rows
            )

    band = None

    return array, gt, ds

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter vegetation

The script resamples a vegetation map with a resolution of 3cm to a 99cm
resolution vegetation map, taking the 90-98th percentile.
"""
from pathlib import Path
from osgeo import gdal
import numpy as np
from proc_toolbox import (open_large_raster, blockshaped, unblockshaped,
                          array_to_raster)

if __name__ == '__main__':

    # Paths to 3cm vegetation map
    hvege_path = Path("/home/rus/shared/Umiujaq/rasters/Hvege_3cm_NORD.tif")
    filtered_path = "/home/rus/shared/Umiujaq/rasters/Vege/Hvege_3cm_filtered_NORD.tif"
    averaged_filtered_path = "/home/rus/shared/Umiujaq/rasters/Vege/Hvege_99cm_filtered_NORD.tif"

    # Open the raster
    vege_raster, vege_gt, vege_ds = open_large_raster(hvege_path)

    # Pixel size of filter
    flt_res  = 33

    # Pad the raster with -9999 at the end if not divisable
    y_padding = ((vege_raster.shape[0] + flt_res) -
                 (vege_raster.shape[0] % flt_res)) - vege_raster.shape[0]
    x_padding = ((vege_raster.shape[1] + flt_res) -
                 (vege_raster.shape[1] % flt_res)) - vege_raster.shape[1]

    padded_raster = np.pad(vege_raster, ((0, y_padding), (0, x_padding)),
                           'constant', constant_values=-9999)

    # Blockshape raster, i.e. cut into blocks
    blocked_raster = blockshaped(padded_raster, flt_res, flt_res)

    # Manipulate blocks
    newarray = []
    merged = []
    for block in blocked_raster:
            filteredblock = np.ma.masked_where((block < np.percentile(block, 90)) |
                                               (block > np.percentile(block, 98)) |
                                               (block < 0), block)
            # Filter the block
            newarray.append(filteredblock)

            # Average filtered values
            merged.append(np.ma.average(filteredblock))

    # Build back array
    filtered_array = unblockshaped(np.ma.stack(newarray),
                                   padded_raster.shape[0],
                                   padded_raster.shape[1])
    # Unpad
    filtered_array = filtered_array[0:-y_padding, 0:-x_padding]
    filtered_array = np.ma.masked_where(filtered_array < 0, filtered_array)
    filtered_array = filtered_array.filled(-9999)

    # Save filtered raster
    array_to_raster(filtered_array, vege_ds.GetProjection(),
                    vege_gt, filtered_path)

    # Build the averaged raster
    resampled = np.reshape(merged, (int(padded_raster.shape[0] / flt_res),
                           int(padded_raster.shape[1] / flt_res)))

    # Save to geotiff
    gtres = list(hneige_raster_sud_gt)
    gtres[1] = 0.99
    gtres[-1] = -0.99
    resampled = resampled.filled(-9999)
    array_to_raster(resampled, vege_ds.GetProjection(), tuple(gtres),
                    averaged_filtered_path)

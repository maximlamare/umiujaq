#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unittests for the umi project
"""
import pytest
import numpy as np
from osgeo import osr, gdal

from geo_functs import (open_large_raster, open_raster, extract_cell_value,)


class Test_gdal(object):
    """Test the gdal operations"""

    def test_read_large(self):

        # Create a small array
        array = np.random.rand(667, 667)
        geotr = (305899.9802459951, 0.03, 0.0,
                 6270170.000074463, 0.0, -0.03)

        # Create a gdal file based on values from a small subset at umi
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create('/vsimem/inmem.tif', array.shape[0],
                                array.shape[0], 1, gdal.GDT_Float32)
        dataset.SetGeoTransform(geotr)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32189)
        dataset.SetProjection(srs.ExportToWkt())
        dataset.GetRasterBand(1).WriteArray(array)
        dataset.FlushCache()

        # Run the read function
        data, gt, ds = open_large_raster('/vsimem/inmem.tif')

        # Close in memory file
        gdal.Unlink('/vsimem/inmem.tif')

        # Test array and gt equality
        np.testing.assert_allclose(array, data, rtol=1e-4)
        assert gt == geotr

    def test_read(self):

        # Create a small array
        array = np.random.rand(667, 667)
        geotr = (305899.9802459951, 0.03, 0.0,
                 6270170.000074463, 0.0, -0.03)

        # Create a gdal file based on values from a small subset at umi
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create('/vsimem/inmem.tif', array.shape[0],
                                array.shape[0], 1, gdal.GDT_Float32)
        dataset.SetGeoTransform(geotr)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32189)
        dataset.SetProjection(srs.ExportToWkt())
        dataset.GetRasterBand(1).WriteArray(array)
        dataset.FlushCache()

        # Run the read function
        data, gt, ds = open_raster('/vsimem/inmem.tif')

        # Close in memory file
        gdal.Unlink('/vsimem/inmem.tif')

        # Test array and gt equality
        np.testing.assert_allclose(array, data, rtol=1e-4)
        assert gt == geotr

    def test_data_extraction(self):

        # Create a gdal object with a known array
        array = np.reshape(np.arange(0, 444889), (667, 667))

        # Create a geotransform
        geotr = (305899.9802459951, 0.03, 0.0,
                 6270170.000074463, 0.0, -0.03)

        cell_0_0 = extract_cell_value(305899.9927, 6270169.9914, array, geotr)
        cell_end_end = extract_cell_value(305919.9416, 6270149.9935, array,
                                          geotr)

        assert cell_0_0 == array[0, 0]

        # Test if the x and y are not inverted: asymetrical location
        assert cell_end_end == array[-1, -2]

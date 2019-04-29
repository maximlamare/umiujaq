#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:20:39 2019

@author: lamarem
"""
import numpy as np
import math
from osgeo import gdal
import pyproj
from shapely.geometry import Point, LineString
from shapely.ops import transform
from functools import partial


def open_raster(inpath):
    """ Open a DEM with gdal, return data and geotransform. """
    ds = gdal.Open(str(inpath))
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    band = None

    return data, gt, ds


def open_large_raster(inpath):
    """ Open a raster with gdal, cutting the raster into chunks. Not
    necessaraly faster, but avoids reading full raster to memory. """

    ds = gdal.Open(str(inpath))
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)

    rows = band.YSize
    cols = band.XSize

    array = np.zeros((rows, cols))

    def optimal_tile_size(rast_obj, N, aspect=0):
        """
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
        pix_per_tile = totalpix / N

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
                        np.round((pix_per_tile / dx) / blocksize[1])
                        * blocksize[1]
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
                        np.round((pix_per_tile / dy) / blocksize[0])
                        * blocksize[0]
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

    xBSize = int(optimal_tile_size(ds, 8, aspect=1)[0])
    yBSize = int(optimal_tile_size(ds, 8, aspect=1)[1])

    for i in range(0, rows, yBSize):
        if i + yBSize < rows:
            numRows = yBSize
        else:
            numRows = rows - i
        for j in range(0, cols, xBSize):
            if j + xBSize < cols:
                numCols = xBSize
            else:
                numCols = cols - j

            # Write chunk to numpy
            array[i : i + numRows, j : j + numCols] = band.ReadAsArray(
                j, i, numCols, numRows
            )

    band = None

    return array, gt, ds


def convert_line(input_line, in_epsg, out_epsg):
    """Convert a shapely Linestring from an EPSG to an other.

    INPUT:
    - input_line: shapely Linestring
    - in_epsg: the input EPSG code of the Linestring
    - out_epsg: the output EPSG code of the Linestring returned

    OUTPUT:
    - output_line: shapely Linestring"""

    project = partial(
        pyproj.transform,
        pyproj.Proj(init="EPSG:%s" % in_epsg),
        pyproj.Proj(init="EPSG:%s" % out_epsg),
    )

    return transform(project, input_line)


def forward_point_search(inlon, inlat, azimuth, distance):
    """Find point from a point, azimuth and distance.

    From a given longitude / latitude, azimuth and distance, find the
    longitude and latitude of a point. Uses a flat earth model, but this is
    fine for short distances, here around 100 m.

    Args:
        inlon (float): longitude of the coordinate in degrees EPSG: 4326
        inlat (float): latitude of the coordinate in degrees EPSG: 4326
        azimuth (float): azimuth of the search direction in degrees (geo.)
        distance (float): distance of the search direction in meters

    Returns:
        point (shapely.geometry.point): shapely point with coordinates in deg.
    """

    # Set the geodetic converter's ellipsoid
    g = pyproj.Geod(ellps="WGS84")

    # Use pyproj to get the terminus (accurate enough for this application)
    new_lon, new_lat, backaz = g.fwd(inlon, inlat, azimuth, distance)

    return Point(new_lon, new_lat)


def azimuth_search(start_point, end_point):

    # Use WGS84 ellipsoid
    g = pyproj.Geod(ellps="WGS84")

    # Calculate azimuth with pyproj
    return round(
        g.inv(start_point.x, start_point.y, end_point.x, end_point.y)[0] % 360
    )


def build_transect(center_point, tr_length, tr_azimuth, crs):

    # Check if you can divide the length by 2
    if tr_length % 2 != 0:
        raise ValueError("Please choose an even transect length.")
    else:
        # Calculate position of start point
        start_pt = forward_point_search(
            center_point.x,
            center_point.y,
            (tr_azimuth - 180) % 360,
            tr_length / 2,
        )

        # Calculate position of end point
        end_pt = forward_point_search(
            center_point.x, center_point.y, tr_azimuth, tr_length / 2
        )

        # Build line from start and end points
        latlon_line = LineString([start_pt, end_pt])

        projected_line = convert_line(latlon_line, 4326, crs)

        return latlon_line, projected_line


def build_points_on_line(inputline, number_of_points):

    # Empty list of coordinates
    point_coords = []

    for distance in np.arange(
        0, inputline.length, inputline.length / number_of_points
    ):

        # Create points along line
        point = inputline.interpolate(distance)

        # Append coordinates
        point_coords.append((point.x, point.y))

    return point_coords


def build_multiple_transects(
    start_point, end_point, width, number_of_transects, crs
):
    """Build multiple transects from 2 points

    From two points, given their longitude and latitude, build multiple
     parrallel transects. The spacing of the transects is defined by the width
     of the transect and the number of lines chosen.

    Args:
        start_point (shapely.geometry.point): position of the first point
        end_point (shapely.geometry.point): position of the first point
        width (float): width of the array of transects to be built in meters
        number_of_transects (int): number of parallel transects to build
        crs (int): EPSG code of the output

    Returns
        transects (list): list of (shapely.geometry.LineString) of the
                          transects
    """

    # Calculate the azimuth between the 2 points
    azimuth = azimuth_search(start_point, end_point)

    # Build the endlines 90 degrees of the guide line
    startline_ll, startline_pr = build_transect(
        start_point, width, azimuth + 90 % 360, crs
    )

    endline_ll, endline_pr = build_transect(
        end_point, width, azimuth + 90 % 360, crs
    )

    # Split the end lines into points
    startline_points = build_points_on_line(startline_pr, number_of_transects)
    endline_points = build_points_on_line(endline_pr, number_of_transects)

    # Group the end points of the new transects
    endpoints = list(zip(startline_points, endline_points))

    # Build lines from endpoints
    transects = []
    for pair in endpoints:
        transects.append(LineString([Point(pair[0]), Point(pair[1])]))

    return transects


def extract_cell_value(lon, lat, data, geotransform):
    """ Get the data in a georeferenced raster.

    Get the pixel value for a given coordinate (in the same projection as the
     raster).

    Args:
        lon (float): x value in the projection of the data.
        lat (float): y value in the projection of the data
        data (np.array): raster values stored in a numpy array
        geotransform (tuple): geotransform information (generally obtained from
                              gdal)

    Returns:
        (float): pixel value at given coordinates

    """
    # Get cell position given the geotransform
    px = int((lon - geotransform[0]) / geotransform[1])
    py = int((lat - geotransform[3]) / geotransform[5])

    return data[py, px]


def transects_data(transects, sampling_dist, data, data_geotrans):
    """
    """

    # Initialise lists of all transects z
    zall = []

    for line in transects:

        # Initialise lists for each transect
        z = []
        distance = []

        for dist in np.arange(0, line.length, sampling_dist):

            # Interpolate points over line of each transect
            point = line.interpolate(dist)

            # Pixel position of the point
            z.append(extract_cell_value(point.x, point.y, data, data_geotrans))

            # Append distance along transect
            distance.append(dist)

        # Append all transects
        zall.append(z)

    return (distance, zall)

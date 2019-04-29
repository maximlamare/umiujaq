# -*- coding: utf-8 -*-
"""
Umiujaq processing script
"""
from pathlib import Path
import sys
import numpy as np
from shapely.geometry import Point, LineString
from pandas import DataFrame
from pandas import HDFStore
import pickle
from argparse import ArgumentParser, ArgumentTypeError
from geo_functs import (
    open_large_raster,
    forward_point_search,
    build_multiple_transects,
    transects_data,
)


def main(
    raster,
    lon,
    lat,
    azi,
    tr_length,
    width,
    nb_transects,
    sampling_distance,
    crs,
    outfolder,
):
    """ Extract data along transects

    For a center point (lat / lon coordinates), a length, width and number of
     transects wanted build the transects and extract the data. """

    # Open the dem
    mns_solnu_sud, gt_mns_solnu_sud, ds_mns_solnu_sud = open_large_raster(
        str(raster)
    )

    # Get endpoints of the centre of the transect
    if tr_length % 2 != 0:
        raise ValueError("Please choose an even transect length.")
    else:
        start_point = forward_point_search(
            lon, lat, azi - 180 % 360, tr_length / 2
        )
        end_point = forward_point_search(lon, lat, azi, tr_length / 2)

    # Build transects
    transects = build_multiple_transects(
        start_point, end_point, width, nb_transects, crs
    )
    # Extract data
    distance, all_ele = transects_data(
        transects, sampling_distance, mns_solnu_sud, gt_mns_solnu_sud
    )

    # Make pandas
    extracted = DataFrame.from_records(all_ele, columns=distance)

    # Replace novalue (here -9999) by np.nan
    extracted.replace(-9999, np.nan, inplace=True)

    # Save extracted values
    extracted.to_hdf(
        outfolder.joinpath(
            raster.name.split(".")[0]
            + "_dist_%s_az_%s_w_%s_nb_%s"
            % (tr_length, azi, width, nb_transects)
        ),
        key="extracted",
        mode="w",
    )

    # Save transects
    with open(
        outfolder.joinpath(
            "transect_dist_%s_az_%s_w_%s_nb_%s.pkl"
            % (tr_length, azi, width, nb_transects)
        ),
        "wb",
    ) as f:
        pickle.dump(transects, f)


if __name__ == "__main__":

    # If no arguments, return a help message
    if len(sys.argv) == 1:
        print(
            'No arguments provided. Please run the command: "python %s -h"'
            "for help." % sys.argv[0]
        )
        sys.exit(2)
    else:
        # Parse Arguments from command line
        parser = ArgumentParser(
            description="Import parameters for the Umiujaq processing"
        )
        parser.add_argument(
            "--raster",
            metavar="Geotiff",
            required=True,
            help="Path to the geotiff from which to" " extract the data",
        )
        parser.add_argument(
            "--lon",
            metavar="Longitude",
            required=True,
            type=float,
            help="Longitude of the center point",
        )
        parser.add_argument(
            "--lat",
            metavar="Latitude",
            required=True,
            type=float,
            help="Latitude of the center point",
        )
        parser.add_argument(
            "--azi",
            metavar="Azimuth",
            required=True,
            type=float,
            help="Azimuth of the transect in degrees",
        )
        parser.add_argument(
            "--length",
            metavar="Transect length",
            required=True,
            type=float,
            help="Length of the transect in meters",
        )
        parser.add_argument(
            "--width",
            metavar="Transect width",
            required=True,
            type=float,
            help="Width of the transect in meters",
        )
        parser.add_argument(
            "--number",
            metavar="Number of transects",
            required=True,
            type=int,
            help="Number of transects across the width",
        )
        parser.add_argument(
            "--samp",
            metavar="Sampling distance",
            required=True,
            type=float,
            help="Distance between points along the transect" " in meters",
        )
        parser.add_argument(
            "--epsg",
            metavar="Epsg code of the data",
            required=True,
            type=int,
            help="EPSG code",
        )
        parser.add_argument(
            "--out",
            metavar="Output path",
            required=True,
            help="Output path of the saved transects",
        )

        input_args = parser.parse_args()

        # Run main
        main(
            Path(input_args.raster),
            input_args.lon,
            input_args.lat,
            input_args.azi,
            input_args.length,
            input_args.width,
            input_args.number,
            input_args.samp,
            input_args.epsg,
            Path(input_args.out),
        )

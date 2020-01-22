# -*- coding: utf-8 -*-
"""
Umiujaq plotting script.

Based on the output of the Profile extraction code, plot the transects.
"""
import sys
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator

import cartopy.crs as ccrs

from geo_functs import open_large_raster


def main(infold, mns_path, epsg_code, outfile, titre):
    """Main plotting function."""
    files = {}
    for file in infold.iterdir():
        files.update({file.name: file})

    # Extract the info
    nb = list(files.keys())[0].split("_")[-1].split(".")[0]
    width = list(files.keys())[0].split("_")[-3]
    azi = list(files.keys())[0].split("_")[-5]

    # Open gdal raster
    MNS_data, MNS_gt, MNS_ds = open_large_raster(str(mns_path))

    # Open transect
    data_tr = {}
    for fname, pth in files.items():
        if "transect" in fname:
            with open(pth, "rb") as f:
                transects = pickle.load(f)
        else:
            data_tr.update({"_".join(fname.split("_")[0:2]): pd.read_hdf(pth)})

    # Get very approximate center of transects
    midishline = transects[int(len(transects) / 2)]
    mid_point = midishline.interpolate(0.5, normalized=True)
    midpoint_buffer = mid_point.buffer(midishline.length / 2)
    envelope = midpoint_buffer.envelope

    # Turn interactive plotting off
    plt.ioff()

    # Create figure
    fig = plt.figure(figsize=(15.4, 6.6))
    fig.suptitle(titre)

    # Epsg
    proj_code = ccrs.epsg(epsg_code)

    # 2 by 2 grid
    gs = GridSpec(ncols=3, nrows=2, figure=fig, width_ratios=[0.1, 1.5, 4])
    ax = plt.subplot(gs[0, 1], projection=proj_code)
    ax1 = plt.subplot(gs[:, 0])
    ax2 = plt.subplot(gs[-1, 1], projection=proj_code)
    ax3 = plt.subplot(gs[:, -1])

    # AX
    mns_masked = np.ma.masked_where(MNS_data < 0, MNS_data)
    extent = (
        MNS_gt[0],
        MNS_gt[0] + MNS_ds.RasterXSize * MNS_gt[1],
        MNS_gt[3] + MNS_ds.RasterYSize * MNS_gt[5],
        MNS_gt[3],
    )

    ax.imshow(
        mns_masked, extent=extent, origin="upper", cmap="gist_earth"
    )

    ax.plot(
        [midishline.coords[0][0], midishline.coords[-1][0]],
        [midishline.coords[0][1], midishline.coords[-1][1]],
        linestyle="-",
        color="red",
        linewidth=1,
    )

    norm = Normalize(vmin=np.min(mns_masked), vmax=np.max(mns_masked))
    cbar = ColorbarBase(
        ax1, cmap=plt.get_cmap("gist_earth"), norm=norm, orientation="vertical"
    )
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.set_ylabel("Altitude / m")

    # AX2
    ax2.imshow(
        mns_masked, extent=extent, origin="upper", cmap="gist_earth"
    )

    for line in transects:
        ax2.plot(
            [line.coords[0][0], line.coords[-1][0]],
            [line.coords[0][1], line.coords[-1][1]],
            linestyle="-",
            color="black",
            alpha=0.6,
            linewidth=0.5,
        )

    ax2.set_extent(
        [
            envelope.bounds[0],
            envelope.bounds[2],
            envelope.bounds[1],
            envelope.bounds[-1],
        ],
        crs=ccrs.epsg(epsg_code),
    )

    ax2.set_title("Zoom on transects", y=-0.2)
    # AX3

    # Plot MNT/ MNS ground
    data_tr["MNT_solnu"].T.plot(
        ax=ax3, color="sienna", alpha=0.1, legend=False
    )
    data_tr["MNT_solnu"].T.mean(axis=1).plot(
        ax=ax3, color="sienna", legend=True, label="Mean summer DTM"
    )

    data_tr["MNS_solnu"].T.plot(
        ax=ax3, color="lightgreen", alpha=0.1, legend=False
    )
    data_tr["MNS_solnu"].T.mean(axis=1).plot(
        ax=ax3, color="lightgreen", legend=True, label="Mean summer DSM"
    )

    # Plot MNS neige
    data_tr["MNS_neige"].T.plot(
        ax=ax3, color="midnightblue", alpha=0.2, legend=False
    )
    data_tr["MNS_neige"].T.mean(axis=1).plot(
        ax=ax3, color="midnightblue", legend=True, label="Mean winter DSM"
    )

    ax3.set_title(
        "Azimuth: %s°, Width: %sm, # of transects: %s" % (azi, width, nb)
    )
    ax3.set_xlabel("Distance along transect / m")
    ax3.set_ylabel("Altitude / m")
    ax3.set_xlim(0, midishline.length)
    ax3.set_ylim(
        np.nanmin(data_tr["MNT_solnu"].T.mean(axis=1)) - 5,
        np.nanmax(data_tr["MNS_neige"].T.mean(axis=1)) + 5,
    )

    ax3.xaxis.set_major_locator(MultipleLocator(10))
    ax3.xaxis.set_minor_locator(MultipleLocator(5))
    ax3.yaxis.set_major_locator(MultipleLocator(1))
    ax3.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax3.xaxis.set_ticks_position("both")
    ax3.yaxis.set_ticks_position("both")
    ax3.tick_params(direction="inout", which="both")

    fig.savefig(infold.joinpath(outfile), bbox_inches="tight", dpi=300)


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
            description="Import parameters for the Umiujaq plotting"
        )
        parser.add_argument(
            "--infold",
            metavar="Input path",
            required=True,
            help="Path to the folder containg the MNS and MNT"
            " neige and solnu extracted data",
        )
        parser.add_argument(
            "--mns", metavar="MNS", required=True, help="Path to MNS solnu"
        )
        parser.add_argument(
            "--epsg",
            metavar="Epsg code of the data",
            required=True,
            type=int,
            help="EPSG code",
        )
        parser.add_argument(
            "--outfile",
            metavar="Outfolder",
            required=True,
            help="Path to the saved pdf",
        )
        parser.add_argument(
            "--titre",
            metavar="Figure title",
            required=False,
            default="Transect",
            type=str,
            help="Title for the figure",
        )

        input_args = parser.parse_args()

        # Run main
        main(
            Path(input_args.infold),
            input_args.mns,
            input_args.epsg,
            input_args.outfile,
            input_args.titre,
        )

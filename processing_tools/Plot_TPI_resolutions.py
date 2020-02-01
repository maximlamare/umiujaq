#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot scores.

Plot the TPI vs resolution scores.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from proc_toolbox import (open_large_raster, calculate_stats)


def plot_stats(data, output):
    """Plot the scores as a function of resolution."""
    # Create figure
    fig, ax = plt.subplots(figsize=(3.54, 2.36), dpi=300,)

    r2n = []
    r2s = []
    for key in data["NORD"]["tpi"]:
        r2n.append((int(''.join(filter(str.isdigit, key))),
                   data["NORD"]["tpi"][key]["stats"]["r2"]))
    for key in data["SUD"]["tpi"]:
        r2s.append((int(''.join(filter(str.isdigit, key))),
                   data["SUD"]["tpi"][key]["stats"]["r2"]))

    r2n.sort(key=lambda tup: tup[0])
    r2s.sort(key=lambda tup: tup[0])

    ax.plot([x[0] for x in r2n], [y[1] for y in r2n], linestyle="-",
            marker='o', markersize=2,
            color="salmon", linewidth=1, label="North site")
    ax.plot([x[0] for x in r2s], [y[1] for y in r2s], linestyle="-",
            marker='o', markersize=2,
            color="steelblue", linewidth=1, label="South site")

    # Labels
    ax.set_xlabel("TPI search distance / m", fontsize=7)
    ax.set_ylabel(r"r$^2$", fontsize=7)

    # Axis
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 1)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(axis='both', which='both', direction='inout', top=True,
                   right=True, left=True,
                   bottom=True, labelsize=7)
    ax.legend(loc=2, prop={'size': 7})
    plt.show()
    fig.savefig(output.joinpath("tpi_resolution.png"), dpi=300,
                bbox_inches="tight")


def main(tpi_folder, sd_north_folder, sd_south_folder, output):
    """Process before plotting."""
    # Initialise dictionnary
    data = {}
    data["NORD"] = {"path": sd_north_folder,
                    "tpi": {}}
    data["SUD"] = {"path": sd_south_folder,
                   "tpi": {}}

    # Open all TPI and perform correlations
    for site in ["NORD", "SUD"]:
        # Get images and open
        for tpi_file in tpi_folder.glob("*.tif"):
            if site in tpi_file.name:
                # Open rasters
                tpi_raster = open_large_raster(tpi_file)[0]
                sd_raster = open_large_raster(data["%s" % site]["path"])[0]
                masked_tpi = np.ma.masked_where(((sd_raster <= 0) |
                                                 (tpi_raster <= -100) |
                                                 (np.isnan(tpi_raster)) |
                                                 (np.isnan(sd_raster))),
                                                tpi_raster)
                masked_sd = np.ma.masked_where(((sd_raster <= 0) |
                                                (tpi_raster <= -100) |
                                                (np.isnan(tpi_raster)) |
                                                (np.isnan(sd_raster))),
                                               sd_raster)
                data["%s" % site]["tpi"].update(
                    {"%s" % tpi_file.name.split('_')[-1].split('.')[0]:
                     {"data": (masked_sd.compressed(),
                               masked_tpi.compressed()),
                     "stats": calculate_stats(masked_tpi.compressed(),
                                              masked_sd.compressed())}})

    # Plot data
    plot_stats(data, output)

    return data


if __name__ == '__main__':

    # Basepath
    basepath = Path("/home/lamarem/Documents/Umiujaq/Papier/Data/"
                    "TPI_SD_correlation/Resolutions")

    # Snow depth at both sites
    sd_path_north = Path("/home/lamarem/Documents/Umiujaq/Papier/Data/"
                         "TPI_SD_correlation/Nord_neige_99cm.tif")
    sd_path_south = Path("/home/lamarem/Documents/Umiujaq/Papier/Data/"
                         "TPI_SD_correlation/Sud_neige_99cm.tif")

    # Specify path to output figure
    output_fig = Path("/home/lamarem/Documents/Umiujaq/Papier/Figures")

    # Run main
    all_data = main(basepath, sd_path_north, sd_path_south, output_fig)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot vegetation scatterplots.

Plot three vegetation vs snow depth scatterplots at different resolutions.
"""
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from proc_toolbox import (open_large_raster, rmse)


def plotting(data, res, output, site):
    """Plot function for large datasets."""
    # Create figure
    fig = plt.figure(figsize=(7, 2.21), dpi=300,)

    # Create gridspec setting for subplots
    outer_grid = gs.GridSpec(1, 2, wspace=0.05,
                             width_ratios=(1, 0.05))
    left_cell = outer_grid[0, 0]
    left_inner_grid = gs.GridSpecFromSubplotSpec(1, 3, left_cell, wspace=0,
                                                 width_ratios=(1, 1, 1))
    right_cell = outer_grid[0, 1]
    right_inner_grid = gs.GridSpecFromSubplotSpec(1, 1, right_cell)

    ax1 = plt.subplot(left_inner_grid[0, 0])
    ax2 = plt.subplot(left_inner_grid[0, 1])
    ax3 = plt.subplot(left_inner_grid[0, 2])
    ax4 = plt.subplot(right_inner_grid[0, 0])

    # Plot first panel
    hex1 = ax1.hexbin(data["%s_cm" % res[0]]["vegetation"],
                      data["%s_cm" % res[0]]["snow"],
                      gridsize=(200, 200), cmap="plasma",
                      norm=LogNorm(vmin=1, vmax=1e2),
                      mincnt=1)
    ax1.plot(data["%s_cm" % res[0]]["vegetation"],
             data["%s_cm" % res[0]]["vegetation"] *
             data["%s_cm" % res[0]]["slope"] +
             data["%s_cm" % res[0]]["intercept"],
             linestyle="-", linewidth=0.5, color="black")
    ax1.text(0.97, 0.05,
             "y = %sx+%s\n"
             r"$r^2$ = %s"
             "\nbias= %s"
             "\nrmse = %s\n"
             "N = %s" % (round(data["%s_cm" % res[0]]["slope"], 2),
                         round(data["%s_cm" % res[0]]["intercept"], 2),
                         round(data["%s_cm" % res[0]]["r2"], 2),
                         round(data["%s_cm" % res[0]]["bias"], 2),
                         round(data["%s_cm" % res[0]]["rmse"], 2),
                         len(data["%s_cm" % res[0]]["vegetation"])
                         ),
             transform=ax1.transAxes,
             fontsize=7,
             ha='right'
             )
    ax1.set_title("1 m")

    # Plot second panel
    ax2.hexbin(data["%s_cm" % res[1]]["vegetation"],
               data["%s_cm" % res[1]]["snow"],
               gridsize=(200, 200), cmap="plasma",
               norm=LogNorm(vmin=1, vmax=1e2),
               mincnt=1)
    ax2.plot(data["%s_cm" % res[1]]["vegetation"],
             data["%s_cm" % res[1]]["vegetation"] *
             data["%s_cm" % res[1]]["slope"] +
             data["%s_cm" % res[1]]["intercept"],
             linestyle="-", linewidth=0.5, color="black")
    ax2.text(0.97, 0.05,
             "y = %sx+%s\n"
             r"$r^2$ = %s"
             "\nbias= %s"
             "\nrmse = %s\n"
             "N = %s" % (round(data["%s_cm" % res[1]]["slope"], 2),
                         round(data["%s_cm" % res[1]]["intercept"], 2),
                         round(data["%s_cm" % res[1]]["r2"], 2),
                         round(data["%s_cm" % res[1]]["bias"], 2),
                         round(data["%s_cm" % res[1]]["rmse"], 2),
                         len(data["%s_cm" % res[1]]["vegetation"])
                         ),
             transform=ax2.transAxes,
             fontsize=7,
             ha='right'
             )
    ax2.set_title("10 m")
    # Plot  panel
    ax3.hexbin(data["%s_cm" % res[2]]["vegetation"],
               data["%s_cm" % res[2]]["snow"],
               gridsize=(200, 200), cmap="plasma",
               norm=LogNorm(vmin=1, vmax=1e2),
               mincnt=1)
    ax3.plot(data["%s_cm" % res[2]]["vegetation"],
             data["%s_cm" % res[2]]["vegetation"] *
             data["%s_cm" % res[2]]["slope"] +
             data["%s_cm" % res[2]]["intercept"],
             linestyle="-", linewidth=0.5, color="black")
    ax3.text(0.97, 0.05,
             "y = %sx+%s\n"
             r"$r^2$ = %s"
             "\nbias= %s"
             "\nrmse = %s\n"
             "N = %s" % (round(data["%s_cm" % res[2]]["slope"], 2),
                         round(data["%s_cm" % res[2]]["intercept"], 2),
                         round(data["%s_cm" % res[2]]["r2"], 2),
                         round(data["%s_cm" % res[2]]["bias"], 2),
                         round(data["%s_cm" % res[2]]["rmse"], 2),
                         len(data["%s_cm" % res[2]]["vegetation"])
                         ),
             transform=ax3.transAxes,
             fontsize=7,
             ha='right'
             )
    ax3.set_title("20 m")

    # Colorbar
    cbar = fig.colorbar(hex1, cax=ax4, orientation='vertical')
    cbar.set_label('Number of points / cell')

    # Axis settings
    plt.gcf().canvas.draw()
    for ax in [ax1, ax2, ax3]:
        # Limits
        ax.set_ylim(0, 3)
        ax.set_xlim(0, 3)

        # Ticks
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))

        ax.tick_params(axis='both', which='both', direction='inout', top=True,
                       right=True, left=True,
                       bottom=True, labelsize=8)
        # Labels
        if ax != ax1:
            ax.set_yticklabels([])

    # Remove overlapping labels
    plt.setp(ax1.get_xmajorticklabels()[-2:-1], visible=False)
    plt.setp(ax2.get_xmajorticklabels()[-2:-1], visible=False)

    # Axis labels
    ax1.set_ylabel("Snow depth / m")
    ax2.set_xlabel("Vegetation height / m")

    # Overall title
    fig.suptitle("Pixel resolution", fontsize=14, x=0.5, y=1.1)

    plt.tight_layout()
    plt.show()

    fig.savefig(output.joinpath("vege_sd_correlation_%s.png" % site), dpi=300,
                bbox_inches="tight")


def main(site, resolutions, vege_folder, snow_folder, output_fig):
    """Prepare data for plotting.

    Inputs:
        site: a string defining the north or south site
        resolutions: a list of 3 pixel resolutions in cm
        vege_folder: folder containing 3 vegetation rasters.
        snow_folder: folder containing 3 snow depth rasters.
        output_fig: path to save figure.

    Note: the rasters should have the site name and the resolution in the file
    name.
    """
    # Make a dictionnary to store data
    corr_data = {}

    # Loop through resolutions to populate the dictionnary
    for res in resolutions:
        # Open vegetation & snow
        vege = open_large_raster(vege_folder.joinpath(
            "%s_vege_%scm.tif" % (site, res)))[0]
        snow = open_large_raster(snow_folder.joinpath(
            "%s_neige_%scm.tif" % (site, res)))[0]
        # Filter rasters
        vege_masked = np.ma.masked_where(((vege <= 0) | (snow <= 0) |
                                          (np.isnan(vege)) |
                                          (np.isnan(snow))), vege)
        snow_masked = np.ma.masked_where(((vege <= 0) | (snow <= 0) |
                                          (np.isnan(vege)) |
                                          (np.isnan(snow))), snow)
        # Compress data
        vege_cm = vege_masked.compressed()
        snow_cm = snow_masked.compressed()

        # Compute stats on data
        slope, intercept, r_value, p_value, std_err = stats.linregress(vege_cm,
                                                                       snow_cm)
        # Feed dictionnary
        corr_data["%s_cm" % res] = {"vegetation": vege_cm,
                                    "snow": snow_cm,
                                    "slope": slope,
                                    "intercept": intercept,
                                    "r2": r_value ** 2,
                                    "p": p_value,
                                    "std_err": std_err,
                                    "bias": np.average(np.array(vege_cm) -
                                                       np.array(snow_cm)),
                                    "rmse": rmse(snow_cm, vege_cm),
                                    }
    # Plot figure
    plotting(corr_data, resolutions, output_fig, site)

    return corr_data


if __name__ == '__main__':

    # Specify site
    site = "Sud"

    # Specify three resolutions in cm
    sizes = [99, 999, 2001]

    # Specify folder path to vegetation rasters
    vege_folder = Path("/home/lamarem/Documents/Umiujaq/Papier/Data/"
                       "Vegetation_SD_correlation/Vege")
    snow_folder = Path("/home/lamarem/Documents/Umiujaq/Papier/Data/"
                       "Vegetation_SD_correlation/Neige")

    # Specify path to output figure
    output_fig = Path("/home/lamarem/Documents/Umiujaq/Papier/Figures")

    # Run main script
    data = main(site, sizes, vege_folder, snow_folder, output_fig)

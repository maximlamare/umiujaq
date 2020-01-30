#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot TPI scatterplot.

Plot a snow depth vs TPI scatterplot.
"""
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from proc_toolbox import (open_large_raster, rmse)


def plot_scatter(tpi_north, sd_north, stat_vals_north,
                 tpi_south, sd_south, stat_vals_south,
                 output):
    """Plot a scatterplot."""
    # Create figure
    fig = plt.figure(figsize=(3.89, 7), dpi=300,)

    # Create gridspec setting for subplots
    outer_grid = gs.GridSpec(1, 2, wspace=0.05,
                             width_ratios=(1, 0.05))
    left_cell = outer_grid[0, 0]
    left_inner_grid = gs.GridSpecFromSubplotSpec(2, 1, left_cell, hspace=0,)
    right_cell = outer_grid[0, 1]
    right_inner_grid = gs.GridSpecFromSubplotSpec(1, 1, right_cell)

    ax1 = plt.subplot(left_inner_grid[0, 0])
    ax2 = plt.subplot(left_inner_grid[1, 0])
    ax3 = plt.subplot(right_inner_grid[0, 0])

    # Add data
    hex1 = ax1.hexbin(tpi_north,
                      sd_north,
                      gridsize=(200, 200), cmap="plasma",
                      norm=LogNorm(vmin=1, vmax=1e2),
                      mincnt=1)
    ax1.plot(tpi_north,
             tpi_north *
             stat_vals_north["slope"] +
             stat_vals_north["intercept"],
             linestyle="-", linewidth=1, color="black")

    ax1.text(0.7, 0.7,
             "y = %sx+%s\n"
             r"$r^2$ = %s"
             "\nbias= %s"
             "\nrmse = %s\n"
             "N = %s" % (round(stat_vals_north["slope"], 2),
                         round(stat_vals_north["intercept"], 2),
                         round(stat_vals_north["r2"], 2),
                         round(stat_vals_north["bias"], 2),
                         round(stat_vals_north["rmse"], 2),
                         len(tpi_north)
                         ),
             transform=ax1.transAxes,
             fontsize=7,
             ha='left'
             )
    ax1.text(0.02, 0.02, "a)", transform=ax1.transAxes)

    ax2.hexbin(tpi_south,
               sd_south,
               gridsize=(200, 200), cmap="plasma",
               norm=LogNorm(vmin=1, vmax=1e2),
               mincnt=1)
    ax2.plot(tpi_south,
             tpi_south *
             stat_vals_south["slope"] +
             stat_vals_south["intercept"],
             linestyle="-", linewidth=1, color="black")
    ax2.text(0.7, 0.7,
             "y = %sx+%s\n"
             r"$r^2$ = %s"
             "\nbias= %s"
             "\nrmse = %s\n"
             "N = %s" % (round(stat_vals_south["slope"], 2),
                         round(stat_vals_south["intercept"], 2),
                         round(stat_vals_south["r2"], 2),
                         round(stat_vals_south["bias"], 2),
                         round(stat_vals_south["rmse"], 2),
                         len(tpi_south)
                         ),
             transform=ax2.transAxes,
             fontsize=7,
             ha='left'
             )
    ax2.text(0.02, 0.02, "b)", transform=ax2.transAxes)

    # Axes
    plt.gcf().canvas.draw()
    for ax in [ax1, ax2]:
        ax.set_ylim(0, 3)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylabel("Snow Depth / m")
        ax.set_xlabel("TPI")
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis='both', which='both', direction='inout', top=True,
                       right=True, left=True,
                       bottom=True, labelsize=8)

    ax1.set_xticklabels([])
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    # Remove overlapping labels
    plt.setp(ax2.get_ymajorticklabels()[-2:-1], visible=False)

    # Colorbar
    cbar = fig.colorbar(hex1, cax=ax3, orientation='vertical')
    cbar.set_label('Number of points / cell')

    plt.show()

    fig.savefig(output.joinpath("tpi_sd_correlation.png"), dpi=300,
                bbox_inches="tight")


def calculate_stats(tpi, sd):
    """Calculate statistics."""
    # Compute linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(tpi, sd)
    rmse_val = rmse(sd, tpi)
    bias = np.average(np.array(sd) - np.array(tpi))

    # Fill dictionnary
    statdic = {"slope": slope,
               "intercept": intercept,
               "r2": r_value ** 2,
               "rmse": rmse_val,
               "bias": bias}

    return statdic


if __name__ == '__main__':

    # Specify path to output figure
    output_fig = Path("/home/lamarem/Documents/Umiujaq/Papier/Figures")

    # NORTH
    # Specify path to TPI and snow depth files
    tpi_path_north = Path("/home/lamarem/Documents/Umiujaq/Papier/Data/"
                          "TPI_SD_correlation/TPI_Nord_99cm_33px.tif")
    sd_path_north = Path("/home/lamarem/Documents/Umiujaq/Papier/Data/"
                         "TPI_SD_correlation/Nord_neige_99cm.tif")
    # SOUTH
    # Specify path to TPI and snow depth files
    tpi_path_south = Path("/home/lamarem/Documents/Umiujaq/Papier/Data/"
                          "TPI_SD_correlation/TPI_Sud_99cm_33px.tif")
    sd_path_south = Path("/home/lamarem/Documents/Umiujaq/Papier/Data/"
                         "TPI_SD_correlation/Sud_neige_99cm.tif")

    # Open data
    tpi_north = open_large_raster(tpi_path_north)[0]
    sd_north = open_large_raster(sd_path_north)[0]
    tpi_south = open_large_raster(tpi_path_south)[0]
    sd_south = open_large_raster(sd_path_south)[0]

    # Filter values
    tpi_masked_north = np.ma.masked_where(((sd_north <= 0) |
                                           (tpi_north <= -100) |
                                           (np.isnan(sd_north)) |
                                           (np.isnan(tpi_north))),
                                          tpi_north)
    sd_masked_north = np.ma.masked_where(((sd_north <= 0) |
                                          (tpi_north <= -100) |
                                          (np.isnan(sd_north)) |
                                          (np.isnan(tpi_north))),
                                         sd_north)
    tpi_masked_south = np.ma.masked_where(((sd_south <= 0) |
                                           (tpi_south <= -100) |
                                           (np.isnan(sd_south)) |
                                           (np.isnan(tpi_south))),
                                          tpi_south)
    sd_masked_south = np.ma.masked_where(((sd_south <= 0) |
                                          (tpi_south <= -100) |
                                          (np.isnan(sd_south)) |
                                          (np.isnan(tpi_south))),
                                         sd_south)
    # Compress data
    tpi_cm_north = tpi_masked_north.compressed()
    sd_cm_north = sd_masked_north.compressed()
    tpi_cm_south = tpi_masked_south.compressed()
    sd_cm_south = sd_masked_south.compressed()

    # Calculate statistics
    stat_values_north = calculate_stats(tpi_cm_north, sd_cm_north)
    stat_values_south = calculate_stats(tpi_cm_south, sd_cm_south)

    # Plot scatter plot
    plot_scatter(tpi_cm_north, sd_cm_north, stat_values_north,
                 tpi_cm_south, sd_cm_south, stat_values_south,
                 output_fig)

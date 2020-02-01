#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot transect figure."""
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from math import floor
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import MultipleLocator
from matplotlib import patheffects
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import matplotlib.lines as lns
import cartopy.crs as ccrs
from proc_toolbox import open_large_raster


def add_interval(ax, xdata, ydata, caps="  "):
    line = ax.add_line(lns.Line2D(xdata, ydata))
    anno_args = {
        'ha': 'center',
        'va': 'center',
        'size': 24,
        'color': line.get_color()
    }
    a0 = ax.annotate(caps[0], xy=(xdata[0], ydata[0]), **anno_args)
    a1 = ax.annotate(caps[1], xy=(xdata[1], ydata[1]), **anno_args)
    return (line,(a0,a1))


def scale_bar(ax, proj, length, location=(0.5, 0.05), linewidth=3,
              units='km', m_per_unit=1000, UTM=18):
    """

    http://stackoverflow.com/a/35705477/1072212
    ax is the axes to draw the scalebar on.
    proj is the projection the axes are in
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
    length is the length of the scalebar in km.
    linewidth is the thickness of the scalebar.
    units is the name of the unit
    m_per_unit is the number of meters in a unit
    """
    # find lat/lon center to find best UTM zone
    x0, x1, y0, y1 = ax.get_extent(proj.as_geodetic())
    # Projection in metres
    utm = ccrs.UTM(UTM)
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(utm)
    # Turn the specified scalebar location into coordinates in metres
    sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbcx - length * m_per_unit/2, sbcx + length * m_per_unit/2]
    # buffer for scalebar
    buffer = [patheffects.withStroke(linewidth=5, foreground="w")]
    # Plot the scalebar with buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
        linewidth=linewidth, path_effects=None)
    
  
    # buffer for text
    buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
    # Plot the scalebar label
    t0 = ax.text(sbcx, sbcy, str(length) + ' ' + units, transform=utm,
        horizontalalignment='center', verticalalignment='bottom',
        path_effects=None, zorder=2)
    left = x0+(x1-x0)*0.05
    # Plot the N arrow
    # t1 = ax.text(left, sbcy, u'\u25B2\nN', transform=utm,
    #     horizontalalignment='center', verticalalignment='bottom',
    #     path_effects=buffer, zorder=2)
    # Plot the scalebar without buffer, in case covered by text buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
        linewidth=linewidth, zorder=3)
    
    
def plot_transects(MNS, transects, data_tr, output):
    """Plot the figure with data and transects."""
    # Create figure
    fig = plt.figure(figsize=(7, 5), dpi=300,)
    
    # Outer grid of 2 horizontal
    outer_grid = gs.GridSpec(1, 2, wspace=0.3,
                             width_ratios=(0.7, 1))
    
    # Cut left panel in 2 vertical
    left_cell = outer_grid[0, 0]    
    left_grid = gs.GridSpecFromSubplotSpec(2, 1, left_cell,)
    
   
    # Cut the top left into 2
    top_left_cell = left_grid[0, 0]
    top_left_grid = gs.GridSpecFromSubplotSpec(1, 2, top_left_cell,
                                               width_ratios= (0.2, 1))
    
    # Cut the bottom left into 2
    bottom_left_cell = left_grid[1, 0]
    bottom_left_grid = gs.GridSpecFromSubplotSpec(1, 2, bottom_left_cell,
                                                  width_ratios= (0.2, 1))

    # Cut the right cell into 3
    right_cell = outer_grid[0, 1]
    right_grid = gs.GridSpecFromSubplotSpec(3, 1, right_cell, hspace=0)

    # Epsg
    proj_code = ccrs.epsg(2951)


    # Right
    ax1 = plt.subplot(top_left_grid[0, 0])
    ax2 = plt.subplot(top_left_grid[0, 1],projection=proj_code)
    ax3 = plt.subplot(bottom_left_grid[0, 1], projection=proj_code)
    ax4 = plt.subplot(right_grid[0, 0])
    ax5 = plt.subplot(right_grid[1, 0])
    ax6 = plt.subplot(right_grid[2, 0])
    ax7 = plt.subplot(bottom_left_grid[0, 0])
    
    # Start plotting
    # MNS sol nu overall
    mns_masked = np.ma.masked_where(MNS[0] < 0, MNS[0])
    extent = (
        MNS[1][0],
        MNS[1][0] + MNS[2].RasterXSize * MNS[1][1],
        MNS[1][3] + MNS[2].RasterYSize * MNS[1][5],
        MNS[1][3],
    )

    im1 = ax2.imshow(
        mns_masked, extent=extent, origin="upper", cmap="gist_earth",
        vmin=125, vmax=140)

    # Make transects to plot
    midishline = transects[int(len(transects) / 2)]
    mid_point = midishline.interpolate(0.5, normalized=True)
    midpoint_buffer = mid_point.buffer(midishline.length / 2)
    envelope = midpoint_buffer.envelope

    # Overlay transect lines
    ax2.plot(
        [midishline.coords[0][0], midishline.coords[-1][0]],
        [midishline.coords[0][1], midishline.coords[-1][1]],
        linestyle="-",
        color="red",
        linewidth=1,
    )

    # Plot zoom on MNS
    im2 = ax3.imshow(
        mns_masked, extent=extent, origin="upper", cmap="gist_earth",
        vmin=130, vmax=138
    )

    for line in transects:
        ax3.plot(
            [line.coords[0][0], line.coords[-1][0]],
            [line.coords[0][1], line.coords[-1][1]],
            linestyle="-",
            color="black",
            alpha=0.6,
            linewidth=0.5,
        )

    ax3.set_extent(
        [
            envelope.bounds[0],
            envelope.bounds[2],
            envelope.bounds[1],
            envelope.bounds[-1],
        ],
        crs=ccrs.epsg(2951),
    )

    # ax3.set_title("Zoom on transects", y=-0.2)
    
    # Connect 
    # ax2.add_patch(ptch.Rectangle((0.2, 0.2), 0.28, 0.28,
    #                                             transform=ax2.transAxes,
    #                                             alpha=0.3, ec="k", fill=False))
    
    
    # Colorbar
    ip = InsetPosition(ax2, [-0.1,0,0.03,1]) 
    ax1.set_axes_locator(ip)

    cbar = fig.colorbar(im1, cax=ax1, ax=[ax1,ax2])
    # cbar = fig.colorbar(im1, cax=ax1, orientation='vertical')
    cbar.set_label("Altitude / m")
    ax1.yaxis.tick_left()   
    cbar.ax.yaxis.set_label_position('left')
    
    ip2 = InsetPosition(ax3, [-0.1,0,0.03,1]) 
    ax7.set_axes_locator(ip2)

    cbar2 = fig.colorbar(im2, cax=ax7, ax=[ax7,ax3])
    # cbar = fig.colorbar(im1, cax=ax1, orientation='vertical')
    cbar2.set_label("Altitude / m")
    ax7.yaxis.tick_left()   
    cbar2.ax.yaxis.set_label_position('left')

   
    # Top transec
    data_tr["MNT_solnu"].T.plot(
        ax=ax4, color="sienna", alpha=0.1, legend=False, label=""
    )
    data_tr["MNT_solnu"].T.mean(axis=1).plot(
        ax=ax4, color="sienna", legend=True, label="Mean summer DEM",
    )

   

    data_tr["MNS_solnu"].T.plot(
        ax=ax4, color="lightgreen", alpha=0.1, legend=False
    )
    data_tr["MNS_solnu"].T.mean(axis=1).plot(
        ax=ax4, color="lightgreen", legend=True, label="Mean summer DSM",
        fontsize=7
    )
    
   
    
    # Plot MNS neige
    data_tr["MNS_neige"].T.plot(
        ax=ax4, color="midnightblue", alpha=0.2, legend=False
    )
    data_tr["MNS_neige"].T.mean(axis=1).plot(
        ax=ax4, color="midnightblue", legend=True, label="Mean winter DEM",
        
    )
    patches, labels = ax4.get_legend_handles_labels()
    print(labels)
    pp = [patches[8],patches[17],patches[-1]]
    ll = [labels[8],labels[17],labels[-1]]
    ax4.legend(pp, ll, loc='best', prop={"size": 6})

    # Plot height
    data_tr["neige"].T.plot(
        ax=ax5, color="midnightblue", alpha=0.2, legend=False,
    )
    data_tr["neige"].T.mean(axis=1).plot(
        ax=ax5, color="midnightblue", legend=True, label="Snow"
    )
    data_tr["vege"].T.plot(
        ax=ax5, color="forestgreen", alpha=0.2, legend=False,
    )
    data_tr["vege"].T.mean(axis=1).plot(
        ax=ax5, color="forestgreen", legend=True, label="Vegetation",
        fontsize=7
    )
    patches, labels = ax5.get_legend_handles_labels()
    print(labels)
    pp = [patches[8], patches[-1]]
    print(pp)
    ll = [labels[8], labels[-1]]
    ax5.legend(pp, ll, loc='best', prop={"size": 6})

     # Plot height
    data_tr["TPI"].T.plot(
        ax=ax6, color="black", alpha=0.2, legend=False, label="TPI"
    )
    data_tr["TPI"].T.mean(axis=1).plot(
        ax=ax6, color="black", legend=False, label="TPI"
    )
    
    scale_bar(ax3, proj_code, 50, location=(0.7, 0.05), linewidth=0.8,
              units='m', m_per_unit=1, UTM = "18V")
    
    scale_bar(ax2, proj_code, 100, location=(0.7, 0.05), linewidth=0.8,
              units='m', m_per_unit=1, UTM = "18V")
    
    
    # Axis
    ax2.set_xticks([])
    ax2.set_yticks([])


    ax4.set_ylim(132, 136)
    ax5.set_ylim(0, 3)
    ax6.set_ylim(-0.55, 0.55)
    
    plt.gcf().canvas.draw()

    for ax in [ax4, ax5, ax6]:
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.tick_params(axis='both', which='both', direction='inout', top=True,
                       right=True, left=True,
                       bottom=True, labelsize=7)
    # plt.setp(ax4.get_legend().get_texts(), fontsize='6')
    # plt.setp(ax5.get_legend().get_texts(), fontsize='6') # for legend text

    ax4.yaxis.set_major_locator(MultipleLocator(1))
    ax4.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax5.yaxis.set_major_locator(MultipleLocator(0.5))
    ax5.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax6.yaxis.set_major_locator(MultipleLocator(0.2))
    ax6.yaxis.set_minor_locator(MultipleLocator(0.1))
    
    ax4.set_ylabel("Altitude / m", fontsize=7)
    ax5.set_ylabel("Height / m", fontsize=7)
    ax6.set_ylabel("TPI", fontsize=7)
    ax6.set_xlabel("Distance along transect / m", fontsize=7)
    for ax in [ax4, ax5]:
        ax.set_xticks([])
    plt.setp(ax5.get_ymajorticklabels()[-2:-1], visible=False)
    
    ax4.set_title("Transect 6", fontsize=10)
    
    plt.show()
    
    fig.savefig(output.joinpath("transect6.svg"), dpi=300,
                bbox_inches="tight")



def main(mnt_sn_path, transects, output):
    """Perform operations to run plotting script."""
    # Open gdal raster
    MNS_data, MNS_gt, MNS_ds, MNS_prj = open_large_raster(str(mnt_sn_path))

    # Open transect
    data_tr = {}
    for fname in transects.iterdir():
        if "transect" in fname.name:
            with open(fname, "rb") as f:
                transects = pickle.load(f)
        else:
            data_tr.update({"_".join(
                fname.name.split("_")[0:2]): pd.read_hdf(fname)})
    
    # Make heading generic
    if "Nord_neige" in data_tr.keys():
        print("North")
        data_tr["neige"] = data_tr.pop("Nord_neige")
        data_tr["vege"] = data_tr.pop("Nord_vege")
        data_tr["TPI"] = data_tr.pop("TPI_Nord")   
    elif "Sud_neige" in data_tr.keys():
        print("South")
        data_tr["neige"] = data_tr.pop("Sud_neige")
        data_tr["vege"] = data_tr.pop("Sud_vege")
        data_tr["TPI"] = data_tr.pop("TPI_Sud")

    print(data_tr.keys())
    # Plot
    plot_transects([MNS_data, MNS_gt, MNS_ds], transects, data_tr, output)

    return data_tr


if __name__ == "__main__":
    # Paths to folders
    mnt_sol_nu = Path("/home/lamarem/Documents/Umiujaq/Papier/Data/"
                      "Transects/TR_6/input/MNS_solnu_99cm_SUD.tif")
    transect_folder = Path("/home/lamarem/Documents/Umiujaq/Papier/Data/"
                           "Transects/TR_6/tr_data/")
    
    output = Path("/home/lamarem/Documents/Umiujaq/Papier/Figures/")

    # Run script
    tr = main(mnt_sol_nu, transect_folder, output)

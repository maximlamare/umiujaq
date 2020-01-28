#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate TPI.

The script is part of the Umiujaq toolbox. It offers the possibility to
compute topographic position index.
"""
from pathlib import Path
import sys
from argparse import ArgumentParser
import numpy as np
from proc_toolbox import (open_large_raster, array_to_raster)


def try_odd(n):
    """Try even number.

    If number is not even raise an exception.
    """
    if n % 2 == 0:
        raise ValueError("Number is not odd.")


def create_window(width):
    """Create window.

    Create a sliding window of a given size. The window
     has to have a width = to an odd number.
    """
    # Test window size
    try_odd(width)
    if width < 3:
        raise ValueError("Window too small!")

    # Build window
    win = np.ones((width, width))

    # Deduce shape from window size
    r_y, r_x = win.shape[0] // 2, win.shape[1] // 2

    # Remove central cell
    win[r_y, r_x] = 0  # let's remove the central cell

    return win, r_y, r_x


def view(offset_y, offset_x, shape, step=1):
    """
    View function.

    Function returning two matching numpy views for moving window routines.
    - 'offset_y' and 'offset_x' refer to the shift in relation to the analysed
     (central) cell
    - 'shape' are 2 dimensions of the data matrix (not of the window!)
    - 'view_in' is the shifted view and 'view_out' is the position of central
     cells
    (see on LandscapeArchaeology.org/2018/numpy-loops/)
    """
    size_y, size_x = shape
    x, y = abs(offset_x), abs(offset_y)

    x_in = slice(x, size_x, step)
    x_out = slice(0, size_x - x, step)

    y_in = slice(y, size_y, step)
    y_out = slice(0, size_y - y, step)

    # the swapping trick
    if offset_x < 0:
        x_in, x_out = x_out, x_in
    if offset_y < 0:
        y_in, y_out = y_out, y_in

    # return window view (in) and main view (out)
    return np.s_[y_in, x_in], np.s_[y_out, x_out]


def main(raster_path, window_size, output):
    """Run TPI index."""
    # Open raster
    raster, gt, ds, prj = open_large_raster(raster_path)

    # Remove the negative values
    raster[raster < 0] = np.nan

    # Create window
    win, r_y, r_x = create_window(window_size)

    # Initialise matrices for temporary data
    mx_temp = np.zeros(raster.shape)
    mx_count = np.zeros(raster.shape)

    # Loop through window and accumulate values
    for (y, x), weight in np.ndenumerate(win):
        # Skip 0 values
        if weight == 0:
            continue
        # Determine views to extract data
        view_in, view_out = view(y - r_y, x - r_x, raster.shape)

        # uUing window weights (eg. for a Gaussian function)
        mx_temp[view_out] += raster[view_in] * weight

        # Track the number of neighbours
        # (this is used for weighted mean :
        # Σ weights*val / Σ weights)
        mx_count[view_out] += weight

    # TPI (spot height – average neighbourhood height)
    tpi = raster - mx_temp / mx_count

    # Save raster
    array_to_raster(tpi, prj, gt, str(output))


if __name__ == '__main__':

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
            description="Parameters for the Umiujaq plotting"
        )
        parser.add_argument(
            "--raster",
            metavar="Input raster",
            required=True,
            help="Path to the raster on which to calculate TPI",
        )
        parser.add_argument(
            "--window",
            metavar="Window size",
            required=True,
            type=int,
            help="Window size for the TPI calculation",
        ),
        parser.add_argument(
            "--outfile",
            metavar="Output file",
            required=True,
            help="Path to the saved TPI raster",
        )

    input_args = parser.parse_args()

    # Run main
    main(Path(input_args.raster),
         input_args.window,
         Path(input_args.outfile),
         )

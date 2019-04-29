#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unittests for the umi project
"""
import pytest
import numpy as np
from shapely.geometry import Point, LineString
from geo_functs import (forward_point_search, build_transect,
                        azimuth_search, build_points_on_line)


class Test_directions(object):
    """Test the serch directions for points"""

    def test_forward_search(self):
        """Test the forward search direction"""

        # Create input point
        forward_point = forward_point_search(-76.482521, 56.558161, 100, 4200)

        # Test against known source
        assert (forward_point.x,
                forward_point.y) == pytest.approx((-76.41494, 56.55159),
                                                  rel=1e-5)

    def test_azimuth_search(self):
        """Test if pyproj returns the correct azimuth (rounded)"""

        # Create start and end points
        startpoint = Point(-76.482521, 56.558161)
        endpoint = Point(-76.41494, 56.55159)

        assert azimuth_search(startpoint, endpoint) == 100


class Test_builds(object):
    """ Test if the different geometrical objects are correctly constructed """

    def test_transect_build(self):
        """Test that the positions of the endpoints are in the correct place"""

        # Set a startpoint
        startpoint = Point(-76.482521, 56.558161)

        # Build line of 100 m long, azimuth 246°
        line = build_transect(startpoint, 200, 246, 32189)[0]

        assert line.coords[0] == pytest.approx((-76.48103532, 56.55852626),
                                               rel=1e-5)
        assert line.coords[-1] == pytest.approx((-76.48400665, 56.55779572),
                                                rel=1e-5)

    def test_points_on_line(self):
        """ Test if the points along a line are correctly built """

        # Create start point
        startpoint = Point(-76.482521, 56.558161)

        # Build line of 6 m long, azimuth 90
        line = build_transect(startpoint, 6, 90, 32189)[0]

        built_points = build_points_on_line(line, 20)

        # Check number of points
        assert len(built_points) == 20

        # Check if point is in correct place
        assert built_points[0] == pytest.approx((-76.48256979, 56.55816100),
                                                rel=1e-5)

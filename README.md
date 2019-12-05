# Umiujaq Lidar processing tools
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
The collection of tools in the Umiujaq processing suite allows to process the DTM and DSM data from Genidrone, to plot elevation profiles along transects.
Development and testing: Maxim Lamare (IGE).
# Content
1. [Python scripts](#python)
2. [Batch processing](#bash)
3. [Processing tools](#tools)

<a name="python"></a>
# Python scripts
**Umi_profiles.py**
This script extracts the values from a raster along a number of defined transects. 
The script needs the following inputs:
 - ***--raster***: the path to a geotiff from which to extract the values
 - ***--lon***: the longitude (in decimal degrees, WGS84) of the centre point of the transect.
 - ***--lat***: the latitude (in decimal degrees, WGS84) of the centre point of the transect.
 - ***--azi***: the azimuth (geographic, in degrees) along which the transect will be built.
 - ***--length***: the length of the transect in meters.
 - ***--width***: the width of the array of transects.
 - ***--number***: the number of transects to be built. The spacing of the transects is determined by the width/number combination.
 - ***--samp***: the interval along the transects for the sampling of the geotiff, in meters.
 - ***--epsg***: the EPSG code of the geotiff, as GDAL cannot always resolve the code automatically.
 - ***--out***: the path of the folder in which the transect data will be saved.
 
*Note: the transect is built out from the centre point. A line is created in the direction of the given azimuth with the lat/lon at the centre of the line. At the end nodes of the line perpendicular lines are created, with a length equal to the specified width. Points are interpolated along the end lines according to the number of transects specified. Then those points are connected to form the transects.* 

Example run:

To extract the data for an array of transects centred at lat/lon = 56.559, -76.481 for an azimuth of 45° (NE), a length of 100m, a width of 4m and 8 transects (every 0.5m), with the geotiff of EPSG 32189 sampled every 10cm along the transects:

    python Umi_profiles.py --raster "/path/to/MNT_solnu_3cm_SUD.tif" --lon -76.481 --lat 56.559 --azi 45 --length 100 --width 4 --number 8 --samp 0.1 --epsg 32189 --out "/save/folder"

**Plot_umi.py**
This script plots the saved transect data output from the `Umi_profiles` script. 
The script needs the following inputs:
 - ***--infold***: the path to the folder containing the saved files from the `Umi_profiles`  script, i.e. the extracted transect data from the MNS and MNT solnu and neige.
 - ***--mns***: path to the MNS solnu to plot the map.
 - ***--epsg***: the EPSG code of the geotiff, as GDAL cannot always resolve the code automatically.
 - ***--titre***: A title for the figure, this title will be present in the figure and will be the figure filename.
 - ***--outfile***: Path to the folder where the pdf will be saved.

Example run:

    python Plot_umi.py --infold /path/to/extracted/data/ --mns /path/to/MNS_solnu_3cm_SUD.tif --epsg 2951 --outfile /save/folder --titre "Transect near instruments"

<a name="bash"></a>
# Batch processing
To process the transect extraction automatically, 2 bash scripts were created.
  
**umi_batch.sh**
The bash script passes arguments to the Python processing script `Umi_profiles.py` (see above), runs the Python script and stores the outputs in a folder organised by azimuth.
The script is designed to run over a list of given azimuths for faster processing.
The scripts needs the following obligatory inputs:
 - ***-a***: a list of azimuths, in quotes separated by a space. I.e. `"0 45 90 135"`.
 - ***-x***: the longitude of the centre point of the transect, in WGS84.
 - **-y:** the longitude of the centre point of the transect, in WGS84.
 - **-l:** the length of the transect, in meters.
 - **-w:** the width of the transect, in meters.
 - **-n:** the number of desired transects. Width between transects = width / number of transects.
 - **-s:** the sampling interval along the transect, in meters.
 - **-e:** the EPSG code of the input data.
 - **-f:** the folder containing the DTM and/or DSM data. In this workflow, the folder needs to contain the bare ground DTM and DSM, and the snow DTM and DSM.
 - **-o:** the folder where the outputs will be stored. If the folder doesn't exist, it is created, if it exists the contents are wiped. The script then creates a folder for each azimuth containing the transect information.
 
Example run:

To extract the data from 8 transects across a width of 4 meters (transect every 0.5 m), each transect being 100 meters long, with the array of transects centred at lat/lon 56.559,-76.481 and for 3 azimuths of 0, 45, and 50°:

    ./umi_batch.sh -a "0 45 50" -x -76.481 -y 56.559 -l 100 -w 4 -n 8 -s 0.1 -e 2951 -o /path/to/output/folder -f /path/to/folder/containing/DEM_data

**umi_batch_plot.sh**
The bash script passes arguments to the Python processing script `Plot_umi.py` (see above), runs the Python script and stores the output figures in a pdf file.

The scripts needs the following obligatory inputs:
 - ***-f***: the folder containing the folders output from the `umi_batch.sh` script or the python `Umi_profiles.py`  script organised by azimuth. 
 - ***-m***: the path to the MNS solnu geotiff for plotting.
 - ***-e***: the EPSG code of the MNS solnu geotiff.
 - ***-o***: the folder in which the output pdf will be saved.
 - ***-t***:  A title for the figure, this title will be present in the figure and will be the figure filename.

Example run:

    ./umi_batch_plot.sh -f /path/to/folder/containing/azi_runs -m /path/to/MNS_solnu_3cm_SUD.tif -e 2951 -o /path/to/output/folder -t "Transect_buisson_long"

<a name="tools"></a>
# Processing tools

**Filter_vegetation.py**
Filters the vegetation height raster by selecting pixels between the 90th and 98th percentiles. The script returns the filtered raster at the original resolution and the resampled raster (average of the filtered pixels) at the resolution set by the processing window size.

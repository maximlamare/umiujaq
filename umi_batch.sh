#!/bin/bash
# Run the Umi_profile extract over a list of azimuths as input

OPTIONS=$(getopt -o x:y:f:a:l:w:n:s:e:o: -- "$@")

if [ $? -ne 0 ]; then
  echo "getopt error"
  exit 1
fi

eval set -- $OPTIONS

while true; do
  case "$1" in
    -x) LON="$2" ; shift ;;
    -y) LAT="$2" ; shift ;;
    -f) INFLD="$2" ; shift;;
    -a) AZI="$2" ; shift ;;
    -l) LENGTH="$2" ; shift ;;
    -w) WIDTH="$2" ; shift ;;
    -n)  NB="$2" ; shift ;;
    -s)  SAMP="$2" ; shift ;;
    -e)  EPSG="$2" ; shift ;;
    -o)  OUT="$2" ; shift ;;
    --)        shift ; break ;;
    *)         echo "unknown option: $1" ; exit 1 ;;
  esac
  shift
done

if [ $# -ne 0 ]; then
  echo "unknown option(s): $@"
  exit 1
fi

if [ -z "$LON" ] || [ -z "$LAT" ] || [ -z "$AZI" ] || [ -z "$INFLD" ] \
    || [ -z "$LENGTH" ] || [ -z "$WIDTH" ] || [ -z "$NB" ] \
    || [ -z "$SAMP" ] || [ -z "$EPSG" ] || [ -z "$OUT" ]; then
    echo "Missing arguments"
    exit 1 
fi

# If output directory doesn't exists create it
if [ ! -d "$OUT" ]; then
    mkdir "$OUT"
fi

# Check if empty and empty it if not
if [ -z "$(ls -A $OUT)" ]; then
   :
else
   rm -r $OUT/*
fi

# Loop over the azimuths
for az in $AZI
do
    # Build folders 
    mkdir $OUT/"az_${az}" 

    for file in ${INFLD}/*
    do
        python ./Umi_profiles.py --raster $file --lon $LON --lat $LAT \
        --azi $az --length $LENGTH --width $WIDTH \
        --number $NB --samp $SAMP --epsg $EPSG \
        --out $OUT/"az_${az}"
    done
done

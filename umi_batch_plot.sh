#!/bin/bash
# Batch plot the Umiujaq profiles

OPTIONS=$(getopt -o f:m:e:o:t: -- "$@")

if [ $? -ne 0 ]; then
  echo "getopt error"
  exit 1
fi

eval set -- $OPTIONS

while true; do
  case "$1" in
    -f) INFLD="$2" ; shift;;
    -m) MNS="$2" ; shift ;;
    -e)  EPSG="$2" ; shift ;;
    -o)  OUT="$2" ; shift ;;
    -t) TITRE="$2" ; shift ;;
    --)        shift ; break ;;
    *)         echo "unknown option: $1" ; exit 1 ;;
  esac
  shift
done

if [ $# -ne 0 ]; then
  echo "unknown option(s): $@"
  exit 1
fi

# Run the plotting script for the folders in the input dir
# Copy the output to a temp folder
var=1
mkdir $OUT/tmp

for fld in $(ls -v $INFLD)
do
    python Plot_umi.py --infold $INFLD/$fld --mns $MNS \
    --epsg $EPSG --outfile tmp.pdf --titre $TITRE

    num=$(printf "%04d\n" $var)

    mv $INFLD/$fld/tmp.pdf $OUT/tmp/tmp${num}.pdf

    ((var=var+1))

done

# Merge the pdfs 
pdftk $OUT/tmp/*.pdf cat output $OUT/$TITRE.pdf

# Remove tmp folder
rm -r $OUT/tmp



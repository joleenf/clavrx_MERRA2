#! /bin/sh

YYYY=${1}
MM=${2}
DD=${3}
# Synoptic Run time is not used for constants file.
HH=${4}
export WGET_CMD="wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies"

# NASA Office Note Repository: https://gmao.gsfc.nasa.gov/pubs/office_notes.php
# GMAO Office Note No. 4 (Version 1.2): Lucchesi, R., 2018. File Specification for GEOS-5 FP (Forward Processing)
# https://gmao.gsfc.nasa.gov/pubs/docs/Lucchesi1203.pdf
#
# NOTE: There is only one constants file, stored 
#       https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/GEOS.fp.asm.const_2d_asm_Nx.00000000_0000.V01.nc4
#       "date" 00000000. "time" 0000.  

let YMD=${YYYY}${MM}${DD}
scripts_home="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

set -x

# Getting download_dir from a change directory in wget_all...Not sure this is desirable.
DOWNLOAD_DIR=$(pwd)
download_dir=${1:-$DOWNLOAD_DIR}
echo $download_dir
cd $download_dir

CONSTANTS_FILEPATH=/ships22/cloud/Ancil_Data/clavrx_ancil_data/dynamic/GOES-FP_INPUT/GEOS.fp.asm.const_2d_asm_Nx.00000000_0000.V01.nc4
FALSE_DATE_TARGET_NAME=GEOS.fp.asm.const_2d_asm_Nx.00000000_0000.V01.nc4
REMOTE_FILE=https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/GEOS.fp.asm.const_2d_asm_Nx.00000000_0000.V01.nc4

# check if file already exists on disk
if [ -s ${CONSTANTS_FILEPATH} ]; then
	cp ${CONSTANTS_FILEPATH} ${download_dir}/${FALSE_DATE_TARGET_NAME}
	echo "${CONSTANTS_FILEPATH} already on disc, copying to ${download_dir}/${FALSE_DATE_TARGET_NAME}"
else
    constants_original_filename=$(basename $CONSTANTS_FILEPATH)
    constants_parent_path=$(dirname $CONSTANTS_FILEPATH)
    echo Getting $constants_original_filename to $constants_parent_path
    ${WGET_CMD} --directory-prefix=$constants_parent_path/ ${REMOTE_FILE}
    if [ -s "$CONSTANTS_FILEPATH" ]; then
        # this should be a constants file, maybe just cp to target_name so this does not need to be downloaded every time?
        cp ${CONSTANTS_FILEPATH} ${download_dir}/${FALSE_DATE_TARGET_NAME}
    else 
        echo "ERROR: ${CONSTANTS_FILEPATH} does not exist, even after wget attempt."
    fi
fi

exit

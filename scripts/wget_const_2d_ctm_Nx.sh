#! /bin/sh

YYYY=${1}
MM=${2}
DD=${3}

# MERRA-2 files are identified by their assimilation stream, of which there are 4 (100, 200, 300, and 400).
# The stream a file comes from depends on its date, as referenced on page 13 of:
#
# Location:
# https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf
#
# Title:
# GMAO Office Note No. 9 (Version 1.1)
# MERRA-2: File Specification
#
# Citation:
# Bosilovich, M. G., R. Lucchesi, and M. Suarez, 2016: MERRA-2: File Specification. GMAO
# Office Note No. 9 (Version 1.1), 73 pp, available from
# http://gmao.gsfc.nasa.gov/pubs/office_notes
#
# NOTE: There is only one constants file here, stored in a "1980" subdirectory for the
#       "date" 00000000. The "stream" is defined as 101, so values are hard-coded here
#       regardless of the YYYY, MM, DD values provided.

set -x
let YMD=${YYYY}${MM}${DD}

TARGET_FILE=MERRA2_101.const_2d_ctm_Nx.00000000.nc4
FALSE_DATE_TARGET_NAME=MERRA2_101.const_2d_ctm_Nx.${YMD}.nc4

# check if file already exists on disk
if [ -s ${TARGET_FILE} ]; then
	cp ${TARGET_FILE} 2d_ctm/${FALSE_DATE_TARGET_NAME} 
	echo "${TARGET_FILE} already on disc, copying to 2d_ctm/${FALSE_DATE_TARGET_NAME}"
	exit
fi

wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --content-disposition https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2C0NXCTM.5.12.4/1980/${TARGET_FILE}

if [ -f "$TARGET_FILE" ]; then
    # this should be a constants file, maybe just cp to target_name so this does not need to be downloaded every time?
    cp ${TARGET_FILE} 2d_ctm/${FALSE_DATE_TARGET_NAME}
else 
    echo "${TARGET_FILE} does not exist."
fi

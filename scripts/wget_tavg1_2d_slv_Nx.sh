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
# Define STREAM by date:

set -x
let YMD=${YYYY}${MM}${DD}

scripts_home="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source ${scripts_home}/get_stream.sh
get_stream ${YMD}

TARGET_FILE=MERRA2_${STREAM}.tavg1_2d_slv_Nx.${YYYY}${MM}${DD}.nc4
REANALYSIS=MERRA2_${STREAM}.tavg1_2d_slv_Nx.${YYYY}${MM}${DD}.nc4
BASEURL=https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXSLV.5.12.4

if [ -s "./2d_slv/${TARGET_FILE}" ] || [ -s "./2d_slv/${REANALYSIS}" ]; then
        echo "${TARGET_FILE} exists"
        exit
fi

wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies ${BASEURL}/${YYYY}/${MM}/${TARGET_FILE}

if [ $? != 0 ]; then
        any_stream ${TARGET_FILE} ${BASEURL}
fi

if [ -s "$TARGET_FILE" ]; then
    mv ${TARGET_FILE} 2d_slv/.
else 
    echo "${TARGET_FILE} does not exist."
    exit 1
fi


exit 

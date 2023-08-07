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

#set -x
let YMD=${YYYY}${MM}${DD}

scripts_home="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source ${scripts_home}/get_stream.sh
source ${scripts_home}/wget_exec.sh
get_stream ${YMD}

TARGET_FILE=MERRA2_${STREAM}.inst3_3d_asm_Np.${YYYY}${MM}${DD}.nc4
REANALYSIS=MERRA2_401.inst3_3d_asm_Np.${YYYY}${MM}${DD}.nc4
HTTP_PATH=https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NPASM.5.12.4

if [ -s "3d_asm/${TARGET_FILE}" ] || [ -s "3d_asm/${REANALYSIS}" ];  then
        echo "${TARGET_FILE} exists"
        exit
fi

${WGET_CMD} ${HTTP_PATH}/${YYYY}/${MM}/${TARGET_FILE}

if [ $? != 0 ]; then
        any_stream ${TARGET_FILE} ${HTTP_PATH}
fi

if [ -s "$TARGET_FILE" ]; then
    mv ${TARGET_FILE} 3d_asm/.
else 
    echo "${TARGET_FILE} does not exist."
    exit 1
fi

exit

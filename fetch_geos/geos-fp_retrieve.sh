#! /bin/sh
# Get geos-fp forecast products with this wget
# NASA Office Note Repository: https://gmao.gsfc.nasa.gov/pubs/office_notes.php
# GMAO Office Note No. 4 (Version 1.2): Lucchesi, R., 2018. File Specification for GEOS-5 FP (Forward Processing)
# https://gmao.gsfc.nasa.gov/pubs/docs/Lucchesi1203.pdf

YYYY=${1}
MM=${2}
DD=${3}
SYNOPTIC_RUN=${4}
FILENAME_ID=${5}
DOWNLOAD_PATH=${6}
geos_version="V01"
export WGET_CMD="wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies"

str_size=`echo $SYNOPTIC_RUN | awk '{print length}'`
if [ ${str_size} ==  2 ]; then
        synoptic_run=$SYNOPTIC_RUN
else
        synoptic_run=`printf "%02d" $SYNOPTIC_RUN`
fi

let YMD=${YYYY}${MM}${DD}

scripts_home="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
forecast_url=https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/forecast/Y${YYYY}/M${MM}/D${DD}/H${synoptic_run}
assimilation_url=https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y${YYYY}/M${MM}/D${DD}


case "${FILENAME_ID}" in
	inst3_3d_asm_Np|inst3_3d_asm_Nv) 
		run_valid=${synoptic_run}00
		url=${forecast_url}
		;;
	tavg1_2d_flx_Nx|tavg1_2d_lnd_Nx|tavg1_2d_rad_Nx|tavg1_2d_slv_Nx) 
		run_valid=${synoptic_run}30
		url=${forecast_url}
		;;
	inst3_2d_asm_Nx)
		run_valid=${synoptic_run}00
                url=${assimilation_url}
		;;
esac

case "${FILENAME_ID}" in
	inst3_2d_asm_Nx) PRODUCT_FILE=GEOS.fp.asm.${FILENAME_ID}.${YYYY}${MM}${DD}_${run_valid}.${geos_version}.nc4
                ;;
	*) PRODUCT_FILE=GEOS.fp.fcst.${FILENAME_ID}.${YYYY}${MM}${DD}_${synoptic_run}+${YYYY}${MM}${DD}_${run_valid}.${geos_version}.nc4
		;;
esac
remote_url=${url}/${PRODUCT_FILE}
product_filepath=${DOWNLOAD_PATH}/${PRODUCT_FILE}

if [ -s "${product_filepath}" ]; then
        echo "${TARGET_FILE} exists at ${DOWNLOAD_PATH}"
        exit
fi

echo Downloading $product_filepath
echo from ${remote_url}
${WGET_CMD} --directory-prefix=$DOWNLOAD_PATH/ ${remote_url}

if [ ! -s "$product_filepath" ]; then
    echo "ERROR: ${PRODUCT_FILE} does not exist at ${DOWNLOAD_PATH}."
    exit 1
fi

exit

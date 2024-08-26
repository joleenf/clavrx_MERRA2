#!/bin/bash
# Run this code to pull merra data and run for specified date
# sh  run_merra4clavrx.sh <YYYYmmdd>

echo starting at `date`

source $HOME/.bash_profile

source /etc/profile
module purge
mamba activate merra2_clavrx

set -x
# USER OPTIONS
BIN_DIR=$HOME/clavrx_MERRA2
M2_DIR=${DYNAMIC_ANCIL}/MERRA_INPUT/tmp/
OUT_DIR=${DYNAMIC_ANCIL}/merra2
# END USER OPTIONS


TMPDIR=${BIN_DIR}/tmp
mkdir -p $TMPDIR
cd $TMPDIR || (hostname;echo \"could not access $TMPDIR\"; exit 1)

INPUT_DATE=${1:-20210801}

YYYY=${INPUT_DATE:0:4}
MM=${INPUT_DATE:4:2}
DD=${INPUT_DATE:6:2}

M2_DIR=${M2_DIR}${YYYY}/${YYYY}_${MM}_${DD}
mkdir -p $M2_DIR

if [ -d "${TMPDIR}/out" ]
then
    rm -rf ${TMPDIR}/out
fi

echo Date from FILELIST: ${INPUT_DATE}
# Create tmp subdirectories for files
cd ${M2_DIR}

set +x
echo "Running wget_all.sh -w $M2_DIR ${YYYY} ${MM} ${DD}"
sh ${BIN_DIR}/scripts/wget_all.sh -w $M2_DIR ${YYYY} ${MM} ${DD}

# Run merra conversion code for clavrx
echo "Running ${BIN_DIR}/merra_for_clavrx.py ${INPUT_DATE}"
python ${BIN_DIR}/merra_for_clavrx.py ${INPUT_DATE}
#
# clean up
#M2_DIR=`dirname $M2_DIR`
echo $M2_DIR

rm -rfv $M2_DIR
#echo finished at: `date`

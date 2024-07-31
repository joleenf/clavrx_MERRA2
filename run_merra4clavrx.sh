#!/bin/bash
# Run this code to pull merra data and run for specified date
# sh  run_merra4clavrx.sh <YYYYmmdd>

echo starting at `date`

source $HOME/.bash_profile

set -x
# USER OPTIONS
BIN_DIR=$HOME/clavrx_MERRA2
M2_DIR=${DYNAMIC_ANCIL}/MERRA_INPUT/tmp/
OUT_DIR=${DYNAMIC_ANCIL}/merra2
# END USER OPTIONS

source /etc/profile
module purge
module load miniconda

mamba activate merra2_clavrx

conda env list

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

sh ${BIN_DIR}/scripts/wget_all.sh -w $M2_DIR ${YYYY} ${MM} ${DD}

# Run merra conversion code for clavrx
python ${BIN_DIR}/merra_for_clavrx.py ${INPUT_DATE}
#
# clean up
#M2_DIR=`dirname $M2_DIR`
echo $M2_DIR

rm -rfv $M2_DIR
#echo finished at: `date`

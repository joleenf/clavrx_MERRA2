#!/bin/bash
echo starting at `date`

# USER OPTIONS
BIN_DIR=/home/bhoover/clavrx-merra2/merra2
M2_DIR=/home/bhoover/clavrx-merra2/MERRA2_FILES
# END USER OPTIONS

source /etc/profile
module purge
#module load shellb3      # Does not exist on solar4
module load miniconda

source activate bhoover-merra2_clavrx

conda env list

TMPDIR=${BIN_DIR}/tmp
mkdir -p $TMPDIR
cd $TMPDIR || (hostname;echo \"could not access $TMPDIR\"; exit 1)

INPUT_DATE=20200414

if [ -d "${TMPDIR}/out" ]
then
    rm -rf ${TMPDIR}/out
fi

echo Date from FILELIST: ${INPUT_DATE}
# Create tmp subdirectories for files
#mkdir -p 3d_ana
#mkdir -p 3d_asm
#mkdir -p 2d_flx
#mkdir -p 2d_slv
#mkdir -p 2d_rad
#mkdir -p 2d_lnd
#mkdir -p 2d_asm
#mkdir -p 2d_ctm
# Copy files for INPUT_DATE into tmp subdirectories
cp ${M2_DIR}/inst6_3d_ana_Nv/MERRA2*.inst6_3d_ana_Nv.${INPUT_DATE}.nc4 3d_ana/.
cp ${M2_DIR}/inst6_3d_ana_Np/MERRA2*.inst6_3d_ana_Np.${INPUT_DATE}.nc4 3d_ana/.
cp ${M2_DIR}/inst3_3d_asm_Np/MERRA2*.inst3_3d_asm_Np.${INPUT_DATE}.nc4 3d_asm/.
cp ${M2_DIR}/tavg1_2d_flx_Nx/MERRA2*.tavg1_2d_flx_Nx.${INPUT_DATE}.nc4 2d_flx/.
cp ${M2_DIR}/tavg1_2d_slv_Nx/MERRA2*.tavg1_2d_slv_Nx.${INPUT_DATE}.nc4 2d_slv/.
cp ${M2_DIR}/tavg1_2d_rad_Nx/MERRA2*.tavg1_2d_rad_Nx.${INPUT_DATE}.nc4 2d_rad/.
cp ${M2_DIR}/tavg1_2d_lnd_Nx/MERRA2*.tavg1_2d_lnd_Nx.${INPUT_DATE}.nc4 2d_lnd/.
cp ${M2_DIR}/inst1_2d_asm_Nx/MERRA2*.inst1_2d_asm_Nx.${INPUT_DATE}.nc4 2d_asm/.
cp ${M2_DIR}/const_2d_ctm_Nx/MERRA2_101.const_2d_ctm_Nx.00000000.nc4 2d_ctm/MERRA2_101.const_2d_ctm_Nx.${INPUT_DATE}.nc4
# Run merra24clavrx.py
python -u ${BIN_DIR}/merra24clavrx.py ${INPUT_DATE}
#
# clean up
#cd /scratch
#rm -rfv $TMPDIR
#
#echo finished at: `date`

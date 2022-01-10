#!/bin/bash
echo starting at `date`

# USER OPTIONS
BIN_DIR=/home/clavrx_ops/clavrx-MERRA2/merra2
M2_DIR=/home/clavrx_ops/clavrx-MERRA2/MERRA2_FILES
# END USER OPTIONS

source /etc/profile
module purge
#module load shellb3      # Does not exist on solar4
module load miniconda

source activate merra2_clavrx

conda env list

TMPDIR=${BIN_DIR}/tmp
mkdir -p $TMPDIR
cd $TMPDIR || (hostname;echo \"could not access $TMPDIR\"; exit 1)

INPUT_DATE=20210801

if [ -d "${TMPDIR}/out" ]
then
    rm -rf ${TMPDIR}/out
fi

echo Date from FILELIST: ${INPUT_DATE}
# Create tmp subdirectories for files
mkdir -p 3d_ana
mkdir -p 3d_asm
mkdir -p 2d_flx
mkdir -p 2d_slv
mkdir -p 2d_rad
mkdir -p 2d_lnd
mkdir -p 2d_asm
mkdir -p 2d_ctm
# Copy files for INPUT_DATE into tmp subdirectories
cp ${M2_DIR}/3d_ana/MERRA2*.inst6_3d_ana_Nv.${INPUT_DATE}.nc4 3d_ana/.
cp ${M2_DIR}/3d_ana/MERRA2*.inst6_3d_ana_Np.${INPUT_DATE}.nc4 3d_ana/.
cp ${M2_DIR}/3d_asm/MERRA2*.inst3_3d_asm_Np.${INPUT_DATE}.nc4 3d_asm/.
cp ${M2_DIR}/2d_flx/MERRA2*.tavg1_2d_flx_Nx.${INPUT_DATE}.nc4 2d_flx/.
cp ${M2_DIR}/2d_slv/MERRA2*.tavg1_2d_slv_Nx.${INPUT_DATE}.nc4 2d_slv/.
cp ${M2_DIR}/2d_rad/MERRA2*.tavg1_2d_rad_Nx.${INPUT_DATE}.nc4 2d_rad/.
cp ${M2_DIR}/2d_lnd/MERRA2*.tavg1_2d_lnd_Nx.${INPUT_DATE}.nc4 2d_lnd/.
cp ${M2_DIR}/2d_asm/MERRA2*.inst1_2d_asm_Nx.${INPUT_DATE}.nc4 2d_asm/.
cp ${M2_DIR}/2d_ctm/MERRA2_101.const_2d_ctm_Nx.${INPUT_DATE}.nc4 2d_ctm/MERRA2_101.const_2d_ctm_Nx.${INPUT_DATE}.nc4
# Run merra24clavrx.py
python -u ${BIN_DIR}/merra24clavrx_brett.py ${INPUT_DATE}
#
# clean up
#cd /scratch
#rm -rfv $TMPDIR
#
#echo finished at: `date`

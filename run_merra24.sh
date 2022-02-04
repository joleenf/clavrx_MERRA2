#!/bin/bash
set -x
# USER OPTIONS
BIN_DIR=$HOME/clavrx_MERRA2
DATA_PATH=/data/clavrx_ops/MERRA_INPUT/
START_DATE=20210801
END_DATE=20210801

LOG_FILE=$HOME/logs/merra_archive/s${START_DATE}_e${END_DATE}run.log

source /etc/profile
module purge
module load miniconda

source activate merra2_clavrx

TMPDIR=${DATA_PATH}/tmp
mkdir -p $TMPDIR
cd $TMPDIR || (hostname;echo \"could not access $TMPDIR\"; exit 1)

if [ -d "${TMPDIR}/out" ]
then
    rm -rf ${TMPDIR}/out
fi

start=$(date -d $START_DATE +%Y%m%d)
end=$(date -d $END_DATE +%Y%m%d)

while [[ $start -le $end ]]
do
	year=${start:0:4}
	month="${start:4:2}"
	day="${start:6:2}"
        sh ${BIN_DIR}/scripts/wget_all.sh -w ${TMPDIR} ${year} ${month} ${day}
        start=$(date -d"$start + 1 day" +"%Y%m%d")
	python -u ${BIN_DIR}/merra24clavrx.py ${start} -i ${TMPDIR}  >> $LOG_FILE 2>&1
	# rm -rfv ${TMPDIR}/${start}/${year}_${month}_${day}
done

# clean up
#cd /scratch
#echo finished at: `date`

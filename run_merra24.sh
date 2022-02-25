#!/bin/bash
#set -x
# USER OPTIONS
BIN_DIR=$HOME/clavrx_MERRA2
DATA_PATH=/data/clavrx_ops/MERRA_INPUT/
START_DATE=20210317
END_DATE=20210317

trap finish EXIT
finish() {
	if [ -z $YEAR_DIR ]
	then
		rm -rfv ${YEAR_DIR}
	fi
}


LOG_FILE=$HOME/logs/merra_archive/s${START_DATE}_e${END_DATE}run.log

source /etc/profile
source ~/.bashrc
conda activate merra2_clavrx

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
        YEAR_DIR=${DATA_PATH}/tmp/${year}  #  Not ideal?? merra code appends year to end of input directory given with -i flag.
        sh ${BIN_DIR}/scripts/wget_all.sh -w ${YEAR_DIR} ${year} ${month} ${day}

	# make sure all data is available
	count=`find ${TMPDIR} -name "*${year}${month}${day}*.nc4" | wc -l`
	if [ $count -lt 9 ]; then
		cmd=`date +"ERROR: ($0=>%Y-%m-%d %H:%M:%S) Missing Input $year ${month} ${day}"`
                echo $cmd
	else
		find ${TMPDIR} -name "*${year}${month}${day}*.nc4"
	fi

	python -u ${BIN_DIR}/merra24clavrx.py ${start} -v -i ${TMPDIR} >> $LOG_FILE 2>&1
        start=$(date -d"$start + 1 day" +"%Y%m%d")
	rm -rfv ${YEAR_DIR}
	# unset does not get "$"
	unset YEAR_DIR
done

start=$(date -d $START_DATE +%Y%m%d)
while [[ $start -le $end ]]
do
	yy=${start:2:2}
	year=${start:0:4}
        month="${start:4:2}"
        day="${start:6:2}"
	cmd="find /apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2/${year} -name merra.${yy}${month}${day}*_F*.hdf -print"
	echo $cmd

	out_count=`$cmd | wc -l`
	if [ $? -eq 0 ]; then
	    if [ ${out_count} != 4 ]; then
                cmd=`date +"ERROR: ($0=>%Y-%m-%d %H:%M:%S) Missing Output from  $yy ${month} ${day}"`
                echo $cmd
		exit 1
	    fi
    else
	    cmd=`date +"ERROR: ($0=>%Y-%m-%d %H:%M:%S) No Output from  $yy ${month} ${day}"`
            echo $cmd
	    exit 1
    fi
    echo "Success $yy ${month} ${day}"
    $cmd
    start=$(date -d"$start + 1 day" +"%Y%m%d")
done
# clean up
#cd /scratch
#echo finished at: `date`

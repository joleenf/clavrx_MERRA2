#!/bin/bash
export PS4=' ${DATETIME_NOW} line:${LINENO} function:${FUNCNAME[0]:+${FUNCNAME[0]}() } cmd: ${BASH_COMMAND} result: '
#
# Requires the merra2_clavrx environment.
#
# Environment Variables:
#   BIN_DIR               Location of this script, merra24clarx.py code and the scripts sub-directory.
#   LOG_DIR               Location of log directory.  Path will be created if not already built.
#   DATA_PATH             MERRA2 input data tmp directory (script creates DATA_PATH tree)
#   START_DATE            First date of MERRA2 data to process
#   END_DATE              Last date of MERRA2 data to process
#
# Log files are placed in
#   $HOME/logs/merra_archive/s${START_DATE}_e${END_DATE}run.log
#       Contains the run information for both the bash script and python code
#   $HOME/logs/merra_archive/inventory_${START_DATE:0:4}_${START_DATE:4:2}.log
#       Contains one line completion messages for input data and final product files.

set -e
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BIN_DIR=$HOME/clavrx_MERRA2
LOG_DIR=$HOME/logs/merra_archive
machine=`uname -a | awk -F" " '{print $2}'`
machine=`echo ${machine%%.*}`

: <<=cut
=pod

=head1 NAME


    run_merra24.sh - Run the python code merra24clavrx.py using a start and end date.

=head1 SYNOPSIS

    sh run_merra24.sh <OPTIONS> <start_date:YYYYMMDD> <end_date:YYYYMMDD>
    example: sh run_merra24.sh -s 20200101 20200102

      where: start_date is the first date to run in YYYYMMDD format
	     end_date   is the last date to run in YYYYMMDD format

   Recognized optional command line arguments
      -s  -- save the input data (default: not saved)
      -h  -- Display usage message


=head1 DESCRIPTION

    Runs the series of dates from start date to end date.  First the MERRA2 input files
    are downloaded and checked for readability.  Then, a check is performed to make sure
    all files have downloaded.  Next, merra24clavrx.py is executed to produce the reanalysis
    model files which can be used as in put to the CLAVRx cloud algorithm.  Finally,
    a check is done to verify that 4 files have been created and can at least be listed by hdp.
    Temporary input directory and input files are removed.

=head2 Requirements

    merra2_clavrx enviroment should be made active.
    scripts/wget_all.sh and support scripts
    python/test_dataset.py

    In addition, script environment variables should be changed as needed by user.
     
      Requires the merra2_clavrx environment.
     
      Environment Variables:
        BIN_DIR               Location of this script, merra24clarx.py code and the scripts sub-directory.
        LOG_DIR               Location of log directory.  Path will be created if not already built.
        DATA_PATH             MERRA2 input data tmp directory (script creates DATA_PATH tree)
        START_DATE            First date of MERRA2 data to process
        END_DATE              Last date of MERRA2 data to process
     
=cut


if [ "$machine" == "vor" ]; then
	if [ "$USER" == "clavrx_ops" ]; then
        	DATA_PATH=/apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/MERRA_INPUT
		OUT_PATH=/apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/
	else
		DATA_PATH=/data/Personal/$USER/MERRA_INPUT
		OUT_PATH=/data/Personal/$USER/test_JF_merra2/clavrx_ancil_data/dynamic/merra2/
	fi
else
	DATA_PATH=/data/clavrx_ops/MERRA_INPUT
	OUT_PATH=/apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2
fi


delete_input=true
while getopts "h|s" flag; do
        case "$flag" in
		s) delete_input=false;;
                h) `/bin/pod2usage $0`
                   exit;;
        esac
done

START_DATE=${@:$OPTIND:1}
END_DATE=${@:$OPTIND+1:1}

if [ -z $START_DATE ] || [ -z $END_DATE ];then
	`/bin/pod2usage $0`
  	 exit
fi

function check_output {
# this section checks if output has been created.
    out_count=0
    find ${OUT_PATH}/${year} -name "merra.${year:2,2}${month}${day}_F*.hdf" -print | while read -r hdf;do
        out_count=$(( out_count + 1))
        hdp list $hdf
	echo "Out count is ${out_count}"
        if [ "$?" -ne "0" ]; then
            cmd=`date +"ERROR: ($0=>%Y-%m-%d %H:%M:%S) Reading $hdf"`
  	    echo $cmd
        fi
    done

    if [ "${out_count}" -ne "4" ]; then
        date +"ERROR: ($0=>%Y-%m-%d %H:%M:%S) Incomplete Merra Output ($out_count) ${year} ${month} ${day}" >> $INVENTORY_FILE
    else
        echo "Success ${year} ${month} ${day}"
        echo "${year} ${month} ${day} Merra Output Complete." >> $INVENTORY_FILE
    fi
}

trap finish EXIT
finish() {
	if [[ -z $YEAR_DIR  &&  "${delete_input}" = true ]];
	then
		cmd="rm -rfv ${YEAR_DIR}"
		eval $cmd
	fi
}


mkdir -p $LOG_DIR

LOG_FILE=${LOG_DIR}/s${START_DATE}_e${END_DATE}run.log
INVENTORY_FILE=${LOG_DIR}/inventory_${START_DATE:0:4}_${START_DATE:4:2}.log

echo "Writing logs to ${LOG_FILE} and ${INVENTORY_FILE}"

source ~/.bashrc
conda activate merra2_clavrx

TMPDIR=${DATA_PATH}/tmp
mkdir -p $TMPDIR
cd $TMPDIR || (hostname;echo \"could not access $TMPDIR\"; exit 1)

start_date=$(date -d $START_DATE +%Y%m%d)
end_date=$(date -d $END_DATE +%Y%m%d)

# this section gets the data and runs the python code.
while [[ $start_date -le $end_date ]]
do
	year=${start_date:0:4}
	month="${start_date:4:2}"
	day="${start_date:6:2}"
        #YEAR_DIR=${TMPDIR}/${year}/${year}_${month}_${day}  #  Not ideal?? merra code appends year to end of input directory given with -i flag.
        YEAR_DIR=${TMPDIR}/${year}  #  Not ideal?? merra code appends year to end of input directory given with -i flag.
	mkdir -p $YEAR_DIR

        sh ${BIN_DIR}/scripts/wget_all.sh -w ${YEAR_DIR} ${year} ${month} ${day}

	# make sure all data is available
	count=`find ${YEAR_DIR} -name "*${year}${month}${day}*.nc4" | wc -l`
	echo $count
	find ${YEAR_DIR} -name "*${year}${month}${day}*.nc4"

	if [ "$count" -lt "9" ]; then
		cmd=`date +"ERROR: ($0=>%Y-%m-%d %H:%M:%S) Missing Input $year ${month} ${day}"`
                start_date=$(date -d"$start_date + 1 day" +"%Y%m%d")
		break
	else
		find ${TMPDIR} -name "*${year}${month}${day}*.nc4"
		echo ${year} ${month} ${day} Input Complete >> $INVENTORY_FILE
	fi

	python -u ${BIN_DIR}/merra24clavrx.py ${start_date} -d ${OUT_PATH} -vvvv -i ${TMPDIR} >> $LOG_FILE 2>&1
	check_output
        start_date=$(date -d"$start_date + 1 day" +"%Y%m%d")
	if [ "${delete_input}" = true ]; then
	    cmd="rm -rfv ${YEAR_DIR}"
	    eval $cmd
        fi
	# unset does not get "$"
	unset YEAR_DIR
done

exit

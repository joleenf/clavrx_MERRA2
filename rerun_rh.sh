function usage() {
cat << EndOfMessage

    Usage: sh $0 YYYY MM <full path of script to run and arguments>"

    Calculate number of days in month given year and date.
    Call run_merra24.sh in screen to run full month.

    YYYY:  use 4-digit date to reduce error in figuring out century.
    MM:    1 or 2-digit month number
    script_path:  Script to run that takes a date as first argument (provided
                  by this code) and any arguments for the secondary script.

    Suggestion:  If only a few days are needed run_merra24.sh directly.

EndOfMessage
    echo $VAR
    exit

}


source ~/.bashrc
conda activate base
conda activate merra2_clavrx
export PS4='L${LINENO}: '
export PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
#set -x

BIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_DIR=$HOME/logs/merra_archive
M2_DIR=/ships22/cloud/Ancil_Data/clavrx_ancil_data/dynamic/MERRA_INPUT/tmp/
MERRA_OUT=/ships22/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2
export PYTHONPATH=$PYTHON_PATH:$BIN_DIR

mkdir -p $LOG_DIR

 
dated_dir() {
	in_date="$@"
	YYYY=${in_date:0:4}
        MM=${in_date:4:2}
        DD=${in_date:6:2}
	export DATED_DIR=${M2_DIR}${YYYY}/${YYYY}_${MM}_${DD}
	echo $DATED_DIR
}
export -f dated_dir

remove_input_files() {
	in_date="$@"
	dated_dir $in_date
	echo "Removing $DATED_DIR and contents"
	rm -rf $DATED_DIR
}
export -f remove_input_files

list_results_of_repair() {
	# This lists the range of rh after repair
	in_date="$@"
        python ${BIN_DIR}/conversions/list_bad_rh.py --dt $in_date --merra_dir $MERRA_OUT --s 00 --rh_range
        python ${BIN_DIR}/conversions/list_bad_rh.py --dt $in_date --merra_dir $MERRA_OUT --s 06 --rh_range
        python ${BIN_DIR}/conversions/list_bad_rh.py --dt $in_date --merra_dir $MERRA_OUT --s 12 --rh_range
        python ${BIN_DIR}/conversions/list_bad_rh.py --dt $in_date --merra_dir $MERRA_OUT --s 18 --rh_range
}
export -f list_results_of_repair

verify_results() {
	# This verifies the file was repaired
	in_date="$@"
        python ${BIN_DIR}/conversions/list_bad_rh.py --dt $in_date --merra_dir $MERRA_OUT --s 00 --early_exit
	echo $?
        python ${BIN_DIR}/conversions/list_bad_rh.py --dt $in_date --merra_dir $MERRA_OUT --s 06 --early_exit
        echo $?
        python ${BIN_DIR}/conversions/list_bad_rh.py --dt $in_date --merra_dir $MERRA_OUT --s 12 --early_exit
	echo $?
        python ${BIN_DIR}/conversions/list_bad_rh.py --dt $in_date --merra_dir $MERRA_OUT --s 18 --early_exit
	echo $?
}
export -f verify_results


run_rh_for_day() {
    in_date="$@"
    python ${BIN_DIR}/conversions/fix_rh.py $in_date
    verify_results $in_date
}
export -f run_rh_for_day

get_merra_for_day() {
	in_date="$@"
        YYYY=${in_date:0:4}
        MM=${in_date:4:2}
        DD=${in_date:6:2}

        dated_dir $in_date 
	echo $DATED_DIR
        mkdir -p $DATED_DIR

        cmd="/bin/bash ${BIN_DIR}/scripts/wget_all.sh -k const_2d_ctm_Nx -w $DATED_DIR ${YYYY} ${MM} ${DD}"
        echo $cmd
        eval $cmd
        cmd="/bin/bash ${BIN_DIR}/scripts/wget_all.sh -k inst6_3d_ana_Np -w $DATED_DIR ${YYYY} ${MM} ${DD}"
        echo $cmd
        eval $cmd
}
export -f get_merra_for_day

rerun_rh_for_month() {
    [[ $# -lt 2 ]] && usage
    YEAR=$1
    MONTH=$2

    test -z $YEAR && usage 
    test -z $MONTH && usage 

    len_year=${#YEAR}
    [[ $len_year -lt 4 ]] && usage
    
    str_size=`echo $MONTH | awk '{print length}'`
    if [ ${str_size} ==  2 ]; then
            month=$MONTH
    else
            month=`printf "%02d" $MONTH`
    fi
    screen_name=`date -d ${YEAR}-${month}-01 +"%B_%Y"`

    ndays=`cal ${MONTH} ${YEAR} | awk 'NF {DAYS = $NF}; END {print DAYS}'`

    # call run merra for this full month
    start_day=${YEAR}${month}01
    end_day=${YEAR}${month}${ndays}

    start_date=$(date -d $start_day +%Y%m%d)
    end_date=$(date -d $end_day +%Y%m%d)

    while [[ $start_date -le $end_date ]];
    do
        # Here is where to run the python....
	get_merra_for_day ${start_date}
	run_rh_for_day ${start_date}
        start_date=$(date -d"$start_date + 1 day" +"%Y%m%d")
    done

    if [ -d "${DATED_DIR}" ]
    then
        rm -rf ${DATED_DIR}
    fi
}

export -f rerun_rh_for_month



run_rh_over_years() {
    for year in {2000..1980};
    do
       for i in {1..12};
       do
          screen_name=run$year_$i
          echo $screen_name
          cmd="/bin/bash rerun_rh_for_month $year $i"
          echo $cmd
          eval $cmd
       done
    done
    }
export -f run_rh_over_years

  # for testing, forward to a sub-function
  # e.g. ./rerun_rh.sh run_rh_over_years
  # e.g. ./rerun_rh.sh rerun_rh_for_month YYYY MM 
  # e.g. ./rerun_rh.sh get_merra_for_day YYYYmmdd
  # e.g. ./rerun_rh.sh run_rh_for_day YYYYmmdd
  # e.g. ./rerun_rh.sh remove_input_files YYYYmmdd
  # e.g. ./rerun_rh.sh dated_dir YYYYmmdd
  # e.g. ./rerun_rh.sh verify_results YYYYmmdd
  # e.g. ./rerun_rh.sh list_results_of_repair YYYYmmdd

"$@"

exit

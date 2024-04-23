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
export PS4='L${LINENO}: '
export PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
#set -x

BIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_DIR=$HOME/logs/merra_archive
M2_DIR=/ships22/cloud/Ancil_Data/clavrx_ancil_data/dynamic/MERRA_INPUT/tmp/
export PYTHONPATH=$PYTHON_PATH:$BIN_DIR

mkdir -p $LOG_DIR

 
run_rh_for_day() {
    in_date="$@"
    conda activate base
    conda activate merra2_clavrx
    capture_scratch=`python ${BIN_DIR}/conversions/fix_rh.py $in_date`
    capture_scratch=`echo $capture_scratch | awk '{print $3}'`
    echo $capture_scratch
}
export -f run_rh_for_day

get_merra_for_day() {
	in_date="$@"
        YYYY=${in_date:0:4}
        MM=${in_date:4:2}
        DD=${in_date:6:2}

        M2_DIR=${M2_DIR}${YYYY}/${YYYY}_${MM}_${DD}
        mkdir -p $M2_DIR

        cmd="/bin/bash ${BIN_DIR}/scripts/wget_all.sh -k const_2d_ctm_Nx -w $M2_DIR ${YYYY} ${MM} ${DD}"
        echo $cmd
        eval $cmd
        cmd="/bin/bash ${BIN_DIR}/scripts/wget_all.sh -k inst6_3d_ana_Np -w $M2_DIR ${YYYY} ${MM} ${DD}"
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

    if [ -d "${M2_DIR}" ]
    then
        rm -rf ${M2_DIR}
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

"$@"

exit

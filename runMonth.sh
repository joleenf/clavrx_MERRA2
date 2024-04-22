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

export PS4='L${LINENO}: '
export PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
#set -x

bin_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_DIR=$HOME/logs/merra_archive

mkdir -p $LOG_DIR

[[ $# -lt 3 ]] && usage
YEAR=$1
shift
MONTH=$1
shift
secondary_script=$1
secondary_script_args="$@"

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

#screen -dm -S $screen_name /bin/bash $bin_dir/run_merra24.sh $start_day $end_day
while [[ $start_date -le $end_date ]];
do
    /bin/bash $secondary_script $start_date $secondary_script_args
    start_date=$(date -d"$start_date + 1 day" +"%Y%m%d")
done

echo "/bin/bash $bin_dir/$secondary_script secondary_script_args for $start_day $end_day"
sh $bin_dir/scripts/count_inventory.sh $YEAR $MONTH >> $LOG_DIR/inventory_${YEAR}.log
exit

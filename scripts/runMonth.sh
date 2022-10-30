function usage() {
cat << EndOfMessage

    Usage: sh $0 YYYY MM"

    Calculate number of days in month given year and date.
    Call run_merra24.sh in screen to run full month.

    YYYY:  use 4-digit date to reduce error in figuring out century.
    MM:    1 or 2-digit month number

    Suggestion:  If only a few days are needed run_merra24.sh directly.

EndOfMessage
    echo $VAR
    exit

}

#set -x

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_DIR=$HOME/logs/merra_archive

mkdir -p $LOG_DIR

YEAR=$1
MONTH=$2
test -d $YEAR && usage 
test -d $MONTH && usage 
len_year=`expr length "$YEAR"`
if [ ${#YEAR} -ne 4 ]; then
	echo \n Year must be 4-digits!!!
	usage
fi

month=`printf "%02d" $MONTH`

screen_name=`date -d ${YEAR}-${month}-01 +"%B_%Y"`

ndays=`cal ${MONTH} ${YEAR} | awk 'NF {DAYS = $NF}; END {print DAYS}'`

# call run merra for this full month
start_day=${YEAR}${month}01
end_day=${YEAR}${month}${ndays}

LOG_FILE=$LOG_DIR/month_${start_day}_${end_day}.log
echo $LOG_FILE

# send stderr and stdout to log file
exec 1>>$LOG_FILE
exec 2>>$LOG_FILE


/bin/bash $scripts_dir/run_merra24.sh $start_day $end_day
echo "/bin/bash $scripts_dir/run_merra24.sh $start_day $end_day"
sh $scripts_dir/count_inventory.sh $YEAR $MONTH >> $LOG_DIR/inventory_${YEAR}_${month}.log
exit

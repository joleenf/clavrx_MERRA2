#!/bin/bash
function usage {
cat << EndOfMessage

    Usage: sh $0 [options] YYYY MM <PRODUCT_ID> <DATA_PATH>

    Example: sh $0 2023 12 gfs /data/www/gfs

    Count files for month in product data directory and
    verify 4 output files per day have been created for each
    day of month under inspection.

       data directory default: $DATA_PATH
       product_ID: ${PRODUCT_ID}

EndOfMessage
    $VAR
    exit
}

oops () {
    echo "Incorrect input"
    usage
}


DATA_PATH="/ships22/cloud/Ancil_Data/clavrx_ancil_data/dynamic/geos/"
PRODUCT_ID=geosfp
trim=`date +"(${0}=>%Y%m%d %H:%M:%S) "`
TODAY=$(date +"%Y%m%d")
YEAR=$(date -d "$TODAY" +"%Y")
MONTH=$(date -d "$TODAY" +"%m")

YEAR=${1:-$YEAR}
MONTH=${2:-$MONTH}
PRODUCT_ID=${3:-$PRODUCT_ID}
DATA_PATH=${4:-$DATA_PATH}
if [ "$#" -lt 2 ]; then
	oops
fi

month="${MONTH##*(0)}"

[ -z "${month}" ] && oops || month=`printf "%02d" $((10#$MONTH))`

# call run merra for this full month
ndays=`cal ${MONTH} ${YEAR} | awk 'NF {DAYS = $NF}; END {print DAYS}'`
start_day=${YEAR}${month}01

DAYS_AGO=2
dayago=$(date -d "$TODAY ${DAYS_AGO} days ago" +"%Y%m%d")
curr_month=$(date -d "$TODAY ${DAYS_AGO} days ago" +"%m")
curr_day=$(date -d "$TODAY ${DAYS_AGO}  days ago" +"%d")
curr_year=$(date -d "$TODAY ${DAYS_AGO}  days ago" +"%Y")

# check if entered date is ahead of schedule
if [ "${YEAR}" -gt "${curr_year}" ]; then
	echo "Year ($YEAR) entered is in the future and no files are expected."
	exit 1
elif [ "${YEAR}" -eq "${curr_year}" ]; then
	if [ "${month}" -gt "${curr_month}" ]; then
		echo "${YEAR} ${month} are in the future and no files are expected."
		exit 1
	fi
fi
if [ $curr_month == $month ];
then
   end_day=${YEAR}${month}${curr_day}
else
   end_day=${YEAR}${month}${ndays}
fi

if [ -z $YEAR ] || [ -z $MONTH ];then
         usage
         exit
fi

month=`printf "%02d" $((10#$MONTH))`

str_month=`date -d ${YEAR}${month}01 +"%B %Y"`

data_path=${DATA_PATH}/${YEAR}

count=0
while [[ $start_day -le $end_day ]];
do
	for synoptic in 00 06 12 18; 
	do
		if [ "$PRODUCT_ID" == "cfsr" ]; then
			filename="$PRODUCT_ID.${start_day:2}${synoptic}_F006.hdf"
		elif [ "$PRODUCT_ID" == "gfs" ]; then
			filename="$PRODUCT_ID.${start_day:2}${synoptic}_F012.hdf"
		else
			filename="$PRODUCT_ID.${start_day:2}${synoptic}_F000.hdf"
		fi
		fcount=`find ${data_path} -name $filename | wc -l`
		if [ $fcount == 0 ]; then
			error_msg="${trim}${data_path}/$filename"
			echo $error_msg
			count=$((count + 1))
		fi
	done
        start_day=$(date -d"$start_day + 1 day" +"%Y%m%d")
done

if [ $count == 0 ]; then
	msg=`date +"(${0}=>%Y%m%d %H:%M:%S) No files missing $YEAR/${month} in $DATA_PATH for $PRODUCT_ID "`
        echo $msg
else
    echo "The list generated represents file patterns not found in ${DATA_PATH}."
fi

exit

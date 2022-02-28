#!/bin/bash
#
# Environment Variables:
#   MERRA2_DATA_PATH      Location merra2 output hdf data from merra24clavrx.py.
#   
# Month and Year must be entered together.
#   MONTH                 Month to check
#   YEAR                  Year to check
#

function usage() {
cat << EndOfMessage

    Usage: sh $0 [options] YYYY MM DD"

    Download data from GES DISC based on date and key"

    Options:
    	-w merra2 data directory 
           (default is pwd).
   	-h Display this usage information

EndOfMessage
    echo $VAR
    exit

}

DATA_PATH=/apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2

[ $# -eq 0 ] && usage
while getopts "w:h" flag; do
	case "$flag" in
		w) DATA_PATH=$OPTARG;;
		h) usage;;
	esac
done

YEAR=${@:$OPTIND:1}
MONTH=${@:$OPTIND+1:1}

month=`printf "%02d" $MONTH`

str_month=`date -d ${YEAR}-${month}-01 +"%B %Y"`

DATA_PATH=${DATA_PATH}/${YEAR}

ndays=`cal ${MONTH} ${YEAR} | awk 'NF {DAYS = $NF}; END {print DAYS}'`
ntotal=$(( $ndays*4 ))

nfiles=`find ${DATA_PATH} -name "merra.${YEAR:2:2}${month}*.hdf" | wc -l`

echo "===="
if [ ${nfiles} -lt ${ntotal} ]; then
	echo "ERROR: Only ${nfiles} files in ${DATA_PATH}. ${ntotal} are expected for $str_month"
	echo "===="
	exit 1
elif [ ${nfiles} -gt ${ntotal} ]; then
	echo "ERROR: $DATA_PATH has (${nfiles}) files. ${ntotal} expected for $str_month"
	echo "===="
        exit 1 
else
	echo "COMPLETE: There are a complete set of files for $str_month: ${nfiles}"
	echo "===="
fi

exit

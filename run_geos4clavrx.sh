#!/bin/bash
# Run this code to pull merra data and run for specified date
function usage() {
cat << EndOfMessage

    =================================================================
    Usage: sh $0 CCYYmmdd HH 
    OR:
           sh $0 <# days ago from today> HH

    Retrieve GOES data for the given CCYYmmdd and HH (00, 06, 12, 18)

    Arg#1
    --------
    CCYYmmdd:  full year month day with no spaces or dashes.
               (Use the string X for the default value of yesterday)
    OR
    n:  number of days ago from today (for reprocessing if needed)
    --------

    Arg#2:  Required!
    HH:    2-digit Hour
    =================================================================

EndOfMessage
    echo $VAR
    exit

}

echo starting at `date`

# USER OPTIONS
BIN_DIR=$HOME/clavrx_MERRA2
M2_DIR=/ships22/cloud/Ancil_Data/clavrx_ancil_data/dynamic/GEOS/tmp/
M2_DIR="/ships22/cloud/scratch/7day/GEOS-FP_INPUT"
OUT_DIR=${DYNAMIC_ANCIL}/geos
# END USER OPTIONS

source /etc/profile
source $HOME/.bash_profile
module purge
mamba activate merra2_clavrx

yesterday=`date -d "1 day ago" +"%Y%m%d"`

INPUT_DATE=${1:-$yesterday}
HOUR=${2:-"00"}

if [[ $((INPUT_DATE)) != $INPUT_DATE ]]; then
	# Then INPUT_DATE is a string!  Is it a request for usage?
	# Or, is it an X, which means yesterday's date should be run?
	case "${INPUT_DATE}" in
		X|x) INPUT_DATE=$yesterday;;
		-h) usage;;
		*) echo "ERROR first input => ${INPUT_DATE} not recognized."
			usage;;
	esac
else
	if [ ${#INPUT_DATE} -lt 8 ]; 
	then
		if [ ${#INPUT_DATE} -eq 6 ];
		then
			# Assume the input was meant to be a date
			echo "ERROR:  The input date must either be number of days ago or a CCYYmmdd date."
			echo "The entry was:  ${INPUT_DATE.}"
			exit 1
		else
			# Otherwise, assume user meant days ago.
			INPUT_DATE=`date -d "${INPUT_DATE} day ago" +"%Y%m%d"`
		fi
	fi

fi


YYYY=${INPUT_DATE:0:4}
MM=${INPUT_DATE:4:2}
DD=${INPUT_DATE:6:2}

M2_DIR=${M2_DIR}/${YYYY}/${YYYY}_${MM}_${HOUR}
echo $M2_DIR
mkdir -p $M2_DIR

echo Date from FILELIST: ${INPUT_DATE}
OUTFILE_EXPECTED=$OUT_DIR/${YYYY}/geosfp.${YYYY:2:2}${MM}${DD}${HOUR}_F000.hdf
if [ -f $OUTFILE_EXPECTED ]; then
	echo "COMPLETE:  Expected $OUTFILE_EXPECTED file found."
	echo "If reprocessing is necessary, please remove the output file and start over."
	exit 1
fi
# Create tmp subdirectories for files
cd ${M2_DIR}


/bin/bash ${BIN_DIR}/fetch_geos/driver-geos-fp-getall.sh ${YYYY} ${MM} ${DD} $HOUR -w $M2_DIR

# Run merra conversion code for clavrx
python ${BIN_DIR}/geosfp_for_clavrx.py ${INPUT_DATE} ${HOUR}
#
# clean up
echo $M2_DIR

if [ ! -f $OUTFILE_EXPECTED ]; then
	echo "ERROR:  Expected $OUTFILE_EXPECTED file not found."
	echo "Retaining data in $M2_DIR"
	exit 1
else
	rm -rfv $M2_DIR
fi
echo finished at: `date`

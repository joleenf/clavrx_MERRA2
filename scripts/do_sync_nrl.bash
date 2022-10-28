#!/bin/bash

#datecmd=`which date`
datecmd=date

export PS4='line:${LINENO} function:${FUNCNAME[0]:+${FUNCNAME[0]}() }cmd: ${BASH_COMMAND} \n result: '
set -x

# command line, can input number of days ago to retrieve old data as first argument
# goes_sync.sh 1

finish () {
    echo "exiting, removing lock" >> $LOG 2>&1
    $datecmd >> $LOG 2>&1
    rmdir $LOCK >> $LOG 2>&1
}

trap finish EXIT

# for some reason, this was a bad thing to remove, so keep set -ex
export days_ago=$1
export data_product=$2

test -n "$days_ago" || export days_ago=0
test -n "$data_product" || export data_product="00"

function do_sync {

    # start log entry
    $datecmd +"%D %H:%M:%S starting $0" >> $LOG 2>&1

    #make dest directory
    if mkdir -p $DEST >> $LOG 2>&1 ; then
      echo "mkdir $DEST succeeded, continuing" >> $LOG 2>&1
    fi

    #mirror data
    lftp -c "set ssl:verify-certificate no;
    open '$URL';
    lcd $DEST;
    cd $REMOTEPATH;
    mirror --only-newer --verbose=2 $lftp_mirror"
    cd $DEST
}

function retrieve_tpw {
     REMOTEPATH=ftp/outgoing/fnmoc/models/navgem_0.5/${YEAR}/${THISDATE}${pwat_hr}/
     do_sync
     cat $DEST/US*${THISDATE}${pwat_hr}*prcp_h20 >> $DEST/navgem_${THISDATE}${data_product}.grib;err=$?
     if [ $err == 1 ]; then
         echo "ERROR: No prcp_h20 found for ${THISDATE}${data_product} from US*${THISDATE}00*prcp_h20"
         exit 1
     fi
}

DIR=$HOME
YEAR=$(${datecmd} +%Y -d "${days_ago} day ago")
THISDATE=$(${datecmd} +"%Y%m%d" -d "${days_ago} day ago")
TODAY=$(${datecmd} +"%Y_%m_%d" -d "${days_ago} day ago")

LOCK=$DIR/.${THISDATE}.lock
LOG=$DIR/logs/nrl_${THISDATE}_sync.log
DEST=/data/Personal/joleenf/data/navgem/$YEAR/${TODAY}/nrl_orig

mkdir -p $DEST
#make a lock directory
if mkdir $LOCK >> $LOG 2>&1 ; then
  echo "mkdir $LOCK succeeded, continuing" >> $LOG 2>&1
else
  echo "Lock (mkdir $LOCK) failed, NRL_NAVGEM_${data_product}_sync assumed running, exiting" >> $LOG 2>&1
  exit 0
fi


URL=https://www.usgodae.org/
REMOTEPATH=ftp/outgoing/fnmoc/models/navgem_0.5/${YEAR}/${THISDATE}${data_product}/
# include glob to capture forecasts up to 18Z but exclude all forecasts 21Z-144Z.
lftp_mirror="--include-glob=US*.0018_0056_00* --include-glob=US*.0018_0056_01* --exclude-glob=*.0018_0056_0[2-9]* --exclude-glob=*.0018_0056_1*"

do_sync

# cat files into one grib
cat $DEST/US*GR1* > $DEST/navgem_${THISDATE}${data_product}.grib

case $data_product in 
    "00"|"06") pwat_hr="00";;
    "12"|"18") pwat_hr="12";;
esac

cat $DEST/US*GR1*${THISDATE}${pwat_hr}*prcp_h20 >> $DEST/navgem_${THISDATE}${data_product}.grib;err=$?
if [ $err != 0 ]; then
    retrieve_tpw
fi

exit

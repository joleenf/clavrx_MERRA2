#!/bin/bash

#datecmd=`which date`
datecmd=gdate

# command line, can input number of days ago to retrieve old data as first argument
# goes_sync.sh 1

finish () {
    echo "exiting, removing lock" >> $LOG 2>&1
    $datecmd >> $LOG 2>&1
    rmdir $LOCK >> $LOG 2>&1
}

trap finish EXIT

# for some reason, this was a bad thing to remove, so keep set -ex
set -ex
export days_ago=$1
export data_product=$2

#echo $days_ago

test -n "$days_ago" || export days_ago=0
test -n "$data_product" || export data_product="00"

function do_sync {

    # start log entry
    $datecmd +"%D %H:%M:%S starting $0" >> $LOG 2>&1

    #make a lock directory
    if mkdir $LOCK >> $LOG 2>&1 ; then
      echo "mkdir $LOCK succeeded, continuing" >> $LOG 2>&1
    else
      echo "Lock (mkdir $LOCK) failed, mrms_${data_product}_sync assumed running, exiting" >> $LOG 2>&1
      exit 0
    fi

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

DIR=$HOME
YEAR=$(${datecmd} +%Y -d "${days_ago} day ago")
THISDATE=$(${datecmd} +"%Y%m%d" -d "${days_ago} day ago")
TODAY=$(${datecmd} +"%Y_%m_%d" -d "${days_ago} day ago")

LOCK=$DIR/.${THISDATE}.lock
LOG=$DIR/logs/${THISDATE}_sync.log
DEST=/Users/joleenf/data/clavrx/navgem/$YEAR/${TODAY}/

mkdir -p $DEST

URL=https://www.usgodae.org/
REMOTEPATH=ftp/outgoing/fnmoc/models/navgem_0.5/${YEAR}/${THISDATE}${data_product}/
lftp_mirror="--newer-than=now-24hours --include-glob=US*.0018_0056_00* --include-glob=US*.0018_0056_01* --exclude-glob=*.0018_0056_0[2-9]* --exclude-glob=*.0018_0056_1*"

do_sync

exit

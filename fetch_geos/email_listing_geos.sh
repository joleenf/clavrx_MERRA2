#!/bin/bash

source ~/.bash_profile

default_back=-1
months_back=${1:$default_back}
BASE="$( cd -P "$( dirname "$0" )" && pwd )"
echo $BASE
# set the date in the middle of the month
year=`date +'%Y'`
month=`date +'%b'`
last_month=`date -d "$(date -d "$month 15 $year" +%F) ${months_back} month ago" +'%Y %m'`
last_month_text=`date -d "$(date -d "$month 15 $year" +%F) ${months_back} month ago" +'%B %Y'`

echo "Running geos inventory for $last_month_text"

`/bin/bash $BASE/list_missing_geos.sh $last_month geosfp $DYNAMIC_ANCIL/geos/ | /bin/mail -s "Inventory for $last_month" "joleen.feltz@ssec.wisc.edu"`

: <<=cut
=pod

=head1 NAME


    previous_month_data.sh 

=head1 SYNOPSIS

    sh previous_month_data.sh 

=head1 DESCRIPTION

=head2 Requirements

=cut

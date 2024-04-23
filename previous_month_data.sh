#!/bin/bash

default_back=-1
months_back=${1:$default_back}
BASE="$( cd -P "$( dirname "$0" )" && pwd )"
echo $BASE
# set the date in the middle of the month
year=`date +'%Y'`
month=`date +'%b'`
last_month=`date -d "$(date -d "$month 15 $year" +%F) ${months_back} month ago" +'%Y %m'`
last_month_text=`date -d "$(date -d "$month 15 $year" +%F) ${months_back} month ago" +'%B %Y'`

echo "Running merra conversion code for $last_month_text"

bin_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
#cmd="screen -dm -S run_last_month /bin/bash $BASE/runMonth.sh $last_month"
cmd="/bin/bash $BASE/runMonth.sh $last_month $bin_dir/run_merra4clavrx.sh"

echo "Run $cmd"

eval $cmd

: <<=cut
=pod

=head1 NAME


    previous_month_data.sh 

=head1 SYNOPSIS

    sh previous_month_data.sh 

=head1 DESCRIPTION

=head2 Requirements

=cut

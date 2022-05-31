#!/bin/bash

default_back=-1
months_back=${1:$default_back}
BASE="$( cd -P "$( dirname "$0" )" && pwd )"
echo $BASE
last_month=`date --date="-${months_back} month" +'%Y %m'`
last_month_text=`date --date="-${months_back} month" +'%B %Y'`

echo "Running merra conversion code for $last_month_text"

#cmd="screen -dm -S run_last_month /bin/bash $BASE/runMonth.sh $last_month"
cmd="/bin/bash $BASE/runMonth.sh $last_month"

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

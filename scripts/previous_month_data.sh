#!/bin/bash

BASE="$( cd -P "$( dirname "$0" )" && pwd )"
echo $BASE
last_month=`date --date='-1 month' +'%Y %m'`
last_month_text=`date --date='-1 month' +'%B%Y'`
last_month_compress=`date --date='-1 month' +'%Y%m'`

echo "Running merra conversion code for $last_month_text"

cmd="screen -dm -S run_last_month /bin/bash $BASE/runMonth.sh $last_month" 

echo "Run $cmd"

exec $cmd

: <<=cut
=pod

=head1 NAME


    previous_month_data.sh 

=head1 SYNOPSIS

    sh previous_month_data.sh 

=head1 DESCRIPTION

=head2 Requirements

=cut

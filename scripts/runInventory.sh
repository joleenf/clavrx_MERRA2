#!/bin/bash

# Run using screen -dm -S run_for_year /bin/bash runYear.sh <year>
# Ex.) screen -dm -S run_2020 /bin/bash runYear.sh 2020

curr_year=`date +"%Y"`
BASE="$( cd -P "$( dirname "$0" )" && pwd )"
YEAR=${1:-$curr_year}

re='^[0-9]+$'
if ! [[ $YEAR =~ $re ]] ; then
	read  -n 4 -p "Enter a year in CCYY form to run an inventory of processed files: " YEAR
fi

for i in {1..12}
do
   screen_name=run$YEAR_$i
   /bin/bash $BASE/count_inventory.sh $YEAR $i;err=$?
done

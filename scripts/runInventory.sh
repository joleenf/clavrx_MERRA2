#!/bin/bash

# Run using screen -dm -S run_for_year /bin/bash runYear.sh <year>
# Ex.) screen -dm -S run_2020 /bin/bash runYear.sh 2020

curr_year=`date +"%Y"`
curr_month=`date +"%m"`
BASE="$( cd -P "$( dirname "$0" )" && pwd )"
YEAR=${1:-$curr_year}

if [[ "${YEAR}" == "${curr_year}" ]];
then
    end_loop=$curr_month
else
    end_loop=12
fi

re='^[0-9]+$'
if ! [[ $YEAR =~ $re ]] ; then
   printf "\\n\\tRun this scripts with a single parameter, the year: /bin/bash $0 YYYY\\n\\n"
   printf "\\tExample: /bin/bash $0 2022\\n\\n"
   exit 1
fi

# make sure directory exists...
mkdir -p /apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2/$YEAR

i=1
while [[ $i -le $end_loop ]]
do
   screen_name=run$YEAR_$i
   /bin/bash $BASE/count_inventory.sh $YEAR $i;err=$?
   ((i++))
done

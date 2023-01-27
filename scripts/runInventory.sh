#!/bin/bash

# Run using screen -dm -S run_for_year /bin/bash runYear.sh <year>
# Ex.) screen -dm -S run_2020 /bin/bash runYear.sh 2020

curr_year=`date +"%Y"`
BASE="$( cd -P "$( dirname "$0" )" && pwd )"
YEAR=${1:-$curr_year}

re='^[0-9]+$'
if ! [[ $YEAR =~ $re ]] ; then
   printf "\\n\\tRun this scripts with a single parameter, the year: /bin/bash $0 YYYY\\n\\n"
   printf "\\tExample: /bin/bash $0 2022\\n\\n"
   exit 1
fi

# make sure directory exists...
mkdir -p /apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2/$YEAR

for i in {1..12}
do
   /bin/bash $BASE/count_inventory.sh $YEAR $i;err=$?
done

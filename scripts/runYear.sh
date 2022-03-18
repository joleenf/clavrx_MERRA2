#!/bin/bash

# Run using screen -dm -S run_for_year /bin/bash runYear.sh <year>
# Ex.) screen -dm -S run_2020 /bin/bash runYear.sh 2020

BASE="$( cd -P "$( dirname "$0" )" && pwd )"
YEAR=${1:-2021}

for i in {1..12};
do
   screen_name=run$YEAR_$i
   cmd="/bin/bash $BASE/runMonth.sh $YEAR $i"
   echo $cmd 
   eval $cmd
done

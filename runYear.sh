#!/bin/bash

# Run using screen -dm -S run_for_year /bin/bash runYear.sh <year>
# Ex.) screen -dm -S run_2020 /bin/bash runYear.sh 2020

BASE="$( cd -P "$( dirname "$0" )" && pwd )"

# YEAR=${1:-2021}
# year=$YEAR

for year in {2000..1980};
do
   for i in {1..12};
   do
      screen_name=run$year_$i
      echo $screen_name
      cmd="/bin/bash $BASE/runMonth.sh $year $i"
      echo $cmd 
      eval $cmd
   done
done

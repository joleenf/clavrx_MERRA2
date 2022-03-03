#!/bin/bash

BASE="$( cd -P "$( dirname "$0" )" && pwd )"
YEAR=${1:-2021}

for i in {1..12};
do
   screen_name=run$YEAR_$i
   cmd="/bin/bash $BASE/runMonth.sh $YEAR $i"
   echo $cmd 
   screen -dm -S $screen_name $cmd
done

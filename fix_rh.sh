#!/bin/bash
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

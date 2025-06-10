#!/bin/sh
# Enter a day, and the 0, 6, 12, 18 will be downloaded.  Trying to 
# enter this from command line was not working because the dates werre
# not setting consistently between direct testing of variable within script
# and entering on command line.
day=$1
for ts in 00 06 12 18; do
	date1=$(date -d "${day}T${ts}:00:00" +"%Y-%m-%dT%H:%M:%S")
	date2=$(date -d "${day}T${ts}:01:00" +"%Y-%m-%dT%H:%M:%S")

   echo $date1
   echo $date2

   download_dir=/data/users/joleenf/GEOS_IT/IT/

   for tag in ASM_I1 OCN_T1 LND_T1 FLX_T1;
   do
   	sips_name=GEOSIT_${tag}_C_SLV
   	~/Projects/asipscli.linux-amd64 ancillary -p ${sips_name} -v 5.29.4 -s $date1 -e $date2 -d $download_dir 
   done
   for tag in ASM_I3;
   do
   	sips_name=GEOSIT_${tag}_C_V72
   	~/Projects/asipscli.linux-amd64 ancillary -p ${sips_name} -v 5.29.4 -s $date1 -e $date2 -d  $download_dir
   done
done

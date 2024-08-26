# This just takes today's month and runs an inventory.  This is 
# meant for backfilling, so the major flaw is today might be the first
# and there is nothing to backfill or inventory.

BASE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")" )"
TODAY=$(date +"%Y %m")
if [ $# == 0 ]; then
	sh ${BASE}/geos_backfill.sh $TODAY
fi
YESTERDAY=$(date -d "1 day ago" +"%Y %m")

if [ "${YESTERDAY}" != "${TODAY}" ]; then
	# we have captured a month transition
	sh ${BASE}/geos_backfill.sh $YESTERDAY
fi

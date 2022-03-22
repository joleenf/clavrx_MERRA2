#! /bin/sh
# Runs all (9) wget_* scripts to collect data for a given YYYY, MM, DD

usage() {
cat << EndOfMessage

    Usage: sh $0 [options] YYYY MM DD"

    Download data from GES DISC based on date and key"

    Options:
        -w download directory (default is pwd).
        -k key (valid keys: all)
        -h Display this usage information

EndOfMessage
    echo $VAR
    exit

}

set -x
export PS4=' ${DATETIME_NOW} line:${LINENO} function:${FUNCNAME[0]:+${FUNCNAME[0]}() } cmd: ${BASH_COMMAND} result: '

oops() {
    printf "\nScript must always have a YYYY MM DD entered regardless of flags used!!!!!\n"
    usage
}

download_dir=$(pwd)
in_key=all

[ $# -eq 0 ] && usage
while getopts "w:k:h" flag; do
	case "$flag" in
		w) download_dir=$OPTARG;;
		k) in_key=$OPTARG;;
		h|*) usage;;
	esac
done

shift $(($OPTIND - 1))
args=$*

YYYY=`echo $args | awk -F" " '{print $1}'`
MM=`echo $args | awk -F" " '{print $2}'`
DD=`echo $args | awk -F" " '{print $3}'`


[ -z "${YYYY}" ] && oops || pass
[ -z "${MM}" ] && oops || pass
[ -z "${DD}" ] && oops || pass


SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export BASE="$( dirname $SCRIPTS_DIR)"

if [ -d ${download_dir} ];
then
	cd ${download_dir}
else
	echo "Directory ${download_dir} does not exist."
	exit 1
fi

case "${in_key}" in
   all) FILETYPES=(inst6_3d_ana_Np inst6_3d_ana_Nv tavg1_2d_slv_Nx
                   tavg1_2d_flx_Nx inst3_3d_asm_Np const_2d_ctm_Nx
                   inst1_2d_asm_Nx tavg1_2d_lnd_Nx tavg1_2d_rad_Nx)
                    ;;
   inst6_3d_ana_Np | inst6_3d_ana_Nv | tavg1_2d_slv_Nx | tavg1_2d_flx_Nx |  \
   inst3_3d_asm_Np | const_2d_ctm_Nx | inst1_2d_asm_Nx | tavg1_2d_lnd_Nx | \
   tavg1_2d_rad_Nx) FILETYPES=(${in_key})
                    ;;
   *) echo "Unkown key $in_key"
      exit 1
      ;;
esac

source $SCRIPTS_DIR/get_stream.sh ${YYYY} ${MM} ${DD}
cd ${download_dir}

for key in "${FILETYPES[@]}"
do
	#${SCRIPTS_DIR}/wget_${key}.sh ${YYYY} ${MM} ${DD}
	if [ "${key}" == "const_2d_ctm_Nx" ]; then
		${SCRIPTS_DIR}/wget_${key}.sh ${YYYY} ${MM} ${DD}
	else
		echo $key
        	eval ${key}
		check_before_any_stream_call $LOCAL_DIR $TARGET_FILE $REANALYSIS
	fi
        cd ${download_dir}
done

rm */robots.txt
rm */robots.txt.*

exit 0

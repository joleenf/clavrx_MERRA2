#! /bin/sh
# Runs all (9) wget_* scripts to collect data for a given YYYY, MM, DD

usage() {
cat << EndOfMessage

    Usage: sh $0 [options] YYYY MM DD"

    Download data from GES DISC based on date and key"

    Options:
        -w download directory (default is pwd).
        -k key (valid keys: all)
        -l lists valid keys.
        -h Display this usage information

EndOfMessage
    echo $VAR
    exit

}

oops() {
    printf "\nScript must always have a YYYY MM DD entered regardless of flags used!!!!!\n"
    usage
}

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export BASE="$( dirname $SCRIPTS_DIR)"

download_dir=$(pwd)
in_key=all
all_keys=(inst6_3d_ana_Np inst6_3d_ana_Nv tavg1_2d_slv_Nx tavg1_2d_flx_Nx inst3_3d_asm_Np const_2d_ctm_Nx inst1_2d_asm_Nx tavg1_2d_lnd_Nx tavg1_2d_rad_Nx)
[ $# -eq 0 ] && usage
while getopts "w:k:h:l" flag; do
	case "$flag" in
		w) download_dir=$OPTARG;;
		k) in_key=$OPTARG;;
		l) echo "Use one of the following keys with -k flag and date input:"
                   echo "all ${all_keys[@]}"
		   exit;;
		h|*) usage;;
	esac
done

shift $(($OPTIND - 1))
args=$*

YYYY=`echo $args | awk -F" " '{print $1}'`
MM=`echo $args | awk -F" " '{print $2}'`
DD=`echo $args | awk -F" " '{print $3}'`


set -x
export PS4='line:${LINENO} function:${FUNCNAME[0]:+${FUNCNAME[0]}() }cmd: ${BASH_COMMAND} \n result: '

[ -z "${YYYY}" ] && oops || continue
[ -z "${MM}" ] && oops || continue
[ -z "${DD}" ] && oops || continue

source $SCRIPTS_DIR/get_stream.sh ${YYYY} ${MM} ${DD}

if [ -d ${download_dir} ];
then
	cd ${download_dir}
else
	echo "Directory ${download_dir} does not exist."
	exit 1
fi

case "${in_key}" in
   all) FILETYPES=(inst6_3d_ana_Np inst6_3d_ana_Nv tavg1_2d_slv_Nx tavg1_2d_flx_Nx inst3_3d_asm_Np const_2d_ctm_Nx inst1_2d_asm_Nx tavg1_2d_lnd_Nx tavg1_2d_rad_Nx)
                    ;;
   inst6_3d_ana_Np | inst6_3d_ana_Nv | tavg1_2d_slv_Nx | tavg1_2d_flx_Nx |  \
   inst3_3d_asm_Np | const_2d_ctm_Nx | inst1_2d_asm_Nx | tavg1_2d_lnd_Nx | \
   tavg1_2d_rad_Nx) FILETYPES=(${in_key})
                    ;;
   *) echo "Unkown key $in_key"
      exit 1
      ;;
esac

cd ${download_dir}

for key in "${FILETYPES[@]}"
do
	#${SCRIPTS_DIR}/wget_${key}.sh ${YYYY} ${MM} ${DD}
	if [ "${key}" == "const_2d_ctm_Nx" ]; then
		echo "Starting New Key (Constants): $key"
		${SCRIPTS_DIR}/wget_${key}.sh ${YYYY} ${MM} ${DD} ${download_dir}
	else
		echo "Starting New Key....$key"
        	eval ${key}
		check_before_any_stream_call $LOCAL_DIR $TARGET_FILE $REANALYSIS
	fi
        yes '' | sed 3q  # print 3 blank lines
        cd ${download_dir}
done

rm */robots.txt
rm */robots.txt.*

exit 0

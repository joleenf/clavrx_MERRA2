#! /bin/sh

# Runs all (9) geos-fp_get_* scripts to collect data for a given YYYY, MM, DD

function usage() {
cat << EndOfMessage

    Usage: sh $0 [options] YYYY MM DD HH
    Example:  sh $0 -k const_2d_ctm_Nx 2024 01 01 00

    Download data from GMAO NCCS Database based on key, date and model run
    NOTE:  Key options must preceed date if a key is being used.

    Options:
    	-w download directory (default is ${download_dir}).
    	-k key (valid keys: all ${!FILETYPES[@]})
   	-h Display this usage information

EndOfMessage
    echo $VAR
    exit

}

function oops() {
    printf "\nScript must always have a YYYY MM DD entered regardless of flags used!!!!!\n"
    usage
}

download_dir=/ships22/cloud/scratch/7day/GEOS-FP_INPUT
in_key=all

[ $# -eq 0 ] && usage
while getopts "w:k:h" flag; do
	case "$flag" in 
		w) download_dir=$OPTARG;;
		k) in_key=$OPTARG;;
		h) usage;;
	esac
done

YYYY=${@:$OPTIND:1}
MM=${@:$OPTIND+1:1}
DD=${@:$OPTIND+2:1}
HH=${@:$OPTIND+3:1}

[[ -z "${YYYY}" ]] && oops
[[ -z "${MM}" ]] && oops
[[ -z "${DD}" ]] && oops
[[ -z "${HH}" ]] && oops

echo $YYYY $MM $DD $HH
echo $download_dir


export PS4='+${LINENO}: '
set -x
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ -d ${download_dir} ];
then
	cd ${download_dir}
else
	echo "Directory ${download_dir} does not exist." 
	exit 1
fi

case "${in_key}" in
   all) FILETYPES=(const_2d_ctm_Nx::2d_ctm inst3_2d_asm_Nx::2d_asm
	           inst3_3d_asm_Np::3d_asm tavg1_2d_flx_Nx::2d_flx 
		   tavg1_2d_lnd_Nx::2d_lnd tavg1_2d_rad_Nx::2d_rad 
		   tavg1_2d_slv_Nx::2d_slv)
                    ;;
   const_2d_ctm_Nx) FILETYPES=(const_2d_ctm_Nx::2d_ctm)
                    ;;
   inst3_2d_asm_Nx) FILETYPES=(inst3_2d_asm_Nx::2d_asm)
		    ;;
   inst3_3d_asm_Np) FILETYPES=(inst3_3d_asm_Np::3d_asm)
                    ;;
   tavg1_2d_flx_Nx) FILETYPES=(tavg1_2d_flx_Nx::2d_flx)
                    ;;
   tavg1_2d_lnd_Nx) FILETYPES=(tavg1_2d_lnd_Nx::2d_lnd)
                    ;;
   tavg1_2d_rad_Nx) FILETYPES=(tavg1_2d_rad_Nx::2d_rad)
                    ;;
   tavg1_2d_slv_Nx) FILETYPES=(tavg1_2d_slv_Nx::2d_slv)
                    ;;
   *) echo "Unkown key $in_key"
      exit 1
      ;;
esac

echo $FILETYPES
for association in "${FILETYPES[@]}"
do
	echo "${association}"
        key="${association%%::*}"
	download_path=$(pwd)/${YYYY}/${MM}_${DD}_${HH}
	mkdir -p ${download_path}
	if [ "${key}" == "const_2d_ctm_Nx" ]; then
   	        ${SCRIPTS_DIR}/geos-fp_get_${key}.sh ${download_path}
        else
		${SCRIPTS_DIR}/geos-fp_retrieve.sh ${YYYY} ${MM} ${DD} ${HH} ${key} $download_path
	fi
done

for robots_file in `find . -name "robots*"`; do
	rm $robots_file
done

exit

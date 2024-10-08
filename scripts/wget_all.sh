#! /bin/sh

# Runs all (9) wget_* scripts to collect data for a given YYYY, MM, DD

function usage() {
cat << EndOfMessage

    Usage: sh $0 [options] YYYY MM DD"

    Download data from GES DISC based on date and key"

    Options:
    	-w download directory (default is pwd).
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

download_dir=$(pwd)
in_key=all

[ $# -eq 0 ] && usage
while getopts "w:k:h" flag; do
	case "$flag" in 
		w) download_dir=$OPTARG;;
		k) in_key=$OPTARG;;
		h) usage;;
	esac
done

set -x
YYYY=${@:$OPTIND:1}
MM=${@:$OPTIND+1:1}
DD=${@:$OPTIND+2:1}

[[ -z "${YYYY}" ]] && oops
[[ -z "${MM}" ]] && oops
[[ -z "${DD}" ]] && oops


SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

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
   inst6_3d_ana_Np) FILETYPES=(inst6_3d_ana_Np)
                    ;;
   inst6_3d_ana_Nv) FILETYPES=(inst6_3d_ana_Nv)
                    ;;
   tavg1_2d_slv_Nx) FILETYPES=(tavg1_2d_slv_Nx)
                    ;;
   tavg1_2d_flx_Nx) FILETYPES=(tavg1_2d_flx_Nx)
                    ;;
   inst3_3d_asm_Np) FILETYPES=(inst3_3d_asm_Np)
                    ;;
   const_2d_ctm_Nx) FILETYPES=(const_2d_ctm_Nx)
                    ;;
   inst1_2d_asm_Nx) FILETYPES=(inst1_2d_asm_Nx)
                    ;;
   tavg1_2d_lnd_Nx) FILETYPES=(tavg1_2d_lnd_Nx)
                    ;;
   tavg1_2d_rad_Nx) FILETYPES=(tavg1_2d_rad_Nx)
                    ;;
   *) echo "Unkown key $in_key"
      exit 1
      ;;
esac

for key in "${FILETYPES[@]}"
do
	echo "${key}"
	${SCRIPTS_DIR}/wget_${key}.sh ${YYYY} ${MM} ${DD}
	cd ${download_dir}
done

rm robots*.txt.tmp
rm robots*.txt.tmp*

exit

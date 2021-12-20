#! /bin/sh

# Runs all (9) wget_* scripts to collect data for a given YYYY, MM, DD

function usage() {
    echo -e "\\nDownload data from GES DISC based on date and key"
    echo -e "Example:"
    echo -e "\\t "$ "sh $0 [options] YYYY MM DD"
    echo -e "\\nOptions:"
    echo -e "\\t-w download directory (default is pwd)."
    echo -e "\\n\\t-k key (valid keys: all ${!FILETYPES[@]})"
    echo -e "\\n\\t-h Display this usage information"
    echo -e "\\n"
    exit 1
}

download_dir=$(pwd)
key=all
declare -A FILETYPES=([inst6_3d_ana_Np]=3d_ana [inst6_3d_ana_Nv]=3d_ana [tavg1_2d_slv_Nx]=2d_slv \
	               [tavg1_2d_flx_Nx]=2d_flx [inst3_3d_asm_Np]=3d_asm [const_2d_ctm_Nx]=2d_ctm \
		       [inst1_2d_asm_Nx]=2d_asm [tavg1_2d_lnd_Nx]=2d_lnd [tavg1_2d_rad_Nx]=2d_rad) 

set -x
while getopts "w:k:h" flag; do
	case "$flag" in 
		w) download_dir=$OPTARG;;
		k) key=$OPTARG;;
		h) usage;;
	esac
done

YYYY=${@:$OPTIND:1}
MM=${@:$OPTIND+1:1}
DD=${@:$OPTIND+2:1}


SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ -d ${download_dir} ];
then
	cd ${download_dir}
else
	echo "Directory ${download_dir} does not exist." 
	exit 1
fi

[ "${key}" == "all" ] && this_set=${!FILETYPES[@]} || this_set=${key}
echo ${this_set}
for file_key in "${!FILETYPES[@]}"
do
	echo "${file_key}"
	mkdir -p ${FILETYPES[$file_key]}
	${SCRIPTS_DIR}/wget_${file_key}.sh ${YYYY} ${MM} ${DD}
	cd ${download_dir}
done

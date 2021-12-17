#! /bin/sh

# Runs all (9) wget_* scripts to collect data for a given YYYY, MM, DD

YYYY=${1}
MM=${2}
DD=${3}

dir=$(pwd)
download_dir=${4:-$dir}
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ -d ${download_dir} ];
then
	cd ${download_dir}
else
	echo "Directory ${download_dir} does not exist." 
	exit 1
fi

declare -A filetypes=([inst6_3d_ana_Np]=3d_ana [inst6_3d_ana_Nv]=3d_ana [tavg1_2d_slv_Nx]=2d_slv \
	               [tavg1_2d_flx_Nx]=2d_flx [inst3_3d_asm_Np]=3d_asm [const_2d_ctm_Nx]=2d_ctm \
		       [inst1_2d_asm_Nx]=2d_asm [tavg1_2d_lnd_Nx]=2d_lnd [tavg1_2d_rad_Nx]=2d_rad) 

for file_key in "${!filetypes[@]}"
do
	echo "${file_key}"
	mkdir -p ${filetypes[$file_key]}
	${SCRIPTS_DIR}/wget_${file_key}.sh ${YYYY} ${MM} ${DD}
	cd ${download_dir}
done


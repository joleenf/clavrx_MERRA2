#! /bin/sh

source ~/.bashrc
conda activate merra2_clavrx

# Runs all (9) wget_* scripts to collect data for a given YYYY, MM, DD

#function usage() {
#cat << EndOfMessage
#
#    Usage: sh $0 [options] YYYY MM DD"
#
#    Download data from GES DISC based on date and key"
#
#    Options:
#    	-w download directory (default is pwd).
#    	-k key (valid keys: all ${!FILETYPES[@]})
#   	-h Display this usage information
#
#EndOfMessage
#    echo $VAR
#    exit
#
#}

download_dir=$(pwd)
in_key=all

if [ $# -eq 0 ]; then 
	`/bin/pod2usage wget_all.sh`
	exit
fi
while getopts "w:k:h" flag; do
	case "$flag" in 
		w) download_dir=$OPTARG;;
		k) in_key=$OPTARG;;
		h) `/bin/pod2usage wget_all.sh`
                   exit;;
	esac
done

YYYY=${@:$OPTIND:1}
MM=${@:$OPTIND+1:1}
DD=${@:$OPTIND+2:1}

echo $YYYY $MM $DD

length_MM=${#MM}
if [ "$length_MM" -eq 1 ]; then MM=`printf "%02d" $MM`;fi

length_DD=${#DD}
if [ "$length_DD" -eq 1 ]; then DD=`printf "%02d" $DD`;fi

export PS4=' ${DATETIME_NOW} line:${LINENO} function:${FUNCNAME[0]:+${FUNCNAME[0]}(): } '
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

mkdir -p $download_dir

if [ -d ${download_dir} ];
then
	cd ${download_dir}
else
	echo "Directory ${download_dir} does not exist." 
	exit 1
fi

case "${in_key}" in
   all) FILETYPES=(inst6_3d_ana_Np::3d_ana inst6_3d_ana_Nv::3d_ana tavg1_2d_slv_Nx::2d_slv
          tavg1_2d_flx_Nx::2d_flx inst3_3d_asm_Np::3d_asm const_2d_ctm_Nx::2d_ctm
          inst1_2d_asm_Nx::2d_asm tavg1_2d_lnd_Nx::2d_lnd tavg1_2d_rad_Nx::2d_rad);;
   inst6_3d_ana_Np) FILETYPES=(inst6_3d_ana_Np::3d_ana)
                    ;;
   inst6_3d_ana_Nv) FILETYPES=(inst6_3d_ana_Nv::3d_ana)
                    ;;
   tavg1_2d_slv_Nx) FILETYPES=(tavg1_2d_slv_Nx::2d_slv)
                    ;;
   tavg1_2d_flx_Nx) FILETYPES=(tavg1_2d_flx_Nx::2d_flx)
                    ;;
   inst3_3d_asm_Np) FILETYPES=(inst3_3d_asm_Np::3d_asm)
                    ;;
   const_2d_ctm_Nx) FILETYPES=(const_2d_ctm_Nx::2d_ctm)
                    ;;
   inst1_2d_asm_Nx) FILETYPES=(inst1_2d_asm_Nx::2d_asm)
                    ;;
   tavg1_2d_lnd_Nx) FILETYPES=(tavg1_2d_lnd_Nx::2d_lnd)
                    ;;
   tavg1_2d_rad_Nx) FILETYPES=(tavg1_2d_rad_Nx::2d_rad)
                    ;;
   *) echo "Unkown key $in_key"
      exit 1
      ;;
esac

for association in "${FILETYPES[@]}"
do
	echo "${association}"
        key="${association%%::*}"
        value="${association##*::}"
	mkdir -p ${value}
	${SCRIPTS_DIR}/wget_${key}.sh ${YYYY} ${MM} ${DD}
	cd ${download_dir}
done

rm robots*.txt.tmp
rm robots*.txt.tmp*

: <<=cut
=pod

=head1 NAME


#
#    Download data from GES DISC based on date and key"
#
    wget_all.sh - Use wget to retrieve merra2 data from the GES-DISC.

=head1 SYNOPSIS

    sh wget_all.sh [options] YYYY MM DD"

      where: YYYY is the year (4 digit required)
             MM   is the 1- or 2-digit month
             DD   is the 2- or 2-digit day.

   Recognized optional command line arguments
      -w <string>  -- Download path for merra2 input data. (default is $pwd)
      -k key <string> (valid keys: all)
       Individual Keys are inst6_3d_ana_Np inst6_3d_ana_Nv tavg1_2d_slv_Nx
                           tavg1_2d_flx_Nx inst3_3d_asm_Np const_2d_ctm_Nx
                           inst1_2d_asm_Nx tavg1_2d_lnd_Nx tavg1_2d_rad_Nx
      -h  -- Display usage message

=head1 DESCRIPTION

    Retieve the MERRA2 data from the GES-DISC using wget.

=head2 Requirements

User credentials file as described by https://disc.gsfc.nasa.gov/data-access.

=cut

exit

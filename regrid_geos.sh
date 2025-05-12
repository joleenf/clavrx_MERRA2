# under the clavrx_MERRA2 repository or other known regrid_tools directory location
# https://github.com/GEOS-ESM/GEOSgcm/wiki/Converting-Cubed-Sphere-Data-to-Lat-Lon-outside-NASA-HPC-ecosystem
# git clone https://github.com/GMAO-SI-Team/regridding-tools.git
# This essentially uses 'nco=5.0.1' or greater, suggesting 5.3.3 due to some fixes to vertical
# 'ESMF=8.2.0=*mpi*' netcdf4 scipy

# Call this script from another script.
# Enter full path of the input file and full path of the desired output name.

source $HOME/.bashrc
module purge
mamba activate base 
mamba activate merra2_clavrx

trap finish exit

function finish {
	if [ -f $TMPFILE ]; then
            rm ${TMPFILE}
	fi
}

TMPFILE="NONE"

function create_directory_from_parsed_input_file {
	local in_dir=$1
	local input_file=$2
	sorting_flag=$3
	# verify this is the basename
	input_filebase=$(basename $input_file)
	file_date=`echo $input_filebase | gawk -F"." '{print substr($6,1,10)}'`
	file_hour=`echo $input_filebase | gawk -F"." '{print substr($6,12,2)}'`

	if "$sorting_flag"; then
		in_data="${in_dir}/${file_date}/${file_hour}00/C180"
		mkdir -p $in_data
		mv $in_dir/$input_filebase $in_data/
		newloc_input_file=$in_data/$input_filebase
	else
		newloc_input_file=$input_file
	fi
	out_dir="${in_dir}/${file_date}/${file_hour}00"
	mkdir -p $out_dir
    cd $out_dir  # so that regrid logs end up in the out_dir
	echo $newloc_input_file
}
function regrid {
	local input_file=$1
	local output_file=$2
	weights=/data/users/joleenf/gridspec/PE180x1080-CF_576x361-DC_conserve.nc4
	map_fl=vrt_prs_geos_L42.nc
	regrid_tools=$HOME/Projects/regridding-tools
	vertical_filename_specification="V72"   # for GEOS-IT 3D files, V72 indicates data are on sigma levels

	TMPFILE=$(mktemp /tmp/tmp.XXXXXXX.nc)
	# Do the work in the output directory (temp files are created with convert_tools.py. Run in a safer +rw location)
	cd $(dirname $output_file)

	# If map_fl file is needed, used ncap2
	#ncap2 -O -v -s 'defdim("plev", 42);plev[$plev]={1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50, 40, 30, 20, 10, 7, 5, 4, 3, 2, 1, 0.7, 0.5, 0.4, 0.3, 0.1};' vrt_prs_geos_L42.nc
	
	regrid_weights_dir=$(dirname $weights)

	# Horizontal regridding using conserve weights (if no weights have been generated, the grid_dir flag
	# tells ncremap to store weights in that directory.
	python $regrid_tools/convert_tool.py -n $input_file -o $TMPFILE -m conserve -i 576 -j 361 -d DC -p PC --grid_dir $regrid_weights_dir

	# Vertical regridding using map file need to explain to ncreamp that the vertical level is "lev"
	# in the input file.
	if [[ $input_file == *${vertical_filename_specification}* ]]; then
	    ncremap --vrt_out=${regrid_weights_dir}/${map_fl} --plev_nm=lev -i $TMPFILE -o $output_file
		ls $output_file
	else
	    mv $TMPFILE $output_file
		ls $output_file
	fi
}

in_dir=$1
sorting_flag=${2:-true}
const_dir="/data/users/joleenf/GEOS_IT/it_const"
echo $sorting_flag

# Input files should be in parent directory. This code sorts (e.g. 2019-01-01/0000/C180)
# Output files will be organized in dated/timestamped directory (e.g. 2019-01-01/0000/)
count=0
for input_file in $(find $in_dir -maxdepth 1 -name "*C180*.nc4"); do
	count+=1
	echo $input_file
	new_input_path=`create_directory_from_parsed_input_file $in_dir $input_file $sorting_flag`
	out_almost=`echo $new_input_path | awk -F"C180x180x6" '{ print $1 "L576x361" $2}'`
	out_dir=$(dirname $(dirname $out_almost))
	echo $out_dir
	out_fn=$out_dir/$(basename $out_almost)
	regrid $new_input_path $out_fn
done
if [[ $count -eq 0 ]]; then
	set -o noglob
	echo "Ooops:  No files found in $in_dir matching regex '*C180*.nc4'"
	set -o noglob
	exit 1
fi
cp $const_dir/GEOS.it.asm.asm_const_0hr_glo_L576x361_slv.GEOS5271.2018-01-01T0000.V01.nc4 $out_dir
exit 0

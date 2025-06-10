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

WEIGHTS=/data/users/joleenf/gridspec/PE180x1080-CF_576x361-DC_conserve.nc4

export PS4='+${LINENO}: '

trap finish exit 

function usage() {
    printf "
    Usage:
        $0 <COMMAND> <input directory where cubed sphere geosIT data area located> <sorting flag>
    Example:
        $0 main /data/users/GEOSIT/ false

    where the input directory is required
    the sorting flag defaults to true.  Set to false if input data has already been sorted
	 to a dated directory, otherwise the data will be moved. In this case of false, the output
	 directory is assumed to be one level up from the input directory because this is the final
	 configuration from this script when run as true.  The sorting flag is only used for main.


    Description:
    $0 -h
    displays this usage message and exits.

	 $0 main <input directory>
	     main
        - Sorts data into dated/synoptic run directories under the input parent path
		    (i.e. /data/users/GEOSIT/2019-01-01/0000 in the example above)
		  - Converts the cubed sphere eta level data to rectilinear lat/lon data with 42 pressure
		    levels
		  - Changes the Dobson unit for total ozone (for backwards compatibility with code written for MERRA)
		  - Copies the constants file to the working directory

	 $0 vertical_regrid <v72 file in C180_v72 form>
	     Regrid the eta level file to a pressure level file.
	 $0 cp_const <output directory>
	     Copies constants file from a fixed directory location to current output directory 
	 $0 update_unit <output directory>
	     Updates dobson unit in ouput asm_inst_1hr file

"
	 exit $err
 }

err=0
function finish {
	for FILE_TO_REMOVE in $TMPFILE $UNPACKED_TEMP $UNPACKED_v72_TEMP; do
   	if [[ ! -z ${FILE_TO_REMOVE+x}  &&  -f $FILE_TO_REMOVE ]]; then
			rm -v ${FILE_TO_REMOVE}
	   fi
	done
	if [ $err != 0 ]; then
		usage
	fi
}

function setup {

COMMAND=$1
in_dir=$2
LOG_DIR=/data/users/$(whoami)/logs

sorting_flag=${3:-true}
echo $sorting_flag

if [[ "$COMMAND" -eq "main" ]]; then
   cmd="$COMMAND $in_dir $sorting_flag"
	if [ ! -d "$in_dir" ]; then
		echo "ERROR, input arg#2 is not a valid directory."
		exit 1
	fi
else
	cmd="$COMMAND $in_dir"
fi

DAY=`date +"%Y-%m-%d"`
# store in home, move to outdir later
SCR_BASE=$(basename -- "$0")
SCR_NAME=${SCR_BASE%.*}
export LOGFILE=${LOG_DIR}/${SCR_NAME}_${COMMAND}.log
echo $LOGFILE

if [ -n "$LOGFILE" ]; then
  exec 1> >(awk '{ print strftime("[ stdout %Y-%m-%dT%H%M%S ]", systime()), $_; fflush(); }' >>${LOGFILE} )
  exec 2> >(awk '{ print strftime("[ stderr %Y-%m-%dT%H%M%S ]", systime()), $_; fflush(); }' >>${LOGFILE} )
set -x
fi

echo $cmd
eval $cmd;err=$?
exit 0
}

function cp_const {
	out_dir=$1
   const_dir="/data/users/joleenf/GEOS_IT/it_const"
   cp -v $const_dir/GEOS.it.asm.asm_const_0hr_glo_L576x361_slv.GEOS5271.2018-01-01T0000.V01.nc4 $out_dir
}

export -f cp_const

function _create_directory_from_parsed_input_file {
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
	   out_dir="${in_dir}/${file_date}/${file_hour}00"
      mkdir -p $out_dir
	else
		newloc_input_file=$input_file
		out_dir=$(dirname $in_dir)
	fi
   cd $out_dir  # so that regrid logs end up in the out_dir
	echo $newloc_input_file
}


function vertical_regrid {
	local input_file=$1
	echo "Vertical Regrid $input_file"
	echo "============================="
	if [ -d "$input_file" ]; then
		file_to_convert=$(find $input_file -name "*v72*.nc4" -print)
	else
		file_to_convert=$input_file
	fi
   map_fl=vrt_prs_geos_L42_hPa.nc
	regrid_weights_dir=$(dirname $WEIGHTS)
	UNPACKED_v72_TEMP=$(mktemp /tmp/new_input.XXXXXXX.nc)
	ncpdq -O -U $file_to_convert $UNPACKED_v72_TEMP
		
   # Vertical regridding using map file need to explain to ncreamp that the vertical level is "lev"
   #  in the input file.   NEED THE NAME AND DIRECTORY FROM $file_to_convert, not TEMP, this is right.
	# levels=`echo $file_to_convert | awk -F"C180x180x6_" '{print substr($2,1,3)}'`
	new_fn=`echo $file_to_convert | awk -F"v72" ' {print $1"p42"$2}'`
		
   # Actually convert from the unpacked file. (This isn't working, because it expects sigma or
	# hybrid-sigma.  GEOS-IT is an eta level model.
	#ncremap --vrt_out=${regrid_weights_dir}/${map_fl} --plev_nm=lev -v QV,QI,QL,T,CLOUD,U,V,PHIS,H,O3 -i $UNPACKED_v72_TEMP -o $new_fn;err=$?

	# can't seem to pass the variables as I need, so just call regrid from here for this case
	if [ $err -eq 1 ]; then
		echo ncremap error in vertical grid subroutine $err
		exit 1
	fi
	# don't want the intermediate file to mess up the python code
	rm $input_file 
}
export -f vertical_regrid

function regrid {
	local input_file=$1
	local output_file=$2
	map_fl=vrt_prs_geos_L42_hPa.nc
	regrid_tools=$HOME/Projects/regridding-tools
	vertical_filename_specification="V72"   # for GEOS-IT 3D files, V72 indicates data are on sigma levels

	TMPFILE=$(mktemp /tmp/tmp.XXXXXXX.nc)
	UNPACKED_TEMP=$(mktemp /tmp/new_input.XXXXXXX.nc)
	# Do the work in the output directory (temp files are created with convert_tools.py. Run in a safer +rw location)
	cd $(dirname $output_file)

	regrid_weights_dir=$(dirname $WEIGHTS)

	# Horizontal regridding using conserve weights (if no weights have been generated, the grid_dir flag
	# tells ncremap to store weights in that directory.
    echo "============================================================"
	ncpdq -O -U $input_file $UNPACKED_TEMP
	python $regrid_tools/convert_tool.py -n $UNPACKED_TEMP -o $TMPFILE -m conserve -i 576 -j 361 -d DC -p PC --grid_dir $regrid_weights_dir

   mv $TMPFILE $output_file
   ls $output_file
}


function update_unit {
	local process_dir=$1
	file_id="asm_inst_1hr"
	varname="TO3"
	new_unit="Dobson"

	for file_to_change in $(find $process_dir -maxdepth 1 -name "*${file_id}*.nc4"); do
		ncdump -h $file_to_change | grep units | grep $varname;err=$?
		echo $file_to_change
		if [ $err -eq 0 ]; then
       	ncatted -O -a units,$varname,m,c,$new_unit  $file_to_change 
		else
			echo "ERROR:  $varname and units not found in $file_to_change"
		fi
	done
}

export -f update_unit

function main {
	# Input files should be in parent directory. This code sorts (e.g. 2019-01-01/0000/C180)
	# Output files will be organized in dated/timestamped directory (e.g. 2019-01-01/0000/)
	count=0
	for input_file in $(find $in_dir -maxdepth 1 -name "*C180*.nc4"); do
		count+=1
		echo "$count: $input_file \\n"
		new_input_path=`_create_directory_from_parsed_input_file $in_dir $input_file $sorting_flag`
		out_dir=$(dirname $(dirname $new_input_path))
		levels=`echo $input_file | awk -F"C180x180x6_" '{print substr($2,1,3)}'`
      	out_almost=`echo $new_input_path | awk -F"C180x180x6" '{ print $1 "L576x361" $2}'`
			out_fn="${out_almost/C180\//""}"
   		echo "Regrid to LAT/LON: $out_fn \\n"
   		regrid $new_input_path $out_fn
		if [ "${levels}" == "v72" ]; then
			vertical_regrid $out_fn
		fi
	done
	
	if [[ $count -eq 0 ]]; then
		set -o noglob
		echo "Ooops:  No files found in $in_dir matching regex '*C180*.nc4'"
		set -o noglob
		exit 1
	fi

	cp_const $out_dir
}
export -f main

if [ "$1" == "-h" ]; then
        usage
else
       setup "$@"
fi

exit 0

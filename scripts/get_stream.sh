function get_stream {
    YMD=$1
    if [[ ${YMD} -lt 19920101 ]]; then
	STREAM=100
    elif [[ ${YMD} -lt 20010101 ]]; then
       	STREAM=200
    elif [[ ${YMD} -lt 20110101 ]]; then
        STREAM=300
    else
    	STREAM=400
    fi
    export $STREAM
}

function any_stream {
    TARGET_FILE=$1
    BASEURL=$2
    TARGET_REGEX=`echo "${TARGET_FILE/${STREAM}/[0-4]0[0-1]}"`
    # try to get any stream.
    wget_cmd="wget -nv --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r --no-parent --no-directories -A ${TARGET_REGEX} ${BASEURL}/${YYYY}/${MM}/"
    eval $wget_cmd
    if [ $? == 0 ]; then
	    TARGET_REGEX=`find . -name ${TARGET_REGEX} -print`
	    echo $TARGET_REGEX
	    python ${BASE}/python/test_dataset.py $TARGET_REGEX
	    if [ $? != 0 ]; then
		    eval $wget_cmd
	    fi
	    python ${BASE}/python/test_dataset.py $TARGET_REGEX
	    if [ $? != 0 ]; then
		    echo "REMOVE ${TARGET_REGEX} second wget attempt also failed to produce a readable file."
		    rm ${TARGET_REGEX}
		    echo "Run $wget_cmd to try again."
            fi
    fi
}

function check_before_any_stream_call {
	LOCAL_DIR=$1
	TARGET_FILE=$2
	REANALYSIS=$2
	if [ -s "${LOCAL_DIR}/${TARGET_FILE}" ] || [ -s "${LOCAL_DIR}/${REANALYSIS}" ];  then
            echo "${TARGET_FILE} exists"
	else
            any_stream $TARGET_FILE $BASEURL
	fi
}

function get_dataset {
	mkdir -p $LOCAL_DIR
	cd $LOCAL_DIR
	check_before_any_stream_call $LOCAL_DIR $TARGET_FILE $REANALYSIS
}


function inst1_2d_asm_Nx {
	export TARGET_FILE=MERRA2_${STREAM}.inst1_2d_asm_Nx.${YYYY}${MM}${DD}.nc4
	export REANALYSIS=MERRA2_401.inst1_2d_asm_Nx.${YYYY}${MM}${DD}.nc4
	export BASEURL=https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I1NXASM.5.12.4
	export LOCAL_DIR=2d_asm
	get_dataset
}

function inst3_3d_asm_Np {
	export TARGET_FILE=MERRA2_${STREAM}.inst3_3d_asm_Np.${YYYY}${MM}${DD}.nc4
	export REANALYSIS=MERRA2_401.inst3_3d_asm_Np.${YYYY}${MM}${DD}.nc4
	export BASEURL=https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NPASM.5.12.4
	export LOCAL_DIR=3d_asm
        get_dataset
}

function inst6_3d_ana_Np {
	export TARGET_FILE=MERRA2_${STREAM}.inst6_3d_ana_Np.${YYYY}${MM}${DD}.nc4
	export REANALYSIS=MERRA2_401.inst6_3d_ana_Np.${YYYY}${MM}${DD}.nc4
	export BASEURL=https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I6NPANA.5.12.4
	export LOCAL_DIR=3d_ana
    	get_dataset
}

function inst6_3d_ana_Nv {
	export TARGET_FILE=MERRA2_${STREAM}.inst6_3d_ana_Nv.${YYYY}${MM}${DD}.nc4
	export REANALYSIS=MERRA2_401.inst6_3d_ana_Nv.${YYYY}${MM}${DD}.nc4
	export BASEURL=https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I6NVANA.5.12.4/
	export LOCAL_DIR=3d_ana
    	get_dataset
}

function tavg1_2d_flx_Nx {
	export TARGET_FILE=MERRA2_${STREAM}.tavg1_2d_flx_Nx.${YYYY}${MM}${DD}.nc4
	export REANALYSIS=MERRA2_401.tavg1_2d_flx_Nx.${YYYY}${MM}${DD}.nc4
	export BASEURL=https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXFLX.5.12.4
	export LOCAL_DIR=2d_flx
    	get_dataset
}

function tavg1_2d_lnd_Nx {
	export TARGET_FILE=MERRA2_${STREAM}.tavg1_2d_lnd_Nx.${YYYY}${MM}${DD}.nc4
	export REANALYSIS=MERRA2_401.tavg1_2d_lnd_Nx.${YYYY}${MM}${DD}.nc4
	export BASEURL=https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXLND.5.12.4
	export LOCAL_DIR=2d_lnd
    	get_dataset
}

function tavg1_2d_rad_Nx {
	export TARGET_FILE=MERRA2_${STREAM}.tavg1_2d_rad_Nx.${YYYY}${MM}${DD}.nc4
	export REANALYSIS=MERRA2_${STREAM}.tavg1_2d_rad_Nx.${YYYY}${MM}${DD}.nc4
	export BASEURL=https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXRAD.5.12.4
	export LOCAL_DIR=2d_rad
    	get_dataset
}

function tavg1_2d_slv_Nx {
	export TARGET_FILE=MERRA2_${STREAM}.tavg1_2d_slv_Nx.${YYYY}${MM}${DD}.nc4
	export REANALYSIS=MERRA2_${STREAM}.tavg1_2d_slv_Nx.${YYYY}${MM}${DD}.nc4
	export BASEURL=https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXSLV.5.12.4
	export LOCAL_DIR=2d_slv
    	get_dataset
}

YYYY=$1
MM=$2
DD=$3
let YMD=${YYYY}${MM}${DD}
get_stream $YMD

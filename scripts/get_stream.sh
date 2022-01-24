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
}

function any_stream {
    TARGET_FILE=$1
    BASEURL=$2
    TARGET_REGEX=`echo "${TARGET_FILE/${STREAM}/[0-4]0[0-1]}"`
    # try to get any stream.
    wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r --no-parent --no-directories -A ${TARGET_REGEX} ${BASEURL}/${YYYY}/${MM}/

    if [ $? == 0 ]; then
	    TARGET_FILE=`ls ${TARGET_REGEX}`
    fi
}

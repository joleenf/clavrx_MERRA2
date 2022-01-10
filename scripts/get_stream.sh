function get_stream {
    YMD=$1
    if [[ ${YMD} -lt 19920101 ]]; then
	STREAM=100
    elif [[ ${YMD} -lt 20010101 ]]; then
       	STREAM=200
    elif [[ ${YMD} -lt 20110101 ]]; then
        STREAM=300
    elif [[ ${YMD} -lt 20210601 ]]; then
	STREAM=400
    else
    	STREAM=401
    fi
}

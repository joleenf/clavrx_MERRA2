# This code has been written to cast the rh from double to float32 and land mask from long back to float32.
# A mistake was noticed in October2024.  These files were crashing clavrx because of incorrect dtypes.  
# This code has been written to rerun merra2 from 2024-backwards to 1980.  Then goes-fp as well.

import numpy as np
import os
import sys
from conversions.conversion_class import output_dtype
from conversions.file_tools import write_global_attributes
from pyhdf.SD import SD, SDC
from textwrap import dedent

LINEBREAK = "\n"

def rebuild_variable(sds, out_hdf, replace_data):
    # Create a dataset named 'd1' to hold 
    info = sds.info()
    if replace_data is None:
        data = sds[:]
    else:
        data = replace_data

    sdc_dtype = output_dtype(info[0], data.dtype)
    d1 = out_hdf.create(info[0], sdc_dtype, info[2])
    for key,value in sds.attributes().items():
        setattr(d1, key, value)
    for index, (key, value) in enumerate(sds.dimensions().items()):
        new_dim = d1.dim(index)
        #print("Dimensions set for ", info, new_dim, key)
        new_dim.setname(key)
    try:
        d1[:] = data
    except ValueError as e:
        print(info[0])
        print(data)

    return d1


def repair_merra(fileName, process_loc="/ships22/cloud/Ancil_Data/clavrx_ancil_data/dynamic/MERRA_INPUT/repair"):
    """Mainly copy processed file by change dtype on rh fields and land mask.
    
    fileName:  full path of the file that needs repairing
    """
    repaired = os.path.join(process_loc, os.path.basename(fileName))

    print(f"Opening {fileName}")
    hdfFile = SD(fileName)
    print(f"Success opening {fileName}")
    new_hdfFile = SD(repaired,SDC.WRITE|SDC.CREATE)


    for varname in hdfFile.datasets():
        val = hdfFile.select(varname)
        if varname in ["rh", "rh at sigma=0.995", "land mask"]:
            new_data = val[:].astype(np.float32)
        else:
            new_data = None
        new_var = rebuild_variable(val, new_hdfFile, replace_data=new_data) 
        val.endaccess()
        new_var.endaccess()

    global_attrs = hdfFile.attributes()
    kl = global_attrs.keys()
    for kn in ["CVS tag: GEOSadas-5_12_4_p38_SLES12_M2-OPS STREAM", "GEOSadas-5_12_4_p38_SLES12_M2-OPS STREAM"]:
        if kn in kl:
            key = kn.split("STREAM")[0].strip()
            try:
                global_attrs["GranuleID"] = global_attrs.pop(f"{key} STREAM")
                global_attrs["History"] = global_attrs.pop(f"{key} History")
            except KeyError as e:
                print(f"{kn} in {kn in kl}")
                raise KeyError(e)
        else:
            continue
    # Eliminate this source because it creates a tangled mess in the global attributes
    # global_attrs["Source"] = "GEOSadas-5_12_4_p38_SLES12_M2-OPS"
    # write global attributes closes the output file
    write_global_attributes(new_hdfFile, global_attrs)
    
    hdfFile.end()

    return repaired

def run_repair(input_path):
    path_repaired = list()
    if os.path.isdir(input_path):
        print("directory")
        original_directory = input_path
        for fn in os.listdir(original_directory):
            print(f"repair_merra  on {fn}")
            fn = os.path.join(original_directory, fn)
            path_repaired.append(repair_merra(fn))
    elif os.path.isfile(input_path):
        original_directory = os.path.dirname(input_path)
        print(original_directory)
        path_repaired.append(repair_merra(input_path))
    else:
        raise ValueError("First argument must either be a file path or directory.")

    move_list = "/data/Personal/clavrx_ops/MERRA_OUTPUT/move_list.txt"
    scr_fn = "/data/Personal/clavrx_ops/MERRA_OUTPUT/move_scr.sh"
    with open(move_list, "w+") as f:
        for out_fn in path_repaired: 
                f.write("{}{}".format(out_fn, LINEBREAK))
    with open(scr_fn, "w+") as scr:
        scr.write("original_directory={}".format(original_directory))
        scr.write(dedent(
        """
        while read out_fn
        do
            eval "$DUMP_HDF4 ${out_fn}";err=$?
            if [ $err == 0 ]; then
                 mv ${out_fn} ${original_directory}
            else
                 echo "Error opening $out_fn"
            fi
        """
        ))
        scr.write("done <{}".format(move_list))

        print("Run /data/Personal/clavrx_ops/MERRA_OUTPUT/move_scr.sh")


def check_repair(data_path):
    """Check repair on a data path."""
    for fn in os.listdir(data_path):
        fn = os.path.join(data_path, fn)
        hdfFile = SD(fn)
        for varname in hdfFile.datasets():
            val = hdfFile.select(varname)
            if varname in ["rh", "rh at sigma=0.995", "land mask"]:
                if val.info()[3] != SDC.FLOAT32:
                    print(f"Needs attention:  {fn}")
                    run_repair(fn)



if __name__ == "__main__":
    #  Argv[1]:  path of data that needs repair, Argv[2]: temporary output path 
    if sys.argv[1].upper() != "X":
        run_repair(sys.argv[1])
    if sys.argv[2].upper() != "X":
        check_repair(sys.argv[2])

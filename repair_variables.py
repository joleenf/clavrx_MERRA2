# This code has been written to cast the rh from double to float32 and land mask from long back to float32.
# A mistake was noticed in October2024.  These files were crashing clavrx because of incorrect dtypes.  
# This code has been written to rerun merra2 from 2024-backwards to 1980.  Then goes-fp as well.
import numpy as np
from conversions.conversion_class import output_dtype
from conversions.file_tools import write_global_attributes
from pyhdf.SD import SD, SDC

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
        print("Dimensions set for ", info, new_dim, key)
        new_dim.setname(key)
    d1[:] = data
    print(d1[:].dtype)
    print(d1.attributes())

    return d1


fileName="conversions/merra.24082300_F000.hdf"
repaired="/data/Personal/clavrx_ops/tmp/repair.24082300_F000.hdf"

hdfFile = SD(fileName)
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
if "CVS tag: GEOSadas-5_12_4_p38_SLES12_M2-OPS STREAM" in global_attrs.keys():
    global_attrs["GranuleID"] = global_attrs.pop("CVS tag: GEOSadas-5_12_4_p38_SLES12_M2-OPS STREAM")
    global_attrs["History"] = global_attrs.pop("CVS tag: GEOSadas-5_12_4_p38_SLES12_M2-OPS History")
    global_attrs["Source"] = "GEOSadas-5_12_4_p38_SLES12_M2-OPS"
# write global attributes closes the output file
write_global_attributes(new_hdfFile, global_attrs)

hdfFile.end()

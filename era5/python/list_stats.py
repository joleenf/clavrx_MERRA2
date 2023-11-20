import sys
import numpy as np
import netCDF4

def variable_stats(nc_filename: str, item: str) -> str:

    ds = netCDF4.Dataset(nc_filename)

    stats = ds.variables[item]
    data_array = np.ma.getdata(stats)

    min_value=np.min(data_array)
    max_value=np.max(data_array)
    fill=data_array.fill_value

    #print(f"{nc_filename} varname:{item} --> {min_value} {max_value} with fill of {fill}")
    print(f"{nc_filename} varname:{item} --> {min_value} {max_value}")

if __name__ == "__main__":
    fn = sys.argv[1]
    var_name = sys.argv[2]
    ds = variable_stats(fn, var_name)

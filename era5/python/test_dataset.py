import netCDF4
import sys

fn = sys.argv[1]

test = netCDF4.Dataset(fn)

print("{} is loadable by netCDF4.Dataset".format(fn))

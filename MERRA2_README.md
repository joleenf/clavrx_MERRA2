1. You may see a warning of the form:

"UserWarning: WARNING: valid_range not used since it cannot be safely cast to variable data type"

This is not a fatal warning. It just means that when python pulls data from the netCDF file during
the enumerate() function, it does not understand the .valid_range attribute (which is used to
mask data) and therefore it isn't carrying it along. Since the time information in the enumerate()
should not be masked to begin with, this *shouldn't* be a concern.

2. We are using FRSEAICE (sea ice-fraction) as the output 'ice fraction' variable, defined in the
FLX file around Lines 235-241 of merra24clavrx.py. This is done because in comparison to GFS .hdf
files, it appears the 'ice fraction' variable from GFS is only sea-ice fraction. In addition, the
MERRA2 ice-fraction variable (FRACI) appears to be broken with fill-values over the ocean, the
entirety of Antarctica, and most of Greenland. This variable is being collected and passed along to
.hdf output as 'FRACI' for later fix/use.

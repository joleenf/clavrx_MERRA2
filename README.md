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
  
3. Input files are pulled from /clavrx-merra2/MERRA2_FILES/wget_all.sh.
For background information and setting up GES DISC creditionals to use wget on NASA servers, see
*[ How to Download Data Files from HTTPS Service with wget ]* https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Download%20Data%20Files%20from%20HTTPS%20Service%20with%20wget


The script calls nine wget scripts to do the full pull-down. Not all nine files are  
currently being used by /clavrx-merra2/merra2/merra24clavrx.py to produce an output file. The  
current formulation does not make use of the inst6_3d_ana_Nv file, which contains analysis data on  
hybrid sigma-pressure coordinate surfaces. This file was intended to provide the data requested at  
the 0.9950 sigma-level, or a level close to it, but it looks like the 10m data is being substituted  
and it's working fine.  
  
4. Currently running the program for one year/month/day at a time using  
/clavrx-merra2/merra2/test_merra24clavrx.sh. The run-script generates the appropriate conda environment  
using merra2_clavrx, which uses the merra2 environment from [merra2_clavrx.yml ]https://github.com/joleenf/clavrx_MERRA2/blob/merra2/merra2_clavrx.yml

``` conda create -f merra2_clavrx.yml ```
  
  
The largest overhaul that I had to do on the original MERRA code was to switch from pyhdf routines to  
netCDF4 routines, because MERRA2 is natively stored in netCDF files and the old pyhdf routines aren't  
working.  
  
After running the run-script for a selected INPUT_DATE, files will appear in /tmp/out/YYYY/. This  
directory is currently being removed every time the run-script is used, so if you iterate over  
multiple dates, you'll scrub the output on every intermediate date and only have the output for the  
last date. You can get rid of this by commenting-out Lines 24-27 of test_merra24clavrx.sh.

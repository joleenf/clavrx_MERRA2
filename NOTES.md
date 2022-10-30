##Notes:

The bash script [run_merra2_4clavrx.sh](run_merra2_4clavrx.sh) runs [merra2/merra24clavrx_brett.py](merra2/merra24clavrx_brett.py) python code.  This 
code has a redundant function (download_data) to
    - check the input directory and download missing data
after the input collection has been inventories (build_input_collection) the code
    - processes the MERRA2 input data (make_merra_one_day)
    - creates HDF output

1. Input files can be retrieved within the python code, or using [scripts/wget_all.sh](scripts/wget_all.sh) independently. Though is is advisable to use the run_merra24.sh script for running both the wget_all.sh and merra24clavrx.py pieces in the correct order.
For more information on retrieving MERRA2 files from GES-DESC, please read 
*[ How to Download Data Files from HTTPS Service with wget ]*(https://disc.gsfc.nasa.gov/information/howto?keywords=Wget&page=1)

- [scripts/wget_all.sh](scripts/wget_all.sh) script calls nine wget scripts.
  -   Some of the data is not used.  The  current python code does not use the inst6_3d_ana_Nv collection.
    The inst6_3d_ana_Nv analysis is on hybrid sigma-pressure coordinate surfaces.
    This collection could provide the data requested at or near the 0.9950 sigma-level.
    The 10m data is currently substituted for the 0.995 sigma-level and is working fine.
  

2. The python code produces the  following warning:
  WARNING:  GES-DESC data access can fail or be incomplete.  This will cause the code to fail late.

> "UserWarning: WARNING: valid_range not used since it cannot be safely cast to variable data type"  
  
When netCDF.Dataset loads the data mask variable, the .valid_range attribute can not be parsed by the enumerate function.  Therefore, the .valid_range is not applied.  Since the time information in the enumerate()  
should not be masked anyway, this *shouldn't* be a concern.  
  
3. FRSEAICE (sea ice-fraction) is the output 'ice fraction' variable, defined in the  
FLX file. This is done to replicate the GFS where It appears the 'ice fraction' variable from GFS is only sea-ice fraction.
  
# ERA-5
ERA5 request following [copernicus API instructions] (https://cds.climate.copernicus.eu/api-how-to#install-the-cds-api-key)
```python
#!/usr/bin/env python
import cdsapi
c = cdsapi.Client()
c.retrieve("reanalysis-era5-pressure-levels",
{
"variable": "temperature",
"pressure_level": "1000",
"product_type": "reanalysis",
"year": "2008",
"month": "01",
"day": "01",
"time": "12:00",
"format": "grib"
}, "download.grib")

```

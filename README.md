##History and Purpose
This code began as a tool to process MERRA reanalysis datasets to calculate fields, and write them in a format similar to GFS forecast data.  The intent is to use these reanalysis datasets as input to the CLAVRx cloud processing system in lieu of GFS data.  The project has evolved to read MERRA2 reanalysis data and includes plans for processing ERA5 data.

##Running Code:
The python code needs the appropriate conda environment from [merra2_clavrx.yml ](merra2_clavrx.yml).   Run conda create once to create the environment from the yml file.

``` conda create -f merra2_clavrx.yml```

The conda environment has been created. Make the environment active.  Type:
``` conda activate merra2_clavrx```

Edit the bash script [run_merra24.sh](run_merra24.sh)
 - See documentation within the bash script.
 - Change the environment variables as needed. 

Run the bash script
  ```bash run_merra24.sh```

  Using:  ```bash run_merra24.sh``` to run the python code is recommended
  Advantages of bash script:
  - Run dates can be updated by the user in vim 
  - downloads merra files
  - creates an inventory log of succes/failure both for data download and final product completion
  - handles input files removal after products are created even when code exits early.

##Notes:

The bash script runs [merra24clavrx.py](merra24clavrx.py) python code.  This 
code has a redundant function (download_data) to
    - check the input directory and download missing data
after the input collection has been inventories (build_input_collection) the code
    - processes the MERRA2 input data (make_merra_one_day)
    - creates HDF output

1. Input files can be retrieved within the python code, or using [scripts/wget_all.sh](scripts/wget_all.sh) independently. Though is is advisable to use the run_merra24.sh script for running both the wget_all.sh and merra24clavrx.py pieces in the correct order.
For more information on retrieving MERRA2 files from GES-DESC, please read 
*[ How to Download Data Files from HTTPS Service with wget ]*(https://disc.gsfc.nasa.gov/information/howto?keywords=Wget&page=1)

- [scripts/wget_all.sh](scripts/wget_all.sh) script calls nine wget scripts.
  -   Some of the data is not used.  The  current python code does not use the inst6_3d_ana_Nv file.
    The inst6_3d_ana_Nv analysis data is on hybrid sigma-pressure coordinate surfaces.
    This file could provide the data requested at or near the 0.9950 sigma-level.
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

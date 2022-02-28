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

Purpose
-------------------
This code began as a tool to process MERRA reanalysis datasets to calculate fields,
and write them in a format similar to GFS forecast data.
The intent is to use these reanalysis datasets as input to the CLAVRx cloud processing system.
The project has evolved to read MERRA2 reanalysis data.

Running Code:
-------------
The python code needs the appropriate conda environment from `merra2_clavrx.yml <merra2_clavrx.yml>`_.

1. Run conda create once to create the environment from the yml file.

   .. code-block:: bash

    conda env create -f merra2_clavrx.yml

2. Make the environment active.  Type:

   .. code-block:: bash

    conda activate merra2_clavrx

3. Follow `How to Download Files from HTTPS Service with wget <https://disc.gsfc.nasa.gov/information/howto?keywords=Wget&page=1>`_ to register, and set up user credential for wget data access from the GES-DISC.

4. Edit bash script `run_merra4clavrx.sh <run_merra4clavrx.sh>`_ to change variables as needed for file paths and run dates. Full months can be run with runMonth.sh, full years with runYear.sh

5. Run the bash script on the command line:

   .. code-block:: bash

    bash run_merra4clavrx.sh

   Using:  run_merra4clavrx.sh to call the python code is recommended
    Advantages of bash script:
    - Run dates can be updated by the user in vim
    - downloads merra files
    - creates an inventory log of succes/failure both for data download and final product completion
    - handles input files removal after products are created even when code exits early.

Checking Runs:
-------------
1. The inventory for any year is counted with
   .. code-block:: bash

    bash scripts/runInventory.sh <YEAR>

2. Missing files can be listed with
   .. code-block:: bash

   bash scripts/list_missing_inventory.sh <YEAR> <MONTH>

Both scripts have default directories which are searched for the merra2 output files.  If these need to be changed, refer to script documentation using a "-h" flag.
runInventory will run on current year if nothing is entered when called from the command line without a year or flag.

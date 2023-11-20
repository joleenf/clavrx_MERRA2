import os
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
from glob import glob as glob
from conversions.tropopause_height import ERA_TROPO_TEMPERATURE

# filter_by_keys={'typeOfLevel': 'maxWind'}
# filter_by_keys={'typeOfLevel': 'tropopause'}
# filter_by_keys={'typeOfLevel': 'isobaricInhPa'}
# filter_by_keys={'typeOfLevel': 'meanSea'}
# filter_by_keys={'typeOfLevel': 'heightAboveGround'}


navgem_data = "/Users/joleenf/data/clavrx/navgem/navgem_*.grib2"
model_run = glob(navgem_data)[0]
tropo_data = xr.open_dataset(model_run, engine="cfgrib",
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'tropopause'}})

isobaric_data = xr.open_dataset(model_run, engine="cfgrib",
                                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})

t2m = xr.open_dataset(model_run, engine="cfgrib",
                      backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround'}})

prmsl = xr.open_dataset(model_run, engine="cfgrib",
                        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}})
print("Hello")
import os
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt

from tp_height import ERA_TROPO_PRESSURE

pressure_levels = "/Users/joleenf/data/clavrx/era5/2021/2021_08_01/2021_08_01_pressureLevels_hourly_1200.nc"

data_press = xr.open_dataset(pressure_levels, engine="netcdf4")
data_press = data_press.isel(time=0)
data_press = data_press.drop_vars("time")
data_press = data_press.rename({"latitude": "lat", "longitude": "lon"})

T = data_press["t"]
p = data_press["level"]
u = data_press["u"]
v = data_press["v"]
pv = data_press["pv"]

new_temp = list()
for i, (lat, lon, pname) in enumerate(latlons):
    temp_selection = T.sel(lat=[lat], lon=[lon], method="nearest")
    temp_selection = temp_selection.rename({"level": "z", "lon": "x", "lat": "y"})
    new_temp.append(temp_selection)

tpressure = ERA_TROPO_PRESSURE(new_temp[0]["z"], new_temp[0], "ZXY")
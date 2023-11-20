import xarray as xr
import numpy as np
from pyhdf.error import HDF4Error
from pyhdf.SD import SD, SDC, SDim

dim_names = {"x": "lon", "y": "lat", "z": "pressure"}

da1 = xr.DataArray(np.random.randn(2, 3, 3),
                   dims=("x", "y", "z"),
                   coords={"x": [-180, 180],
                           "y": [-10, 0, 10],
                           "z": [1000, 950, 900]},
                   name="da1").astype(np.float32)
da2 = xr.DataArray(np.random.randn(2, 3, 4),
                   dims=("x", "y", "z"),
                   coords={"x": [-180, 180],
                           "y": [-10, 0, 10],
                           "z": [975, 950, 925, 900]},
                   name="da2").astype(np.float32)

da3 = xr.DataArray(np.random.randn(2, 3),
                   dims=("x", "y"),
                   coords={"x": [-180, 180],
                           "y": [-10, 0, 10],
                           },
                   name="da3").astype(np.float32)

sd_out = SD("test.hdf", SDC.WRITE | SDC.CREATE | SDC.TRUNC)

for xr_da in [da1, da2, da3]:
    out_sds = sd_out.create(xr_da.name, SDC.FLOAT32, xr_da.shape)

    for dim in xr_da.dims:
        axis_num = xr_da.get_axis_num(dim)
        datasets = sd_out.datasets()

        final_dim_name = dim_names[dim]

        out_sds.dim(axis_num).setname(final_dim_name)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  This file is part of the conversion code to produce CLAVRx compatible files from
#  reanalysis data from MERRA and ERA5 datasets.
#  This is a general class which accepts NetCDF4 Dataset objects, reads and formats
#  the variables as needed for various variable conversions.
#
#  Mixture of fill choices for the data represents how the data was filled in older
#  code.  The extrapolation and vapor pressure routines rely on masked arrays to work
#  properly.  The extrapolation below surface will not fill to the lowest level when
#  nan values are present.  Masked arrays were filled with NaN, but as far as I can tell
#  that action produces no affect on a masked array.
#
#  Other functions handle final filling of data with CLAVRx fill values, reshaping
#  and extrapolation below the surface.
"""Conversion Class and Functions to handle Reanalysis Data read by netCDF4.Dataset."""
from __future__ import annotations

import logging
from typing import Optional, TypedDict, Union

import numpy as np
from pyhdf.SD import SDC

CLAVRX_FILL = 9.999e20
COMPRESSION_LEVEL = 6  # 6 is the gzip default; 9 is best/slowest/smallest fill

LOG = logging.getLogger(__name__)


def _reshape(data: np.ndarray, ndims_out: int, fill: Union[float, None]) -> np.ndarray:
    """Do a bunch of manipulation needed to make MERRA look like CFSR.

    * For arrays with dims (level, lat, lon) make level the last dim.
    * All CFSR fields are continuous but MERRA sets below-ground values to fill.
    * CFSR starts at 0 deg lon but merra starts at -180.
    """
    if (ndims_out == 3) or (ndims_out == 2):
        data = _shift_lon(data)

    if ndims_out != 3:
        return data
    # do extrapolation before reshape
    # (extrapolate fn depends on a certain dimensionality/ordering)
    data = _extrapolate_below_sfc(data, fill)
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 0, 1)
    data = data[:, :, ::-1]  # clavr-x needs toa->surface not surface->toa
    return data


def _refill(data: np.ndarray, old_fill: float) -> np.ndarray:
    """Assumes CLAVRx fill value instead of variable attribute."""
    if data.dtype in (np.float32, np.float64):
        data[np.isnan(data)] = CLAVRX_FILL
        data[data == old_fill] = CLAVRX_FILL

    return data


def _shift_lon_2d(data: np.ndarray) -> np.ndarray:
    """Assume dims are 2d and (lat, lon)."""
    nlon = data.shape[1]
    halfway = nlon // 2
    tmp = data.copy()
    data[:, 0:halfway] = tmp[:, halfway:]
    data[:, halfway:] = tmp[:, 0:halfway]
    return data


def _shift_lon(data: np.ndarray) -> np.ndarray:
    """Make lon start at 0deg instead of -180.

    Assume dims are (level, lat, lon) or (lat, lon)
    """
    if len(data.shape) == 3:
        for l_ind in np.arange(data.shape[0]):
            data[l_ind] = _shift_lon_2d(data[l_ind])
    elif len(data.shape) == 2:
        data = _shift_lon_2d(data)
    return data


def _extrapolate_below_sfc(t: np.ndarray, fill: float) -> np.ndarray:
    """Set below ground fill values to lowest good value instead of exptrapolation.

    Major difference between CFSR and MERRA is that CFSR extrapolates
    below ground and MERRA sets to fill. For now,
    """
    # Algorithm: For each pair of horizontal indices, find lowest vertical index
    #            that is not CLAVRX_FILL. Use this data value to fill in missing
    #            values down to bottom index.

    lowest_good = t[0] * 0.0 + fill
    lowest_good_ind = np.zeros(lowest_good.shape, dtype=np.int64)
    for l_ind in np.arange(t.shape[0]):
        t_at_l = t[l_ind]
        t_at_l_good = t_at_l != fill
        replace = t_at_l_good & (lowest_good == fill)
        lowest_good[replace] = t_at_l[replace]
        lowest_good_ind[replace] = l_ind

    for l_ind in np.arange(t.shape[0]):
        t_at_l = t[l_ind]
        # Limit extrapolation to bins below lowest good bin:
        t_at_l_bad = (t_at_l == fill) & (l_ind < lowest_good_ind)
        t_at_l[t_at_l_bad] = lowest_good[t_at_l_bad]

    return t


class CommandLineMapping(TypedDict):
    """Type hints for result of the argparse parsing."""

    start_date: str
    end_date: Optional[str]
    store_temp: bool
    base_path: str
    input_path: str


class ReanalysisConversion:
    """Handles extracting variables from netCDF4.Dataset."""

    def __init__(
            self,
            nc_dataset=None,
            in_name=None,
            out_name=None,
            out_units=None,
            ndims_out=None,
            time_ind=None,
            not_masked=True,
            nan_fill=False
    ) -> None:
        """Based on variable, adjust shape, apply fill and determine dtype."""
        self.nc_dataset = nc_dataset
        self.in_name = in_name
        self.out_name = out_name
        self.out_units = out_units
        self.ndims_out = ndims_out

        self.fill = self._get_fill
        self.data = self._get_data(time_ind, not_masked, nan_fill)

    def __repr__(self):
        """Report the name conversion when creating this object."""
        str_template = "Input name {} ==> Output Name: {}"
        return str_template.format(self[self.in_name].long_name, self.out_name)

    def __getitem__(self, item):
        """Access data in the NetCDF dataset variable by variable key."""
        return self.nc_dataset.variables[item]

    def out_name(self):
        """Return variable name as defined in the output file."""
        return self.out_name

    @property
    def _get_fill(self):
        """Get the fill value of this data."""
        if "_FillValue" in self[self.in_name].ncattrs():
            fill = self[self.in_name].getncattr("_FillValue")
        elif "missing_value" in self[self.in_name].ncattrs():
            fill = self[self.in_name].getncattr("missing_value")
        else:
            fill = None

        return fill

    def _get_data(self, time_ind, not_masked, nan_fill):
        """Get data and based on dimensions reorder axes, truncate TOA, apply fill."""
        data = np.ma.getdata(self[self.in_name])

        # want only dependent arrays as masked arrays, everything else can be a regular np.array.
        if not_masked:
            data = np.asarray(data.filled())

        if self.in_name == "lev" and len(data) != 42:
            # insurance policy while levels are hard-coded in unit conversion fn's
            # also is expected based on data documentation:
            # https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf
            raise ValueError(
                "Incorrect number of levels {} rather than 42.".format(len(data))
            )

        # select time indice.
        ndims_in = len(data.shape)
        if ndims_in in (3, 4):
            # note, vars w/ 3 spatial dims will be 4d due to time
            data = data[time_ind]

        # apply special cases
        if self.in_name in ("lev", "level"):
            data = data.astype(np.float32)
            data = np.flipud(data)  # clavr-x needs toa->surface
        elif self.in_name in ("lon", "longitude"):
            data = self._reorder_lon(data)
        else:
            pass

        if self.fill is not None:
            data = self.apply_fill(data, self.fill, self.out_name, nan_fill)

        return data

    @staticmethod
    def apply_fill(data: np.ndarray, fill_value, variable_name: str, nan_fill: bool):
        """Apply different fill value to data in special cases."""
        if variable_name == "water equivalent snow depth":
            #  Special case: set snow depth missing values to 0 matching CFSR behavoir.
            data[data == fill_value] = 0.0
        if nan_fill:
            data[data == fill_value] = np.nan   # no effect on masked arrays.
        else:
            pass  # default from old code and this matters for extrapolation below surface.

        return data

    @staticmethod
    def _reorder_lon(data):
        """Reorder longitude as needed for datasets.

        Merra2:  Stack halfway to end and then start to halfway.
        """
        tmp = np.copy(data)
        halfway = data.shape[0] // 2
        data = np.r_[tmp[halfway:], tmp[:halfway]]

        if data.max() > 180.:
            data = data - 180.

        return data

    @property
    def long_name(self):
        """Update long_name from input name if function changes output."""
        raise NotImplementedError

    @property
    def _create_output_dtype(self):
        """Convert between string and the equivalent SD.<DTYPE>."""
        nc4_dtype = self.data.dtype
        if (nc4_dtype == "single") | (nc4_dtype == "float32"):
            sd_dtype = SDC.FLOAT32
        elif (nc4_dtype == "double") | (nc4_dtype == "float64"):
            sd_dtype = SDC.FLOAT64
        elif nc4_dtype == "uint32":
            sd_dtype = SDC.UINT32
        elif nc4_dtype == "int32":
            sd_dtype = SDC.INT32
        elif nc4_dtype == "uint16":
            sd_dtype = SDC.UINT16
        elif nc4_dtype == "int16":
            sd_dtype = SDC.INT16
        elif nc4_dtype == "int8":
            sd_dtype = SDC.INT8
        elif nc4_dtype == "char":
            sd_dtype = SDC.CHAR
        else:
            raise ValueError("UNSUPPORTED NC4 DTYPE FOUND:", nc4_dtype)

        if self.out_name in ["pressure levels", "level"] and sd_dtype == SDC.FLOAT64:
            sd_dtype = SDC.FLOAT32  # don't want double

        return sd_dtype

    def _modify_shape(self):
        """Modify shape based on output characteristics."""
        if self.out_name == 'total ozone' and len(self.data.shape) == 3:
            # b/c we vertically integrate ozone to get dobson units here
            shape = (self.data.shape[1], self.data.shape[2])
        elif self.ndims_out == 3:
            # clavr-x needs level to be the last dim
            shape = (self.data.shape[1], self.data.shape[2], self.data.shape[0])
        else:
            shape = self.data.shape

        return shape

    def update_output(self, sd, in_file_short_value, data_array, out_fill):
        """Finalize output variables."""
        out_sds = sd["out"].create(self.out_name, self._create_output_dtype, self._modify_shape())
        out_sds.setcompress(SDC.COMP_DEFLATE, value=COMPRESSION_LEVEL)
        self.set_dim_names(out_sds)
        if self.out_name == "lon":
            out_sds.set(_reshape(data_array, self.ndims_out, None))
        else:
            LOG.info("Writing %s", self)
            out_sds.set(_refill(_reshape(data_array, self.ndims_out, out_fill), out_fill))

        if out_fill is not None:
            out_sds.setfillvalue(CLAVRX_FILL)
        if self.out_units is not None:
            out_sds.units = self.out_units

        if "units" in self.nc_dataset.variables[self.in_name].ncattrs():
            unit_desc = " in [{}]".format(self[self.in_name].units)
        else:
            unit_desc = ""
        out_sds.source_data = ("{}->{}{}".format(in_file_short_value,
                               self.in_name, unit_desc))
        out_sds.units = self[self.in_name].units

        out_sds.long_name = self.long_name()
        out_sds.endaccess()

    def set_dim_names(self, out_sds):
        """Set dimension names in hdf file."""
        if self.in_name == "lat" or self.in_name == "latitude":
            out_sds.dim(0).setname("lat")
        elif self.in_name == "lon" or self.in_name == "longitude":
            out_sds.dim(0).setname("lon")
        elif self.in_name == "lev" or self.in_name == "level":
            out_sds.dim(0).setname("level")
        elif self.ndims_out == 2:
            out_sds.dim(0).setname("lat")
            out_sds.dim(1).setname("lon")
        elif self.ndims_out == 3:
            out_sds.dim(0).setname("lat")
            out_sds.dim(1).setname("lon")
            out_sds.dim(2).setname("level")
        else:
            raise ValueError("unsupported dimensionality")

        return out_sds

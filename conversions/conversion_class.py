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
from typing import List, Optional, TypedDict, Union

import numpy as np
import pint
import pyhdf
from pyhdf.SD import SDC

from conversions import CLAVRX_FILL, COMPRESSION_LEVEL, Q_, ureg

LOG = logging.getLogger(__name__)


def output_dtype(out_name, nc4_dtype):
    """Convert between string and the equivalent SD.<DTYPE>."""
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

    if out_name in ["pressure levels", "level"] and sd_dtype == SDC.FLOAT64:
        sd_dtype = SDC.FLOAT32  # don't want double

    return sd_dtype


class CommandLineMapping(TypedDict):
    """Type hints for result of the argparse parsing."""

    start_date: str
    end_date: Optional[str]
    store_temp: bool
    base_path: str
    input_path: str
    files: List[str]
    local: List[str]


def pint_unit_from_str(unit_str):
    """Convert unit string to pint units using custom dictionary if necessary."""
    # a re.match would be better, this is simpler for me to read.
    convert_dict = {"kg kg-1": "kg/kg", "m s-2": (ureg.meters / ureg.seconds ** 2),
                    "m s-1": (ureg.meters / ureg.seconds),
                    "kg/m^2": (ureg.kg / ureg.meters ** 2),
                    "kg m-2": (ureg.kg / ureg.meters ** 2),
                    "m+2 s-2": (ureg.meters ** 2 / ureg.seconds ** 2)}

    if unit_str in convert_dict.keys():
        pint_unit = convert_dict[unit_str]
    else:
        try:
            pint_unit = ureg.Unit(unit_str)
        except TypeError as pint_error:
            raise TypeError(f"{pint_error} from {unit_str}")

    return pint_unit


class ReanalysisConversion:
    """Handles extracting variables from netCDF4.Dataset."""

    def __init__(
            self,
            nc_dataset=None,
            shortname=None,
            out_name=None,
            out_units=None,
            ndims_out=None,
            not_masked=True,
            nan_fill=False,
            time_ind=None
    ) -> None:
        """Based on variable, adjust shape, apply fill and determine dtype."""
        self.nc_dataset = nc_dataset
        self.shortname = shortname
        self.out_name = out_name
        self.out_units = out_units
        self.ndims_out = ndims_out

        self.fill = self._get_fill
        self.data = self._get_data(not_masked, nan_fill, time_ind)

    def __repr__(self):
        """Report the name conversion when creating this object."""
        str_template = "Input name {} ==> Output Name: {}"

        return str_template.format(self[self.shortname].long_name, self.out_name)

    def __getitem__(self, item):
        """Access data in the NetCDF dataset variable by variable key."""
        return self.nc_dataset.variables[item]

    def out_name(self):
        """Return variable name as defined in the output file."""
        return self.out_name

    @property
    def _get_fill(self):
        """Get the fill value of this data."""
        if "_FillValue" in self[self.shortname].ncattrs():
            fill = self[self.shortname].getncattr("_FillValue")
        elif "missing_value" in self[self.shortname].ncattrs():
            fill = self[self.shortname].getncattr("missing_value")
        else:
            fill = None

        return fill

    def _get_data(self, not_masked, nan_fill, time_ind):
        """Get data and based on dimensions reorder axes, truncate TOA, apply fill."""
        data_var = self[self.shortname]
        print(f"Getting data for {self.shortname}")
        data = np.ma.getdata(data_var)

        # want only dependent arrays as masked arrays, everything else can be a regular np.array.
        if not_masked:
            data = np.asarray(data.filled())

        if self.shortname == "lev" and len(data) != 42:
            # insurance policy while levels are hard-coded in unit conversion fn's
            # also is expected based on data documentation:
            # https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf
            raise ValueError(
                "Incorrect number of levels {} rather than 42.".format(len(data))
            )

        # select time index.
        if "time" in data_var.dimensions:
            # note, vars w/ 3 spatial dims will be 4d due to time
            data = data[time_ind]

        # apply special cases
        if self.shortname in ("lev", "level"):
            data = data.astype(np.float32)
            data = np.flipud(data)  # clavr-x needs toa->surface
        elif self.shortname in ("lon", "longitude"):
            data = self._reorder_lon(self.shortname, data)

        if self.fill is not None:
            data = self.apply_fill(data, self.fill, self.out_name, nan_fill)

        # add units, make exception for units that are not defined in pint.
        if data_var.units not in ["degrees_north", "degrees_east",
                                  "1=land, 0=ocean, greenland and antarctica are land"]:
            pint_unit = pint_unit_from_str(data_var.units)
            data = Q_(data, pint_unit)

        return data

    def updateAttr(self, attr_name, replace_with):
        setattr(self, attr_name, replace_with)

    @staticmethod
    def apply_fill(data: np.ndarray, fill_value, variable_name: str, nan_fill: bool):
        """Apply different fill value to data in special cases."""
        if variable_name == "water equivalent snow depth":
            #  Special case: set snow depth missing values to 0 matching CFSR.
            data[data == fill_value] = 0.0
        if nan_fill:
            data[data == fill_value] = np.nan   # no effect on masked arrays.
        else:
            pass  # default from old code and this matters for extrapolation below surface.

        return data

    @staticmethod
    def _reorder_lon(in_name, data):
        """Reorder longitude as needed for datasets."""
        raise NotImplementedError

    @property
    def long_name(self):
        """Update long_name from input name if function changes output."""
        raise NotImplementedError

    def _modify_shape(self):
        """Modify shape based on output characteristics."""
        if len(self.data.shape) == 3:
            if self.out_name == 'total ozone':
                # b/c we vertically integrate ozone to get dobson units here
                shape = (self.data.shape[1], self.data.shape[2])
            else:
                # clavr-x needs level to be the last dim
                shape = (self.data.shape[1], self.data.shape[2], self.data.shape[0])
        else:
            shape = self.data.shape

        return shape

    def _reshape(self, fill: Union[float, None]) -> np.ndarray:
        """Do a bunch of manipulation needed to make MERRA look like CFSR.

        * For arrays with dims (level, lat, lon) make level the last dim.
        * All CFSR fields are continuous but MERRA sets below-ground values to fill.
        * CFSR starts at 0 deg lon but merra starts at -180.
        """
        data = self.data
        ndims_out = len(self.data.shape)
        if ndims_out in [2, 3]:
            data = self._shift_lon()

        if ndims_out == 3:
            # do extrapolation before reshape
            # (extrapolate fn depends on a certain dimensionality/ordering)
            data = self._extrapolate_below_sfc(data, fill)
            data = np.swapaxes(data, 0, 2)
            data = np.swapaxes(data, 0, 1)
            data = data[:, :, ::-1]  # clavr-x needs toa->surface not surface->toa
        return data

    @staticmethod
    def _refill(data: np.ndarray, old_fill: float) -> np.ndarray:
        """Assumes CLAVRx fill value instead of variable attribute."""
        if data.dtype in (np.float32, np.float64):
            data[np.isnan(data)] = CLAVRX_FILL
            data[data == old_fill] = CLAVRX_FILL
        print(data.max())
        return data

    @staticmethod
    def _shift_lon_2d(data: np.ndarray) -> np.ndarray:
        """Assume dims are 2d and (lat, lon)."""
        nlon = data.shape[1]
        halfway = nlon // 2
        tmp = data.copy()
        data[:, 0:halfway] = tmp[:, halfway:]
        data[:, halfway:] = tmp[:, 0:halfway]

        # get index where condition is true
        # create a deque? rotate n steps to right

        return data

    def _shift_lon(self) -> np.ndarray:
        """Make lon start at 0deg instead of -180.

        Assume dims are (level, lat, lon) or (lat, lon)
        """
        data = self.data.copy()
        if len(data.shape) == 3:
            for l_ind in np.arange(data.shape[0]):
                data[l_ind] = self._shift_lon_2d(data[l_ind])
        elif len(data.shape) == 2:
            data = self._shift_lon_2d(data)
        return data

    @staticmethod
    def _extrapolate_below_sfc(t: np.ndarray, fill: Optional[float]) -> np.ndarray:
        """Set below ground fill values to lowest good value instead of extrapolation.

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

    @property
    def get_units(self):
        """If possible, get units from ncattrs."""
        try:
            units = self.nc_dataset.variables[self.shortname].getncattr("units")
        except AttributeError:
            LOG.info(f"No Units found for {self.shortname}")
        if isinstance(self.data, pint.Quantity):
            units = str(self.data.units)
        return units

    def update_output(self, sd, in_file_short_value):
        """Finalize output variables."""
        print(f"Writing {self}")

        out_sds = sd["out"].create(self.out_name,
                                   output_dtype(self.out_name, self.data.dtype),
                                   self._modify_shape())

        out_sds.setcompress(SDC.COMP_DEFLATE, value=COMPRESSION_LEVEL)
        self.set_dim_names(out_sds)

        try:
            out_sds.units = self.out_units
        except pyhdf.error.HDF4Error as e:
            print(e)
            print(out_sds.units)

        if self.out_name == "lon":
            out_sds.set(self._reshape(fill=None))
        else:
            if isinstance(self.data, pint.Quantity):
                self.updateAttr("data", self.data.magnitude)
            else:
                self.updateAttr("data", self.data)
            out_data = self._refill(self._reshape(fill=self.fill), self.fill)
            out_sds.set(out_data)

        if self.fill is not None:
            out_sds.setfillvalue(CLAVRX_FILL)
        # if self.out_units is not None:
        #     out_sds.units = self.out_units
        # elif self.out_units in ("none", "None"):
        #     out_sds.units = "1"
        # else:
        #     out_sds.units = self.get_units

        #unit_desc = " in [{}]".format(out_sds.units)

        out_sds.source_data = f"{in_file_short_value}->{self.shortname}{out_sds.units}"

        out_sds.long_name = self.long_name()
        out_sds.endaccess()

    def set_dim_names(self, out_sds):
        """Set dimension names in hdf file."""
        if self.shortname == "lat" or self.shortname == "latitude":
            out_sds.dim(0).setname("lat")
        elif self.shortname == "lon" or self.shortname == "longitude":
            out_sds.dim(0).setname("lon")
        elif self.shortname == "lev" or self.shortname == "level":
            out_sds.dim(0).setname("level")
        elif self.ndims_out == 2:
            out_sds.dim(0).setname("lat")
            out_sds.dim(1).setname("lon")
        elif self.ndims_out == 3:
            out_sds.dim(0).setname("lat")
            out_sds.dim(1).setname("lon")
            out_sds.dim(2).setname("level")
        else:
            msg_str = "unsupported dimensionality ({}) for {} ==> {}."
            raise ValueError(msg_str.format(self.ndims_out,
                                            self.shortname,
                                            self.out_name))

        msg_str = "Out {} for {} ==> {}."
        msg_str = msg_str.format(out_sds.dimensions(),
                                 self.shortname,
                                 self.out_name)
        LOG.debug(msg_str)

        return out_sds


class MerraConversion(ReanalysisConversion):
    """Adjust longitude as appropriate for MERRA data."""

    @staticmethod
    def _reorder_lon(in_name, data):
        """Reorder longitude as needed for datasets.

        Merra2:  Stack halfway to end and then start to halfway.
        """
        if in_name in "lon":
            tmp = np.copy(data)
            halfway = data.shape[0] // 2
            data = np.r_[tmp[halfway:], tmp[:halfway]]
        else:
            raise ValueError("Unexpected Merra Longitude Variable name {}".format(in_name))

        return data

    def long_name(self):
        """Return long name from input file unless there is a special case."""
        if self.out_name == "height":
            long_name = "Geopotential Height"
        else:
            long_name = self[self.shortname].long_name

        return long_name

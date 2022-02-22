# -*- coding: utf-8 -*-
"""Module to convert MERRA2 re-analysis data to hdf files for use in CLAVR-x.

MERRA-2 Data is downloaded from the GES DISC server and converted to
output files which can be used as input into the CLAVR-x cloud product.

Example:
    Run the merra24clavrx.py code with a single <year><month><day>::

        $ python merra24clavrx.py 20010801

Optional arguments::
  -h, --help            show this help message and exit
  -e end_date, --end_date end_date
                        End date as YYYYMMDD not needed when processing one date. (default: None)
  -t, --tmp             Use to store downloaded input files in a temporary location. (default: False)
  -d [BASE_PATH], --base_dir [BASE_PATH]
                        Parent path used for input (in absence of -t/--tmp flag) and final location.
                        year subdirectory appends to this path.
                        (default: /apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2/)
  -v, --verbose         each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG (default: 0)

"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import OrderedDict
from pathlib import Path

import sys
import os
import logging
import subprocess
import tempfile

try:
    from pandas import date_range
    from pyhdf.SD import SD, SDC
    from netCDF4 import Dataset, num2date
    from datetime import datetime, timedelta
    from typing import Union, Optional, Dict, TypedDict
    import numpy as np
except ImportError as e:
    print("Import Error {}".format(e))
    print("Type:  conda activate merra2_clavrx")
    sys.exit(1)

np.seterr(all='ignore')

LOG = logging.getLogger(__name__)

comp_level = 6  # 6 is the gzip default; 9 is best/slowest/smallest file

FOCUS_VAR = ["rh"]
Q_ARR = [0, 0.25, 0.5, 0.75, 1.0]

# this is trimmed to the top CFSR level (i.e., exclude higher than 10hPa)
LEVELS = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700,
          650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50, 40,
          30, 20, 10, 7, 5, 4, 3, 2, 1, 0.7, 0.5, 0.4, 0.3, 0.1]  # [hPa]

TOP_LEVEL = 10  # [hPa] This is the highest CFSR level, trim anything higher.
CLAVRX_FILL = 9.999e20

OUT_PATH_PARENT = '/apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2/'


def quantile_stats(in_array):
    return np.nanquantile(in_array, Q_ARR)


def write_debug(data_array, where_msg, var_name):
    """For debugging purposes, print quartiles at various points in program."""

    if isinstance(data_array, np.ma.core.MaskedArray):
        data_array = np.ma.filled(data_array, np.nan)
    quantiles = quantile_stats(data_array)
    quantiles_string = "[{}]".format(', '.join(map(str, quantiles)))
    txt = "{} {} \n {}".format(var_name, where_msg, quantiles_string)
    LOG.info(txt)
    return quantiles


def no_conversion(a):
    return a


def fill_bad(a):
    return a * np.nan


# the merra4clavrx rosetta stone one key for each output var
OUTPUT_VARS_ROSETTA = {
    # --- data vars from 'inst6_3d_(ana)_Np'
    'MSL pressure': {
        'in_file': 'ana',
        'in_varname': 'SLP',
        'out_units': 'hPa',
        'units_fn': lambda a: a / 100.0,  # scale factor for Pa --> hPa
        'ndims_out': 2
    },
    'temperature': {
        'in_file': 'ana',
        'in_varname': 'T',
        'out_units': 'K',
        'units_fn': no_conversion,
        'ndims_out': 3
    },
    'surface pressure': {
        'in_file': 'ana',
        'in_varname': 'PS',
        'out_units': 'hPa',
        'units_fn': lambda a: a / 100.0,  # scale factor for Pa --> hPa
        'ndims_out': 2
    },
    'height': {
        'in_file': 'ana',
        'in_varname': 'H',
        'out_units': 'km',
        'units_fn': lambda a: a / 1000.0,  # scale factor for m --> km
        'ndims_out': 3
    },
    'u-wind': {
        'in_file': 'ana',
        'in_varname': 'U',
        'out_units': 'm/s',
        'units_fn': no_conversion,
        'ndims_out': 3
    },
    'v-wind': {
        'in_file': 'ana',
        'in_varname': 'V',
        'out_units': 'm/s',
        'units_fn': no_conversion,
        'ndims_out': 3
    },
    'rh': {
        'in_file': 'ana',
        'in_varname': 'QV',
        'out_units': '%',
        'units_fn': None,  # special case due to add'l inputs
        'ndims_out': 3
    },
    'total ozone': {
        'in_file': 'ana',
        'in_varname': 'O3',
        'out_units': 'Dobson',
        'units_fn': None,  # special case due to add'l inputs
        'ndims_out': 2
    },
    'o3mr': {
        'in_file': 'ana',
        'in_varname': 'O3',
        'out_units': 'kg/kg',
        'units_fn': no_conversion,
        'ndims_out': 3
    },
    # --- data vars from 'tavg1_2d_(slv)_Nx'
    'tropopause pressure': {
        'in_file': 'slv',
        'in_varname': 'TROPPT',
        'out_units': 'hPa',
        'units_fn': lambda a: a / 100.0,  # scale factor for Pa --> hPa
        'ndims_out': 2
    },
    'tropopause temperature': {
        'in_file': 'slv',
        'in_varname': 'TROPT',
        'out_units': 'K',
        'units_fn': no_conversion,
        'ndims_out': 2
    },
    'u-wind at sigma=0.995': {  # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'U10M',
        'out_units': 'm/s',
        'units_fn': no_conversion,
        'ndims_out': 2
    },
    'v-wind at sigma=0.995': {  # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'V10M',
        'out_units': 'm/s',
        'units_fn': no_conversion,
        'ndims_out': 2
    },
    'surface temperature': {  # XXX confirm skin temp is correct choice for 'surface temperature'
        'in_file': 'slv',
        'in_varname': 'TS',
        'out_units': 'K',
        'units_fn': no_conversion,
        'ndims_out': 2
    },
    'temperature at sigma=0.995': {  # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'T10M',
        'out_units': 'K',
        'units_fn': no_conversion,
        'ndims_out': 2
    },
    'rh at sigma=0.995': {  # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'QV10M',
        'out_units': '%',
        'units_fn': fill_bad,  # XXX how to get p at sigma=0.995 for RH conversion?
        'ndims_out': 2
    },
    'u-wind at 50M': {  # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'U50M',
        'out_units': 'm/s',
        'units_fn': no_conversion,
        'ndims_out': 2
    },
    'v-wind at 50M': {  # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'V50M',
        'out_units': 'm/s',
        'units_fn': no_conversion,
        'ndims_out': 2
    },
    # --- data vars from 'tavg1_2d_(flx)_Nx'
    'planetary boundary layer height': {
        'in_file': 'flx',
        'in_varname': 'PBLH',
        'out_units': 'km',
        'units_fn': lambda a: a / 1000.0,  # scale factor for m --> km
        'ndims_out': 2
    },
    'ice fraction': {
        'in_file': 'flx',
        'in_varname': 'FRSEAICE',
        'out_units': 'none',
        'units_fn': no_conversion,
        'ndims_out': 2
    },
    # --- data vars from 'tavg1_2d_(lnd)_Nx'
    'water equivalent snow depth': {
        'in_file': 'lnd',
        'in_varname': 'SNOMAS',
        'out_units': 'kg/m^2',
        'units_fn': no_conversion,  # special case in do_conversion will set fill values to zero
        'ndims_out': 2
    },
    # --- data vars from 'inst3_3d_(asm)_Np'
    'clwmr': {
        'in_file': 'asm3d',
        'in_varname': 'QL',
        'out_units': 'kg/kg',
        'units_fn': no_conversion,
        'ndims_out': 3
    },
    'cloud ice water mixing ratio': {
        'in_file': 'asm3d',
        'in_varname': 'QI',
        'out_units': 'kg/kg',
        'units_fn': no_conversion,
        'ndims_out': 3
    },
    # --- data vars from 'inst1_2d_(asm)_Nx'
    'total precipitable water': {
        'in_file': 'asm2d',
        'in_varname': 'TQV',
        'out_units': 'cm',
        'units_fn': lambda a: a / 10.0,  # scale factor for kg/m^2 (mm) --> cm
        'ndims_out': 2
    },
    # --- data vars from 'tavg1_2d_(rad)_Nx'
    'total cloud fraction': {
        'in_file': 'rad',
        'in_varname': 'CLDTOT',
        'out_units': 'none',
        'units_fn': no_conversion,
        'ndims_out': 2
    },
    # --- geoloc vars from 'ana'
    'lon': {
        'in_file': 'ana',
        'in_varname': 'lon',
        'out_units': None,
        'units_fn': no_conversion,
        'ndims_out': 1
    },
    'lat': {
        'in_file': 'ana',
        'in_varname': 'lat',
        'out_units': None,
        'units_fn': no_conversion,
        'ndims_out': 1
    },
    'pressure levels': {
        'in_file': 'ana',
        'in_varname': 'lev',
        'out_units': 'hPa',
        'units_fn': no_conversion,  # yes this is indeed correct.
        'ndims_out': 1
    },
    #  level is needed for 3D variable in python and McV, but clavrx needs pressure levels (reversed)
    'level': {
        'in_file': 'ana',
        'in_varname': 'lev',
        'out_units': 'hPa',
        'units_fn': no_conversion,  # yes this is indeed correct.
        'ndims_out': 1
    },
}


class CommandLineMapping(TypedDict):
    """Type hints for result of the argparse parsing."""
    start_date: str
    end_date: Optional[str]
    store_temp: bool
    base_path: str
    input_path: str


class MerraConversion(object):
    """ Based on the variable, adjust shape and perform operations on variable to stage for output."""

    def __init__(self, datasets, in_dataset, in_name, out_name, out_units,
                 units_fn, ndims_out, time_ind):
        self.in_dataset = in_dataset
        self.in_name = in_name
        self.out_name = out_name
        self.out_units = out_units
        # this function will be applied to data for units conversion
        self.units_fn = units_fn
        self.ndims_out = ndims_out
        self.time_ind = time_ind

        self.nc_dataset = datasets[in_dataset]
        LOG.info('Get data for {} {}'.format(self.in_dataset, self.in_name))

        self.data = self._get_data
        self.data_sd_type = self._create_output_dtype
        self.shape = self._modify_shape

    def __repr__(self):
        return "Input name {} ==> Output Name: {}".format(self.in_name, self.out_name)

    def __getitem__(self, item):
        """"Wrapper to access the data in the NetCDF dataset variable by variable key."""
        return self.nc_dataset.variables[item]

    @property
    def _get_data(self):
        """Get data and based on dimensions reorder axes, truncate TOA, apply fill."""

        data = np.ma.getdata(self[self.in_name])
        self.fill = data.get_fill_value()

        data = np.asarray(data)

        if self.in_name == 'lev' and len(data) != 42:
            # insurance policy while levels are hard-coded in unit conversion fn's
            # also is expected based on data documentation: https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf
            raise ValueError('Incorrect number of levels {} rather than 42.'.format(len(data)))

        ndims_in = len(data.shape)
        if ndims_in == 3:
            # note, vars w/ 3 spatial dims will be 4d due to time
            data = data[self.time_ind]
        if ndims_in == 4:
            data = data[self.time_ind]
            data = _trim_toa(data)

        # apply special cases
        if self.out_name in ['pressure levels', 'level']:
            data = data[0:len(LEVELS)].astype(np.float32)  # trim to top CFSR level
            data = np.flipud(data)  # clavr-x needs toa->surface
        elif self.out_name == 'lon':
            tmp = np.copy(data)
            halfway = data.shape[0] // 2
            data = np.r_[tmp[halfway:], tmp[:halfway]]
        else:
            pass

        if self.fill is not None:
            if self.out_name == 'water equivalent snow depth':
                #  Special case: set snow depth missing values to 0 matching CFSR behavoir.
                data = np.where(data == self.fill, 0.0, data)
            else:
                data = np.where(data == self.fill, np.nan, data)

        return data

    @property
    def _create_output_dtype(self):
        # BTH: netCDF4 uses strings to track datatypes - need to do a conversion
        #      between the string and the equivalent SD.<DTYPE>
        data_sd_type = nc4_to_sd_dtype(self.data.dtype)
        if self.out_name in ['pressure levels', 'level']:
            data_sd_type = SDC.FLOAT32  # don't want double

        return data_sd_type

    @property
    def _modify_shape(self):
        """Modify shape based on output characteristics."""
        if self.out_name == 'total ozone':
            # b/c we vertically integrate ozone to get dobson units here
            shape = (self.data.shape[1], self.data.shape[2])
        elif self.ndims_out == 3:
            # clavr-x needs level to be the last dim
            shape = (self.data.shape[1], self.data.shape[2], self.data.shape[0])
        else:
            shape = self.data.shape

        return shape

    def update_output(self, sd, data_array):
        """Finalize output variables."""
        ndims = self.ndims_out
        out_fill = self.fill

        out_sds = sd['out'].create(self.out_name, self.data_sd_type, self.shape)
        out_sds.setcompress(SDC.COMP_DEFLATE, value=comp_level)
        set_dim_names(self.in_name, self.ndims_out, out_sds)
        if self.out_name == 'lon':
            out_sds.set(_reshape(data_array, self.ndims_out, None))
        else:
            out_sds.set(_refill(_reshape(data_array, ndims, out_fill), out_fill))

        try:
            out_sds.setfillvalue(CLAVRX_FILL)
            if self.out_units is not None:
                setattr(out_sds, 'units', self.out_units)
            if 'units' in sd[self.in_dataset].variables[self.in_name].ncattrs():
                u = ' in [' + sd[self.in_dataset].variables[self.in_name].units + ']'
            else:
                u = ''
            setattr(out_sds, 'source_data',
                    'MERRA->' + self.in_dataset + '->' + self.in_name + u)
            setattr(out_sds, 'long_name', sd[self.in_dataset].variables[self.in_name].long_name)
            # setattr(out_sds, '_FillValue', CLAVRX_FILL)
            # setattr(out_sds, 'missing_value', in_sds.attributes()['missing_value'])
            # not sure about diff. btwn missing_value and fmissing_value
            # setattr(out_sds, 'fmissing_value', in_sds.attributes()['fmissing_value'])
        except KeyError:
            # dims will fail...  XXX don't want failed attribute to stop other attributes!
            pass
        out_sds.endaccess()


def set_dim_names(in_name, ndims_out, out_sds):
    if in_name == 'lat':
        out_sds.dim(0).setname('lat')
    elif in_name == 'lon':
        out_sds.dim(0).setname('lon')
    elif in_name == 'lev':
        out_sds.dim(0).setname('level')
    elif ndims_out == 2:
        out_sds.dim(0).setname('lat')
        out_sds.dim(1).setname('lon')
    elif ndims_out == 3:
        out_sds.dim(0).setname('lat')
        out_sds.dim(1).setname('lon')
        out_sds.dim(2).setname('level')
    else:
        raise ValueError("unsupported dimensionality")

    return out_sds


def qv_to_rh(qv, t, ps=None):
    """ Specific Humidity [kg/kg] -> relative humidity [%] """
    # See Petty Atmos. Thermo. 4.41 (p. 65), 8.1 (p. 140), 8.18 (p. 147)
    levels = map(lambda a: a * 100.0, LEVELS)  # [hPa] -> [Pa]

    # Saturation vapor pressure:
    #  http://faculty.eas.ualberta.ca/jdwilson/EAS372_13/Vomel_CIRES_satvpformulae.html
    # >273.16: w.r.t liquid water
    # 253.16 < T < 273.16: weighted interpolation of water / ice
    # < 253.16: w.r.t ice
    mix_lo = 253.16
    mix_hi = 273.16
    mix_ind = (t > mix_lo) & (t < mix_hi)
    ice_ind = t <= mix_lo
    es_wmo = 10.0 ** (10.79574 * (1. - 273.16 / t)
                      - 5.02800 * np.log10(t / 273.16)
                      + 1.50475 * 10. ** -4. * (1. - 10. ** (-8.2969 * (t / 273.16 - 1)))
                      + 0.42873 * 10. ** -3. * (10. ** (4.76955 * (1 - 273.16 / t)) - 1.)
                      + 0.78614) * 100.0  # [Pa]
    es_tot = es_wmo.copy()
    ei_gg = 10.0 ** (
            -9.09718 * (273.16 / t - 1.)
            - 3.56654 * np.log10(273.16 / t)
            + 0.876793 * (1. - t / 273.16)
            + np.log10(6.1071)
    ) * 100.0  # [Pa]
    es_tot[ice_ind] = ei_gg[ice_ind]
    liq_weight = (t - mix_lo) / (mix_hi - mix_lo)
    ice_weight = (mix_hi - t) / (mix_hi - mix_lo)
    emix = ice_weight * ei_gg + liq_weight * es_wmo
    es_tot[mix_ind] = emix[mix_ind]

    # Vapor pressure e, to "a good approximation":
    # e = qv / 0.622 # still need to multiply by pressure @ each level
    # or, using unapproximated form:
    e = 1.0 / (0.622 / qv + (1.0 - 0.622))  # still need to multiply by pressure @ each level
    if ps is None:
        # 3D RH field
        for i, lev in enumerate(levels):
            e[i, :, :] = e[i, :, :] * lev  # we've already cut out time dim
    else:
        # RH @ 10m: multiply by surface pressure
        e = e * ps
    rh = e / es_tot * 100.0  # relative humidity [%]
    rh[rh > 100.0] = 100.0  # clamp to 100% to mimic CFSR
    return rh


def kgkg_to_dobson(data):
    """Convert Ozone mixing ratio in [kg/kg] to dobson units."""
    du = 2.69e16  # TODO add units+doc from Mike's spreadsheet. (dobson units:  molecules per kilometer squared maybe?).
    g = 9.8
    av = 6.02e23
    mq = 0.048
    md = 0.028966
    const = 0.01 * av / (g * md)
    nlevels = data.shape[0]
    total = np.zeros(data[0].shape).astype('float32')
    for j in range(1, nlevels):
        mmr_j = data[j]
        mmr_j_1 = data[j - 1]
        good_j = np.logical_not(np.isnan(mmr_j))
        good_j_1 = np.logical_not(np.isnan(mmr_j_1))
        good = good_j & good_j_1
        vmr_j = mmr_j * md / mq
        vmr_j_1 = mmr_j_1 * md / mq
        ppmv = 0.5 * (vmr_j + vmr_j_1)
        dp = LEVELS[j - 1] - LEVELS[j]
        dobs_layer = ppmv * dp * const / du
        total[good] = (total[good] + dobs_layer[good])
    return total


def total_ozone(data, fill_value):
    """Return total ozone in dobson units."""
    dobson = kgkg_to_dobson(data)
    dobson[np.isnan(dobson)] = fill_value

    return dobson


def rh_at_sigma(temp10m, sfc_pressure, sfc_pressure_fill, data):
    temp_k = temp10m  # temperature in [K] (Y, X) not in (time, Y, X)

    sfc_pressure[sfc_pressure == sfc_pressure_fill] = np.nan  # pressure in [Pa]

    rh_sigma = qv_to_rh(data, temp_k, ps=sfc_pressure)
    rh_sigma[np.isnan(rh_sigma)] = sfc_pressure_fill

    return rh_sigma


def apply_conversion(units_fn, data, fill):
    """Special handling of converted data apply fill to converted data after function."""
    converted = data.copy()
    converted = units_fn(converted)

    if np.isnan(data).any():
        converted[np.isnan(data)] = fill

    return converted


def nc4_to_sd_dtype(nc4_dtype: np.dtype) -> int:
    # netCDF4 stores dtype as a string, pyhdf.SD stores dtype as a symbolic
    # constant. To properly convert, we need to go through an if-trap series
    # to identify the appropriate SD_dtype
    #
    # SD_dtype = nc4_to_sd_dtype(nc4_dtype)
    #
    # see, e.g. https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.int32
    # to troubleshoot when an unassigned nc4_dtype appears
    #

    if (nc4_dtype == 'single') | (nc4_dtype == 'float32'):
        sd_dtype = SDC.FLOAT32
    elif (nc4_dtype == 'double') | (nc4_dtype == 'float64'):
        sd_dtype = SDC.FLOAT64
    elif nc4_dtype == 'uint32':
        sd_dtype = SDC.UINT32
    elif nc4_dtype == 'int32':
        sd_dtype = SDC.INT32
    elif nc4_dtype == 'uint16':
        sd_dtype = SDC.UINT16
    elif nc4_dtype == 'int16':
        sd_dtype = SDC.INT16
    elif nc4_dtype == 'int8':
        sd_dtype = SDC.INT8
    elif nc4_dtype == 'char':
        sd_dtype = SDC.CHAR
    else:
        raise ValueError("UNSUPPORTED NC4 DTYPE FOUND:", nc4_dtype)
    return sd_dtype


def _reshape(data, ndims_out, fill):
    """ Make MERRA look like CFSR:
    
      * Convert array dimensions of (level, lat, lon) to (lat, lon, level)
      * CFSR fields are continuous but MERRA below-ground values are set to fill.
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


def _refill(data, old_fill):
    """Clavr-x assumes a fill value instead of reading from attributes"""
    if (data.dtype == np.float32) or (data.dtype == np.float64):
        if np.isnan(data).any():
            data[np.isnan(data)] = CLAVRX_FILL
        data[data == old_fill] = CLAVRX_FILL
    return data


def _shift_lon_2d(data):
    """ Helper function that assumes dims are 2d and (lat, lon) """
    nlon = data.shape[1]
    halfway = nlon // 2
    tmp = data.copy()
    data[:, 0:halfway] = tmp[:, halfway:]
    data[:, halfway:] = tmp[:, 0:halfway]
    return data


def _shift_lon(data):
    """ Make lon start at 0deg instead of -180. 
    
    Assume dims are (level, lat, lon) or (lat, lon)
    """
    if len(data.shape) == 3:
        for l_ind in np.arange(data.shape[0]):
            data[l_ind] = _shift_lon_2d(data[l_ind])
    elif len(data.shape) == 2:
        data = _shift_lon_2d(data)
    return data


def _extrapolate_below_sfc(t, fill):
    """Major difference between CFSR and MERRA is that CFSR extrapolates
       below ground and MERRA sets to fill. For now, set below
       ground fill values to lowest good value instead of exptrapolation.
    """
    # Algorithm: For each pair of horizontal indices, find the lowest vertical index
    #            that is not CLAVRX_FILL. Use this data value to fill in missing
    #            values all the down to bottom index.
    lowest_good = t[0] * 0.0 + fill
    lowest_good_ind = np.zeros(lowest_good.shape, dtype=np.int64)
    for l_ind in np.arange(t.shape[0]):
        t_at_l = t[l_ind]
        t_at_l_good = (t_at_l != fill)
        replace = t_at_l_good & (lowest_good == fill)
        lowest_good[replace] = t_at_l[replace]
        lowest_good_ind[replace] = l_ind

    for l_ind in np.arange(t.shape[0]):
        t_at_l = t[l_ind]
        # Limit extrapolation to bins below lowest good bin:
        t_at_l_bad = (t_at_l == fill) & (l_ind < lowest_good_ind)
        t_at_l[t_at_l_bad] = lowest_good[t_at_l_bad]

    return t


def _merra_land_mask(data: np.ndarray, mask_sd: Dataset) -> np.ndarray:
    """ Convert fractional merra land mask to 1=land 0=ocean.

    XXX TODO: need to add in FRLANDICE so antarctica and greenland get included. (Is this done?)
    :rtype: np.ndarray
    """
    frlandice = mask_sd.variables['FRLANDICE'][0]  # 0th time index
    data = frlandice + data
    return data > 0.25


def _hack_snow(data: np.ndarray, mask_sd: Dataset) -> np.ndarray:
    """ Force greenland/antarctica to be snowy like CFSR """

    # Special case: set snow depth missing values to match CFSR behavior.
    frlandice = mask_sd.variables['FRLANDICE'][0]  # 0th time index
    data[frlandice > 0.25] = 100.0
    return data


def _trim_toa(data: np.ndarray) -> np.ndarray:
    """Trim the top of the atmosphere."""
    if len(data.shape) != 3:
        LOG.warning('Warning: why did you run _trim_toa on a non-3d var?')
    # at this point (before _reshape), data should be (level, lat, lon) and
    # the level dim should be ordered surface -> toa
    return data[0:len(LEVELS), :, :]


def make_merra_one_day(in_files: Dict[str, Path], out_dir: Path, mask_fn: str):
    "parser"" Read input, parse times, and run conversion on one day at a time."""

    merra_sd = dict()
    for file_name_key in in_files.keys():
        merra_sd[file_name_key] = Dataset(in_files[file_name_key])
    merra_sd['mask'] = Dataset(mask_fn)

    try:
        # --- build a list of all times in all input files
        times = dict()
        time_set = dict()
        for file_name_key in in_files.keys():
            times[file_name_key] = []
            time_set[file_name_key] = set()
            if 'time' in merra_sd[file_name_key].variables.keys():
                t_sds = merra_sd[file_name_key].variables['time']
                t_units = t_sds.units  # expect format "minutes since %Y-%m-%d %H:%M:%S"
                base_time = datetime.strptime(t_units + ' UTC', 'minutes since %Y-%m-%d %H:%M:%S %Z')
                # A non-zero time_hack_offset is going to be equal to base_time.minute
                time_hack_offset = base_time.minute
            else:
                raise ValueError("Couldn't find time coordinate in this file")
            for (i, t) in enumerate(t_sds):
                analysis_time = num2date(t, t_units)
                if analysis_time.minute == time_hack_offset:
                    # total hack to deal with non-analysis products being on the half-hour 
                    analysis_time = analysis_time - timedelta(minutes=time_hack_offset)
                # This is just to get the index for a given timestamp later:
                times[file_name_key].append((i, analysis_time))
                time_set[file_name_key].add(analysis_time)
        # find set of time common to all input files
        common_times = (time_set['ana'] &
                        time_set['flx'] &
                        time_set['slv'] &
                        time_set['lnd'] &
                        time_set['asm3d'] &
                        time_set['asm2d'] &
                        time_set['rad'])

        # if we don't have 4 common times something is probably terribly wrong
        if len(common_times) != 4:
            raise ValueError('Input files have not produced common times')

        out_fnames = []
        for out_time in sorted(common_times):  # only sorted for debugging purposes.
            LOG.info('    working on time: {}'.format(out_time))
            out_fname = str(out_dir.joinpath(out_time.strftime('merra.%y%m%d%H_F000.hdf')))
            LOG.info(out_fname)
            out_fnames.append(out_fname)
            merra_sd['out'] = SD(out_fname, SDC.WRITE | SDC.CREATE | SDC.TRUNC)  # TRUNC will clobber existing

            # --- determine what time index we want from input files
            time_inds = dict()
            for file_name_key in in_files.keys():
                # Get the index for the current timestamp:
                time_inds[file_name_key] = [i for (i, t) in times[file_name_key] if t == out_time][0]

            # --- prepare input data variables
            output_vars = dict()
            for out_key in OUTPUT_VARS_ROSETTA:
            #for out_key in ['rh', 'temperature', 'pressure levels', 'level', 'lat', 'lon']:
                rsk = OUTPUT_VARS_ROSETTA[out_key]
                output_vars[out_key] = MerraConversion(merra_sd, rsk['in_file'], rsk['in_varname'], out_key,
                                                       rsk['out_units'], rsk['units_fn'], rsk['ndims_out'],
                                                       time_inds[rsk['in_file']])
            # to insure that all the variables are filled before attempting the data conversions,
            # read all the output_vars and then convert before setting the hdf variables out.
            #for out_key in ['rh', 'temperature', 'pressure levels', 'level', 'lat', 'lon']:
            for out_key in OUTPUT_VARS_ROSETTA:
                var_fill = output_vars[out_key].fill
                out_data = output_vars[out_key].data
                if out_key == 'total ozone':
                    out_data = total_ozone(out_data, var_fill)
                elif out_key == 'rh':
                    out_data = qv_to_rh(out_data, output_vars['temperature'].data)
                elif out_key == 'rh at sigma=0.995':
                    temp_T10M = output_vars['temperature at sigma=0.995'].data
                    # this is not an output variable, so just use MerraConversion to consistently
                    # extract the data from the source file.
                    ps_pa = MerraConversion(merra_sd, "slv", "PS",
                                            "surface pressure at sigma=0.995",
                                            "hPa", None, 2, output_vars[out_key].time_ind).data
                    out_data = rh_at_sigma(temp_T10M, ps_pa, var_fill, out_data)
                elif out_key == 'water equivalent snow depth':
                    out_data = _hack_snow(out_data, merra_sd['mask'])
                else:
                    out_data = apply_conversion(output_vars[out_key].units_fn, out_data, var_fill)

                output_vars[out_key].update_output(merra_sd, out_data)

            # --- handle surface height and static land mask from constants (mask_fn) (no time dependence)
            geopotential_height = MerraConversion(merra_sd, 'mask', 'PHIS', 'surface height', 'km',
                                                  lambda a: a / 9806.6,  # geopotential (m^2 s^-2) => height h/(1000.*g)
                                                  2, 0)
            out_data = apply_conversion(geopotential_height.units_fn, geopotential_height.data,
                                        geopotential_height.fill)
            geopotential_height.update_output(merra_sd, out_data)

            land_mask = MerraConversion(merra_sd, 'mask', 'FRLAND', 'land mask',
                                        '1=land, 0=ocean, greenland and antarctica are land',
                                        _merra_land_mask, 2, 0,)
            land_mask_calculated = _merra_land_mask(land_mask.data, merra_sd['mask'])
            land_mask.update_output(merra_sd, land_mask_calculated)
            # --- handle ice-fraction and land ice-fraction from constants (mask_fn) specially
            # use FRSEAICE (ice-fraction): GFS uses sea-ice fraction as 'ice fraction'.
            # This version of ice-fraction is broken, so it is being shielded from CLAVR-x
            # use with the output name 'FRACI' until this gets figured out.
            sea_ice_fraction = MerraConversion(merra_sd, 'mask', 'FRACI',
                                               'FRACI',  # see comment "use FRSEAICE (ice-fraction)..."
                                               'none', no_conversion, 2, 0)
            sea_ice_fraction.update_output(merra_sd, sea_ice_fraction.data)
            land_ice_fraction = MerraConversion(merra_sd, 'mask', 'FRLANDICE',
                                                'land ice fraction', 'none', no_conversion, 2, 0)
            land_ice_fraction.update_output(merra_sd, land_ice_fraction.data)

            # --- write global attributes
            var = merra_sd['out'].select('temperature')
            nlevel = var.dimensions(full=False)['level']
            nlat = var.dimensions(full=False)['lat']
            nlon = var.dimensions(full=False)['lon']
            setattr(merra_sd['out'], 'NUMBER OF LATITUDES', nlat)
            setattr(merra_sd['out'], 'NUMBER OF LONGITUDES', nlon)
            setattr(merra_sd['out'], 'NUMBER OF PRESSURE LEVELS', nlevel)
            setattr(merra_sd['out'], 'NUMBER OF O3MR LEVELS', nlevel)
            setattr(merra_sd['out'], 'NUMBER OF RH LEVELS', nlevel)
            setattr(merra_sd['out'], 'NUMBER OF CLWMR LEVELS', nlevel)
            lat = merra_sd['out'].select('lat')
            lon = merra_sd['out'].select('lon')
            attr = merra_sd['out'].attr('LATITUDE RESOLUTION')
            attr.set(SDC.FLOAT32, lat.get()[1] - lat.get()[0])
            attr = merra_sd['out'].attr('LONGITUDE RESOLUTION')
            attr.set(SDC.FLOAT32, lon.get()[1] - lon.get()[0])
            attr = merra_sd['out'].attr('FIRST LATITUDE')
            attr.set(SDC.FLOAT32, lat.get()[0])
            attr = merra_sd['out'].attr('FIRST LONGITUDE')
            attr.set(SDC.FLOAT32, lon.get()[0])
            setattr(merra_sd['out'], 'GRIB TYPE', 'not applicable')  # XXX better to just not write this attr?
            setattr(merra_sd['out'], '3D ARRAY ORDER', 'ZXY')  # XXX is this true here?
            setattr(merra_sd['out'], 'MERRA STREAM', "{}".format(merra_sd['ana'].GranuleID.split('.')[0]))
            setattr(merra_sd['out'], 'MERRA History', "{}".format(merra_sd['ana'].History))
            [a.endaccess() for a in [var, lat, lon]]

            merra_sd['out'].end()

    finally:
        for file_name_key in in_files.keys():
            # merra_source_data[k].end()
            LOG.info("Finished {}".format(file_name_key))

    return out_fnames


def download_data(inpath: Union[str, Path], file_glob: str,
                  file_type: str, get_date: datetime) -> Path:
    """Download data with wget scripts if needed."""

    inpath.mkdir(parents=True, exist_ok=True)
    LOG.info("Current In path is {} looking for {}".format(inpath,file_glob))
    try:
        file_list = list(inpath.glob(file_glob))[0]
    except IndexError as e:
        # TODO: Build and OPeNDAP request in python to retrieve data.
        # In short term: use wget
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_name = os.path.join(script_dir, "scripts", "wget_all.sh")
        cmd = 'sh {} -w {} -k {} {}'.format(script_name, inpath.parent,
                                            file_type, get_date.strftime("%Y %m %d"))
        cmd = ['sh', script_name,  '-w', inpath.parent,  '-k',  file_type, get_date.strftime("%Y %m %d")] 
        try:
            proc = subprocess.run(cmd, text=True)
        except subprocess.CalledProcessError as e:
            LOG.error(e.returncode)
            LOG.error(e.output)

        if proc.returncode != 0:
            LOG.info(proc.stdout)
            LOG.error(proc.stderr)

        try:
            file_list = list(inpath.glob(file_glob))[0]
        except IndexError as e:
            raise FileNotFoundError("{}/{}".format(inpath, file_glob))

    return file_list


def build_input_collection(desired_date: datetime, in_path: Path) -> Dict[str, Path]:
    """Use datetime to build mapping of downloaded input files to process for output."""

    date_str_arg = desired_date.strftime("%Y%m%d")
    # BTH: Define mask_file here
    mask_fn = download_data(in_path.joinpath("2d_ctm"),
                            f'MERRA2_101.const_2d_ctm_Nx.{date_str_arg}.nc4',
                            "const_2d_ctm_Nx", desired_date)
    LOG.info('Processing date: {}'.format(desired_date.strftime('%Y-%m-%d')))

    in_files = {
        'ana': download_data(in_path.joinpath("3d_ana"),
                             f'MERRA2*ana_Np.{date_str_arg}.nc4',
                             'inst6_3d_ana_Np', desired_date),
        'flx': download_data(in_path.joinpath("2d_flx"),
                             f'MERRA2*flx_Nx.{date_str_arg}.nc4',
                             'tavg1_2d_flx_Nx', desired_date),
        'slv': download_data(in_path.joinpath("2d_slv"),
                             f'MERRA2*slv_Nx.{date_str_arg}.nc4',
                             'tavg1_2d_slv_Nx', desired_date),
        'lnd': download_data(in_path.joinpath("2d_lnd"),
                             f'MERRA2*lnd_Nx.{date_str_arg}.nc4',
                             'tavg1_2d_lnd_Nx', desired_date),
        'asm3d': download_data(in_path.joinpath("3d_asm"),
                               f'MERRA2*asm_Np.{date_str_arg}.nc4',
                               'inst3_3d_asm_Np', desired_date),
        'asm2d': download_data(in_path.joinpath("2d_asm"),
                               f'MERRA2*asm_Nx.{date_str_arg}.nc4',
                               'inst1_2d_asm_Nx', desired_date),
        'rad': download_data(in_path.joinpath("2d_rad"),
                             f'MERRA2*rad_Nx.{date_str_arg}.nc4',
                             'tavg1_2d_rad_Nx', desired_date),
        'mask_file': mask_fn,
    }
    return in_files


def process_merra(base_path=None, input_path=None, start_date=None, end_date=None, store_temp=False) -> None:

    out_path_parent = base_path
    try:
        start_dt = datetime.strptime(start_date, '%Y%m%d')
    except ValueError:
        print('usage:\n    python merra4clavrx.py 20090101')
        sys.exit()

    if end_date is not None:
        end_dt = datetime.strptime(end_date, '%Y%m%d')
    else:
        end_dt = start_dt

    for dt in date_range(start_dt, end_dt, freq='D'):
        year = dt.strftime("%Y")
        year_month_day = dt.strftime("%Y_%m_%d")
        out_path_full = Path(out_path_parent).joinpath(year)

        try:
            out_path_full.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            msg = "Oops!  {} \n Enter a valid directory with -d flag".format(e)
            raise OSError(msg)

        if store_temp:
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                in_data = build_input_collection(dt, Path(tmp_dir_name))
                mask_file = str(in_data.pop('mask_file'))
                LOG.debug("Mask File {}".format(mask_file))
                out_list = make_merra_one_day(in_data, out_path_full, mask_file)
                LOG.info(', '.join(map(str, out_list)))
        else:
            input_path = Path(input_path).joinpath(year, year_month_day)
            input_path.mkdir(parents=True, exist_ok=True)
            in_data = build_input_collection(dt, input_path)
            mask_file = str(in_data.pop('mask_file'))
            LOG.debug("Mask File {}".format(mask_file))
            out_list = make_merra_one_day(in_data, out_path_full, mask_file)
            LOG.info(', '.join(map(str, out_list)))


def argument_parser() -> CommandLineMapping:
    """Parse command line for merra24clavrx.py."""
    parse_desc = (
        """\nRetrieve merra2 data from GES DISC
                    and process for clavrx input."""
    )
    formatter = ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=parse_desc,
                            formatter_class=formatter)
    group = parser.add_mutually_exclusive_group()

    parser.add_argument('start_date', action='store',
                        type=str, metavar='start_date',
                        help="Desired processing date as YYYYMMDD")
    parser.add_argument('-e', '--end_date', dest='end_date', action='store',
                        type=str, required=False, default=None,
                        metavar='end_date',
                        help="End date as YYYYMMDD not needed when processing one date.")
    # store_true evaluates to False when flag is not in use (flag invokes the store_true action)
    group.add_argument('-t', '--tmp', dest='store_temp', action='store_true',
                        help="Use to store downloaded input files in a temporary location.")
    group.add_argument('-i', '--input', dest='input_path', action='store', nargs='?',
                        type=str, required=False, default=OUT_PATH_PARENT, const=OUT_PATH_PARENT,
                        help="Data Input path (in absence of -t/--tmp flag) year/year_month_day subdirs append to path.")
    parser.add_argument('-d', '--base_dir', dest='base_path', action='store', nargs='?', 
                        type=str, required=False, default=OUT_PATH_PARENT, const=OUT_PATH_PARENT,
                        help="Parent path used final location year subdirectory appends to this path.")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=2,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG')

    args = vars(parser.parse_args())
    verbosity = args.pop('verbosity', None)

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(format='%(module)s:%(lineno)d:%(levelname)s:%(message)s', level=levels[min(3, verbosity)])

    return args


if __name__ == '__main__':

    parser_args = argument_parser()
    process_merra(**parser_args)

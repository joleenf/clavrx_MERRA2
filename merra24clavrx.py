# -*- coding: utf-8 -*-
"""Module to convert MERRA2 re-analysis data to hdf files for use in CLAVR-x.

usage: merra24clavrx.py [-h] [-e end_date] [-t | -i [INPUT_PATH]] [-d [BASE_PATH]] [-v] start_date

Retrieve merra2 data from GES DISC and process for clavrx input.

positional arguments:
  start_date            Desired processing date as YYYYMMDD

optional arguments:
  -h, --help            show this help message and exit
  -e end_date, --end_date end_date
                        End date as YYYYMMDD not needed when processing one date.
                        (default: None)
  -t, --tmp             Use to store downloaded input files in a temporary location.
                        (default: False)
  -i [INPUT_PATH], --input [INPUT_PATH]
                        Data Input path (in absence of -t/--tmp flag) year/year_month_day
                        subdirs append to path. (default: /apollo/cloud/Ancil_Data/
                        clavrx_ancil_data/dynamic/merra2/)
  -d [BASE_PATH], --base_dir [BASE_PATH]
                        Parent path used final location year subdirectory appends to this path.
                        (default: /apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2/)
  -v, --verbose         each occurrence increases verbosity 1 level through
                        CRITICAL-ERROR-WARNING-INFO-DEBUG (default: 0)

"""

import logging
import os
import subprocess
import sys
import tempfile
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import yaml

try:
    from datetime import datetime, timedelta
    from typing import Dict, Optional, TypedDict, Union

    import numpy as np
    from netCDF4 import Dataset, num2date
    from pandas import date_range
    from pyhdf.SD import SD, SDC
except ImportError as e:
    print("Import Error {}".format(e))
    print("Type:  conda activate merra2_clavrx")
    sys.exit(1)

np.seterr(all="ignore")

LOG = logging.getLogger(__name__)

COMPRESSION_LEVEL = 6  # 6 is the gzip default; 9 is best/slowest/smallest fill
TOP_LEVEL = 10  # [hPa] This is the highest CFSR level, trim anything higher.
CLAVRX_FILL = 9.999e20

OUT_PATH_PARENT = "/apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2/"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT_DIR, 'yamls', 'MERRA2_vars.yaml'), "r") as yml:
    OUTPUT_VARS_ROSETTA = yaml.load(yml, Loader=yaml.Loader)

levels_listings = OUTPUT_VARS_ROSETTA.pop("defined levels")
LEVELS = levels_listings["hPa_levels"]


class CommandLineMapping(TypedDict):
    """Type hints for result of the argparse parsing."""

    start_date: str
    end_date: Optional[str]
    store_temp: bool
    base_path: str
    input_path: str


class MerraConversion:
    """MerraConversion Handles Reading Data and Output Setup."""

    def __init__(
            self,
            nc_dataset,
            in_name,
            out_name,
            out_units,
            ndims_out,
            time_ind,
    ):
        """Based on variable, adjust shape, apply fill and determine dtype."""
        self.nc_dataset = nc_dataset
        self.in_name = in_name
        self.out_name = out_name
        self.out_units = out_units
        self.ndims_out = ndims_out

        self.fill = self._get_fill
        self.data = self._get_data(time_ind)

    def __repr__(self):
        """Report the name conversion when creating this object."""
        return "Input name {} ==> Output Name: {}".format(self.in_name, self.out_name)

    def __getitem__(self, item):
        """Access data in the NetCDF dataset variable by variable key."""
        return self.nc_dataset.variables[item]

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

    def _get_data(self, time_ind):
        """Get data and based on dimensions reorder axes, truncate TOA, apply fill."""
        data = np.ma.getdata(self[self.in_name])

        data = np.asarray(data)

        if self.in_name == "lev" and len(data) != 42:
            # insurance policy while levels are hard-coded in unit conversion fn's
            # also is expected based on data documentation:
            # https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf
            raise ValueError(
                "Incorrect number of levels {} rather than 42.".format(len(data))
            )

        ndims_in = len(data.shape)
        if ndims_in == 3:
            # note, vars w/ 3 spatial dims will be 4d due to time
            data = data[time_ind]
        if ndims_in == 4:
            data = data[time_ind]
            data = self._trim_toa(data)

        # apply special cases
        if self.in_name == "lev":
            # trim to top CFSR level
            data = data[0: len(LEVELS)].astype(np.float32)
            data = np.flipud(data)  # clavr-x needs toa->surface
        elif self.in_name == "lon":
            tmp = np.copy(data)
            halfway = data.shape[0] // 2
            data = np.r_[tmp[halfway:], tmp[:halfway]]
        else:
            pass

        if self.fill is not None:
            if self.out_name == "water equivalent snow depth":
                #  Special case: set snow depth missing values to 0 matching CFSR behavoir.
                data = np.where(data == self.fill, 0.0, data)
            else:
                data = np.where(data == self.fill, np.nan, data)

        return data

    @staticmethod
    def _trim_toa(data: np.ndarray) -> np.ndarray:
        """Trim the top of the atmosphere."""
        if len(data.shape) != 3:
            LOG.warning("Warning: why did you run _trim_toa on a non-3d var?")
        # at this point (before _reshape), data should be (level, lat, lon) and
        # the level dim should be ordered surface -> toa
        return data[0:len(LEVELS), :, :]

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

        if self.out_name in ["pressure levels", "level"]:
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

    def update_output(self, sd, in_file_short_value, data_array):
        """Finalize output variables."""
        out_fill = self.fill

        out_sds = sd["out"].create(self.out_name, self._create_output_dtype, self._modify_shape())
        out_sds.setcompress(SDC.COMP_DEFLATE, value=COMPRESSION_LEVEL)
        self.set_dim_names(out_sds)
        if self.out_name == "lon":
            out_sds.set(_reshape(data_array, self.ndims_out, None))
        else:
            if self.out_name == "rh":
                new = _refill(_reshape(data_array, self.ndims_out, out_fill), out_fill)
                out_sds.set(new)
            else:
                out_sds.set(_refill(_reshape(data_array, self.ndims_out, out_fill), out_fill))

        out_sds.setfillvalue(CLAVRX_FILL)
        if self.out_units is not None:
            out_sds.units = self.out_units

        if "units" in self.nc_dataset.variables[self.in_name].ncattrs():
            unit_desc = " in [{}]".format(self[self.in_name].units)
        else:
            unit_desc = ""
        out_sds.source_data = ("MERRA->{}->{}{}".format(in_file_short_value,
                                                        self.in_name,
                                                        unit_desc))
        out_sds.long_name = self[self.in_name].long_name
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


def total_saturation_pressure(temp_in_k, mix_lo=253.16, mix_hi=273.16):
    """Calculate the total saturation pressure.

    :param temp_in_k: Temperature in kelvin at all pressure levels
    :param mix_lo:
    :param mix_hi:
    :return: Total saturation pressure
    """
    saturation_vapor_pressure_wmo = (vapor_pressure_liquid_water_wmo(temp_in_k))
    es_total = saturation_vapor_pressure_wmo.copy()

    vapor_pressure_ice = vapor_pressure_over_ice(temp_in_k)
    ice_ind = temp_in_k <= mix_lo
    es_total[ice_ind] = vapor_pressure_ice[ice_ind]

    mix_ind = (temp_in_k > mix_lo) & (temp_in_k < mix_hi)
    liq_weight = (temp_in_k - mix_lo) / (mix_hi - mix_lo)
    ice_weight = (mix_hi - temp_in_k) / (mix_hi - mix_lo)

    e_mix = ice_weight * vapor_pressure_ice + liq_weight * saturation_vapor_pressure_wmo
    es_total[mix_ind] = e_mix[mix_ind]

    return es_total


def vapor_pressure_liquid_water_wmo(temp_in_kelvin):
    """Calculate the Vapor pressure over liquid water below 0°C by WMO Formula."""
    # Saturation vapor pressure:
    #  http://faculty.eas.ualberta.ca/jdwilson/EAS372_13/Vomel_CIRES_satvpformulae.html
    # >273.16: w.r.t liquid water
    # 253.16 < T < 273.16: weighted interpolation of water / ice
    # < 253.16: w.r.t ice

    es_wmo = 10.0 ** (10.79574 * (1. - 273.16 / temp_in_kelvin)
                      - 5.02800 * np.log10(temp_in_kelvin / 273.16)
                      + 1.50475 * 10. ** -4. *
                      (1. - 10. ** (-8.2969 * (temp_in_kelvin / 273.16 - 1)))
                      + 0.42873 * 10. ** -3. *
                      (10. ** (4.76955 * (1 - 273.16 / temp_in_kelvin)) - 1.)
                      + 0.78614) * 100.0  # [Pa])

    return es_wmo


def vapor_pressure_over_ice(temp_in_kelvin):
    """Calculate the vapor pressure over ice using the Goff Gratch equation."""
    goff_gratch_vapor_pressure_ice = (10.0 ** (
            -9.09718 * (273.16 / temp_in_kelvin - 1.0)
            - 3.56654 * np.log10(273.16 / temp_in_kelvin)
            + 0.876793 * (1.0 - temp_in_kelvin / 273.16)
            + np.log10(6.1071)
    ) * 100.0)  # [Pa]

    return goff_gratch_vapor_pressure_ice


def vapor_pressure_approximation(qv, sfc_pressure, plevels):
    """Approximate the vapor pressure using specific humidity and surface pressure.

    :param qv: specific humidity of the water vapor
    :param sfc_pressure: surface pressure
    :param plevels: hPa levels of pressure
    :return: a good approximation of vapor pressure (still should multiply by pressure @ each level)
    """
    vapor_pressure = 1.0 / (
            0.622 / qv + (1.0 - 0.622)
    )  # still need to multiply by pressure @ each level
    if sfc_pressure is None:
        # 3D RH field
        for i, lev in enumerate(plevels):
            # already cut out time dim
            vapor_pressure[i, :, :] = vapor_pressure[i, :, :] * lev
    else:
        # RH @ 10m: multiply by surface pressure
        vapor_pressure = vapor_pressure * sfc_pressure

    return vapor_pressure


def qv_to_rh(specific_humidity, temp_k, press_at_sfc=None):
    """Convert Specific Humidity [kg/kg] -> relative humidity [%]."""
    # See Petty Atmos. Thermo. 4.41 (p. 65), 8.1 (p. 140), 8.18 (p. 147)
    levels = map(lambda a: a * 100.0, LEVELS)  # [hPa] -> [Pa]

    es_tot = total_saturation_pressure(temp_k)

    vapor_pressure = vapor_pressure_approximation(specific_humidity, press_at_sfc, levels)

    relative_humidity = vapor_pressure / es_tot * 100.0  # relative hu¬idity [%]
    relative_humidity[relative_humidity > 100.0] = 100.0  # clamp to 100% to mimic CFSR
    return relative_humidity


def dobson_layer(mmr_data_array, level_index):
    """Calculate a dobson layer from a 3D mmr data array given a level index.

    :param mmr_data_array: Mass mixing ratio data array
    :param level_index: index of current level
    :return: dobson layer
    """
    dobson_unit_conversion = 2.69e16  # 1 DU = 2.69e16 molecules cm-2
    gravity = 9.8  # m/s^2
    avogadro_const = 6.02e23  # molecules/mol
    o3_molecular_weight = 0.048  # kg/mol
    dry_air_molecular_weight = 0.028966  # kg/mol molecular weight of dry air.

    const = 0.01 * avogadro_const / (gravity * dry_air_molecular_weight)
    vmr_j = mmr_data_array[level_index] * dry_air_molecular_weight / o3_molecular_weight
    vmr_j_1 = mmr_data_array[level_index - 1] * dry_air_molecular_weight / o3_molecular_weight
    ppmv = 0.5 * (vmr_j + vmr_j_1)
    delta_pressure = LEVELS[level_index - 1] - LEVELS[level_index]
    o3_dobs_layer = ppmv * delta_pressure * const / dobson_unit_conversion

    return o3_dobs_layer


def total_ozone(mmr, fill_value):
    """Calculate total column ozone in dobson units (DU), from ozone mixing ratio in [kg/kg]."""
    nlevels = mmr.shape[0]
    dobson = np.zeros(mmr[0].shape).astype("float32")
    for j in range(1, nlevels):
        good_j = np.logical_not(np.isnan(mmr[j]))
        good_j_1 = np.logical_not(np.isnan(mmr[j - 1]))
        good = good_j & good_j_1
        dobs_layer = dobson_layer(mmr, j)
        dobson[good] = dobson[good] + dobs_layer[good]

    dobson[np.isnan(dobson)] = fill_value

    return dobson


def rh_at_sigma(temp10m, sfc_pressure, sfc_pressure_fill, data):
    """Calculate the rh at sigma using 10m fields."""
    temp_k = temp10m  # temperature in [K] (Y, X) not in (time, Y, X)

    # pressure in [Pa]
    sfc_pressure[sfc_pressure == sfc_pressure_fill] = np.nan

    rh_sigma = qv_to_rh(data, temp_k, press_at_sfc=sfc_pressure)
    rh_sigma[np.isnan(rh_sigma)] = sfc_pressure_fill

    return rh_sigma


def _reshape(data, ndims_out, fill):
    """Make MERRA look like CFSR.

    * Convert array dimensions of (level, lat, lon) to (lat, lon, level)
    * CFSR fields are continuous but MERRA below-ground values are set to fill.
    * CFSR starts at 0 deg lon but merra starts at -180.
    """
    if ndims_out in (2, 3):
        data = _shift_lon(data)

    if ndims_out == 3:
        # do extrapolation before reshape (depends on a certain dimensionality/ordering)
        data = _extrapolate_below_sfc(data, fill)
        data = np.swapaxes(data, 0, 2)
        data = np.swapaxes(data, 0, 1)
        data = data[:, :, ::-1]  # clavr-x needs toa->surface not surface->toa
    return data


def _refill(data, old_fill):
    """Assumes CLAVRx fill value instead of variable attribute."""
    if (data.dtype == np.float32) or (data.dtype == np.float64):
        if np.isnan(data).any():
            data[np.isnan(data)] = CLAVRX_FILL
        data[data == old_fill] = CLAVRX_FILL
    return data


def _shift_lon_2d(data):
    """Assume dims are 2d and (lat, lon)."""
    nlon = data.shape[1]
    halfway = nlon // 2
    tmp = data.copy()
    data[:, 0:halfway] = tmp[:, halfway:]
    data[:, halfway:] = tmp[:, 0:halfway]
    return data


def _shift_lon(data):
    """Make lon start at 0deg instead of -180.

    Assume dims are (level, lat, lon) or (lat, lon)
    """
    if len(data.shape) == 3:
        for l_ind in np.arange(data.shape[0]):
            data[l_ind] = _shift_lon_2d(data[l_ind])
    elif len(data.shape) == 2:
        data = _shift_lon_2d(data)
    return data


def _extrapolate_below_sfc(t, fill):
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


def _merra_land_mask(data: np.ndarray, mask_sd: Dataset) -> np.ndarray:
    """Convert fractional merra land mask to 1=land 0=ocean.

    use FRLANDICE so antarctica and greenland get included.
    :rtype: np.ndarray
    """
    frlandice = mask_sd.variables["FRLANDICE"][0]  # 0th time index
    data = frlandice + data
    return data > 0.25


def _hack_snow(data: np.ndarray, mask_sd: Dataset) -> np.ndarray:
    """Force greenland/antarctica to be snowy like CFSR."""
    # Special case: set snow depth missing values to match CFSR behavior.
    frlandice = mask_sd.variables["FRLANDICE"][0]  # 0th time index
    data[frlandice > 0.25] = 100.0
    return data


def apply_conversion(scale_func, data, fill):
    """Apply fill to converted data after function."""
    converted = data.copy()

    if scale_func == "total_ozone":
        converted = total_ozone(data, fill)
    else:
        converted = scale_func(converted)

        converted[data == fill] = fill
        if np.isnan(data).any():
            converted[np.isnan(data)] = fill

    return converted


def get_common_time(datasets: Dict[str, Dataset]):
    """Find common start time among the input datasets."""
    # --- build a list of all times in all input files
    dataset_times = dict()
    time_set = dict()
    for ds_key in datasets.keys():
        dataset_times[ds_key] = []
        time_set[ds_key] = set()
        if "time" in datasets[ds_key].variables.keys():
            t_sds = datasets[ds_key].variables["time"]
            t_units = t_sds.units  # fmt "minutes since %Y-%m-%d %H:%M:%S"
            base_time = datetime.strptime(
                t_units + " UTC", "minutes since %Y-%m-%d %H:%M:%S %Z"
            )
            # A non-zero time_hack_offset = base_time.minute
            time_hack_offset = base_time.minute
        else:
            raise ValueError("Couldn't find time coordinate in this file")
        for (i, t) in enumerate(t_sds):
            # format %y doesn't work with gregorian time.
            analysis_time = (num2date(t, t_units, only_use_python_datetimes=True,
                                      only_use_cftime_datetimes=False))
            if analysis_time.minute == time_hack_offset:
                # total hack to deal with non-analysis products being on the half-hour
                analysis_time = analysis_time - \
                                timedelta(minutes=time_hack_offset)
            # This is just to get the index for a given timestamp later:
            dataset_times[ds_key].append((i, analysis_time))
            time_set[ds_key].add(analysis_time)
    # find set of time common to all input files
    ds_common_times = (
            time_set["ana"]
            & time_set["flx"]
            & time_set["slv"]
            & time_set["lnd"]
            & time_set["asm3d"]
            & time_set["asm2d"]
            & time_set["rad"]
    )

    # if we don't have 4 common times something is probably terribly wrong
    if len(ds_common_times) != 4:
        raise ValueError("Input files have not produced common times")

    return dataset_times, ds_common_times


def get_time_index(input_files, file_times, current_time):
    """Determine the time index from the input files based on the output time."""
    # --- determine time index we want from input files
    time_index = dict()
    for file_name_key in input_files:
        # Get the index for the current timestamp:
        if file_name_key == "mask":
            time_index[file_name_key] = 0
        else:
            time_index[file_name_key] = [i for (i, t) in file_times[file_name_key]
                                         if t == current_time][0]
    return time_index


def write_global_attributes(data_sd):
    """Write global attributes."""
    out_sd = data_sd["out"]
    var = out_sd.select("temperature")
    nlevel = var.dimensions(full=False)["level"]
    nlat = var.dimensions(full=False)["lat"]
    nlon = var.dimensions(full=False)["lon"]
    setattr(out_sd, "NUMBER OF LATITUDES", nlat)
    setattr(out_sd, "NUMBER OF LONGITUDES", nlon)
    setattr(out_sd, "NUMBER OF PRESSURE LEVELS", nlevel)
    setattr(out_sd, "NUMBER OF O3MR LEVELS", nlevel)
    setattr(out_sd, "NUMBER OF RH LEVELS", nlevel)
    setattr(out_sd, "NUMBER OF CLWMR LEVELS", nlevel)
    lat = out_sd.select("lat")
    lon = out_sd.select("lon")
    attr = out_sd.attr("LATITUDE RESOLUTION")
    attr.set(SDC.FLOAT32, lat.get()[1] - lat.get()[0])
    attr = out_sd.attr("LONGITUDE RESOLUTION")
    attr.set(SDC.FLOAT32, lon.get()[1] - lon.get()[0])
    attr = out_sd.attr("FIRST LATITUDE")
    attr.set(SDC.FLOAT32, lat.get()[0])
    attr = out_sd.attr("FIRST LONGITUDE")
    attr.set(SDC.FLOAT32, lon.get()[0])
    setattr(
        out_sd, "GRIB TYPE", "not applicable"
    )
    setattr(out_sd,
            "3D ARRAY ORDER", "YXZ")
    setattr(out_sd,
            "MERRA STREAM",
            "{}".format(data_sd["ana"].GranuleID.split(".")[0]))
    setattr(out_sd, "MERRA History",
            "{}".format(data_sd["ana"].History))
    for a in [var, lat, lon]:
        a.endaccess()
    data_sd["out"].end()


def write_output_variables(datasets, output_vars):
    """Calculate the final output and write to file."""
    for out_key, rsk in OUTPUT_VARS_ROSETTA.items():
        if out_key == "surface_pressure_at_sigma":
            continue
        units_fn = rsk["units_fn"]
        var_fill = output_vars[out_key].fill
        out_data = output_vars[out_key].data
        if out_key == "rh":
            out_data = qv_to_rh(out_data,
                                output_vars["temperature"].data)
            out_data[np.isnan(out_data)] = output_vars[
                "temperature"
            ].fill  # keep to match original code
        elif out_key == "rh at sigma=0.995":
            temp_t10m = output_vars["temperature at sigma=0.995"].data
            ps_pa = output_vars["surface_pressure_at_sigma"].data
            out_data = rh_at_sigma(temp_t10m, ps_pa,
                                   var_fill, out_data)
        elif out_key == "water equivalent snow depth":
            out_data = _hack_snow(out_data, datasets["mask"])
        elif out_key == "land mask":
            out_data = _merra_land_mask(out_data, datasets["mask"])
        else:
            out_data = apply_conversion(units_fn, out_data, var_fill)

        output_vars[out_key].update_output(datasets, rsk["in_file"], out_data)


def make_merra_one_day(run_dt: datetime, input_path: Path, out_dir: Path):
    """Read input, parse times, and run conversion on one day at a time."""
    in_files = build_input_collection(run_dt, input_path)

    merra_sd = dict()
    for file_name_key, file_name in in_files.items():
        merra_sd[file_name_key] = Dataset(file_name)

    times, common_times = get_common_time(merra_sd)

    out_fnames = []
    for out_time in sorted(common_times):
        LOG.info("    working on time: %s", out_time)
        out_fname = str(out_dir.joinpath(out_time.strftime("merra.%y%m%d%H_F000.hdf")))
        out_fnames.append(out_fname)
        merra_sd["out"] = SD(
            out_fname, SDC.WRITE | SDC.CREATE | SDC.TRUNC
        )  # TRUNC will clobber existing

        time_inds = get_time_index(in_files, times, out_time)

        # --- prepare input data variables
        out_vars = dict()
        for out_key, rsk in OUTPUT_VARS_ROSETTA.items():
            LOG.info("Get data from %s for %s", rsk["in_file"], rsk["in_varname"])
            out_vars[out_key] = MerraConversion(
                merra_sd[rsk["in_file"]],
                rsk["in_varname"],
                out_key,
                rsk["out_units"],
                rsk["ndims_out"],
                time_inds[rsk["in_file"]],
            )

        write_output_variables(merra_sd, out_vars)
        write_global_attributes(merra_sd)

    return out_fnames


def download_data(inpath: Union[str, Path], file_glob: str,
                  file_type: str, get_date: datetime) -> Path:
    """Download data with wget scripts if needed."""
    inpath.mkdir(parents=True, exist_ok=True)
    LOG.info("Current In path is %s looking for %s", inpath, file_glob)
    file_list = list(inpath.glob(file_glob))
    if len(file_list) == 0:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_name = os.path.join(script_dir, "scripts", "wget_all.sh")
        cmd = [
            "sh",
            script_name,
            "-w",
            str(inpath.parent),
            "-k",
            file_type,
            get_date.strftime("%Y %m %d"),
        ]
        try:
            proc = subprocess.run(cmd, text=True, check=True)
            LOG.info(" ".join(proc.args))
            if proc.returncode != 0:
                LOG.info(proc.stdout)
                LOG.error(proc.stderr)
        except subprocess.CalledProcessError as proc_error_noted:
            raise subprocess.CalledProcessError from proc_error_noted

        file_list = list(inpath.glob(file_glob))

        if len(file_list) == 0:
            raise FileNotFoundError("{} not found at {}.".format(file_glob, inpath))

    return file_list[0]


def build_input_collection(desired_date: datetime,
                           in_path: Path) -> Dict[str, Path]:
    """Use datetime to build input file collection."""
    date_str_arg = desired_date.strftime("%Y%m%d")
    # BTH: Define mask_file here
    mask_fn = download_data(
        in_path.joinpath("2d_ctm"),
        f"MERRA2_101.const_2d_ctm_Nx.{date_str_arg}.nc4",
        "const_2d_ctm_Nx",
        desired_date,
    )
    LOG.info("Processing date: %s", desired_date.strftime("%Y-%m-%d"))

    in_files = {
        "ana": download_data(
            in_path.joinpath("3d_ana"),
            f"MERRA2*ana_Np.{date_str_arg}.nc4",
            "inst6_3d_ana_Np",
            desired_date,
        ),
        "flx": download_data(
            in_path.joinpath("2d_flx"),
            f"MERRA2*flx_Nx.{date_str_arg}.nc4",
            "tavg1_2d_flx_Nx",
            desired_date,
        ),
        "slv": download_data(
            in_path.joinpath("2d_slv"),
            f"MERRA2*slv_Nx.{date_str_arg}.nc4",
            "tavg1_2d_slv_Nx",
            desired_date,
        ),
        "lnd": download_data(
            in_path.joinpath("2d_lnd"),
            f"MERRA2*lnd_Nx.{date_str_arg}.nc4",
            "tavg1_2d_lnd_Nx",
            desired_date,
        ),
        "asm3d": download_data(
            in_path.joinpath("3d_asm"),
            f"MERRA2*asm_Np.{date_str_arg}.nc4",
            "inst3_3d_asm_Np",
            desired_date,
        ),
        "asm2d": download_data(
            in_path.joinpath("2d_asm"),
            f"MERRA2*asm_Nx.{date_str_arg}.nc4",
            "inst1_2d_asm_Nx",
            desired_date,
        ),
        "rad": download_data(
            in_path.joinpath("2d_rad"),
            f"MERRA2*rad_Nx.{date_str_arg}.nc4",
            "tavg1_2d_rad_Nx",
            desired_date,
        ),
        "mask": mask_fn,
    }
    return in_files


def process_merra(base_path=None, input_path=None, start_date=None,
                  end_date=None, store_temp=False) -> None:
    """Build base path, determine data collection and call merra processing."""
    out_path_parent = base_path
    try:
        start_dt = datetime.strptime(start_date, "%Y%m%d")
    except ValueError:
        print("usage:\n    python merra4clavrx.py 20090101")
        sys.exit()

    if end_date is not None:
        end_dt = datetime.strptime(end_date, "%Y%m%d")
    else:
        end_dt = start_dt

    for data_dt in date_range(start_dt, end_dt, freq="D"):
        year = data_dt.strftime("%Y")
        year_month_day = data_dt.strftime("%Y_%m_%d")
        out_path_full = Path(out_path_parent).joinpath(year, year_month_day)

        try:
            out_path_full.mkdir(parents=True, exist_ok=True)
        except OSError as os_err_noted:
            raise OSError("Enter valid directory with -d flag") from os_err_noted

        if store_temp:
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                out_list = make_merra_one_day(data_dt, Path(tmp_dir_name),
                                              out_path_full)
                LOG.info(", ".join(map(str, out_list)))
        else:
            input_path = Path(input_path).joinpath(year)
            input_path.mkdir(parents=True, exist_ok=True)
            out_list = make_merra_one_day(data_dt, input_path, out_path_full)
            LOG.info(", ".join(map(str, out_list)))


def argument_parser() -> CommandLineMapping:
    """Parse command line for merra24clavrx.py."""
    parse_desc = """\nRetrieve merra2 data from GES DISC
                    and process for clavrx input."""
    formatter = ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=parse_desc, formatter_class=formatter)
    group = parser.add_mutually_exclusive_group()

    parser.add_argument(
        "start_date",
        action="store",
        type=str,
        metavar="start_date",
        help="Desired processing date as YYYYMMDD",
    )
    parser.add_argument(
        "-e",
        "--end_date",
        dest="end_date",
        action="store",
        type=str,
        required=False,
        default=None,
        metavar="end_date",
        help="End date as YYYYMMDD not needed when processing one date.",
    )
    # store_true evaluates to False when flag is not used
    # (flag invokes the store_true action)
    group.add_argument(
        "-t",
        "--tmp",
        dest="store_temp",
        action="store_true",
        help="Use to store downloaded input files in a temporary location.",
    )
    group.add_argument(
        "-i",
        "--input",
        dest="input_path",
        action="store",
        nargs="?",
        type=str,
        required=False,
        default=OUT_PATH_PARENT,
        const=OUT_PATH_PARENT,
        help="Data Input path (in absence of -t/--tmp flag) "
             "year/year_month_day subdirs append to path.",
    )
    parser.add_argument(
        "-d",
        "--base_dir",
        dest="base_path",
        action="store",
        nargs="?",
        type=str,
        required=False,
        default=OUT_PATH_PARENT,
        const=OUT_PATH_PARENT,
        help="Parent path used final location year subdirectory "
             "appends to this path.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbosity",
        action="count",
        default=0,
        help="each occurrence increases verbosity 1 level through "
             "CRITICAL-ERROR-WARNING-INFO-DEBUG",
    )

    args = vars(parser.parse_args())
    verbosity = args.pop("verbosity", None)

    levels = [
        logging.CRITICAL,
        logging.ERROR,
        logging.WARN,
        logging.INFO,
        logging.DEBUG,
    ]
    logging.basicConfig(
        format="%(module)s:%(lineno)d:%(levelname)s:%(message)s",
        level=levels[min(4, verbosity)],
    )

    return args


if __name__ == "__main__":
    parser_args = argument_parser()
    process_merra(**parser_args)

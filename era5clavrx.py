# -*- coding: utf-8 -*-
"""Module to convert ERA5 re-analysis data to hdf files for use in CLAVR-x.

MERRA-2 Data is downloaded from the GES DISC server and converted to
output files which can be used as input into the CLAVR-x cloud product.

Example:
    Run the era5clavrx.py code with a single <year><month><day>::

        $ python era5clavrx.py 20010801

Optional arguments::
  -h, --help            show this help message and exit
  -e end_date, --end_date end_date
                        End date as YYYYMMDD not needed when processing one date. (default: None)
  -t, --tmp             Use to store downloaded input files in a temporary location. (default: False)
  -d [BASE_PATH], --base_dir [BASE_PATH]
                        Parent path used for input (in absence of -t/--tmp flag) and final location.
                        year subdirectory appends to this path.
                        (default: /apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/era5/)
  -v, --verbose         each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG (default: 0)

"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import sys
import os
import logging
import tempfile
import yaml
from merra24clavrx import MerraConversion, CommandLineMapping, total_ozone

try:
    from pyhdf.SD import SD, SDC
    from netCDF4 import Dataset, num2date
    from datetime import datetime, timedelta
    from typing import Union, Optional, Dict, TypedDict
    import cdsapi
    import numpy as np
    import pandas as pd
except ImportError as e:
    print("Import Error {}".format(e))
    print("Type:  conda activate merra2_clavrx")
    sys.exit(1)

np.seterr(all='ignore')

LOG = logging.getLogger(__name__)

comp_level = 6  # 6 is the gzip default; 9 is best/slowest/smallest file

FOCUS_VAR = ["rh"]
Q_ARR = [0, 0.25, 0.5, 0.75, 1.0]

# no_conversion = lambda a: a  # ugh why doesn't python have a no-op function...
# fill_bad = lambda a: a * np.nan

OUT_PATH_PARENT = '/apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/era5/'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT_DIR, 'yamls', 'ERA5_vars.yaml'), "r") as yml:
    OUTPUT_VARS_DICT = yaml.safe_load(yml)

# this is trimmed to the top CFSR level (i.e., exclude higher than 10hPa)
levels_listings = OUTPUT_VARS_DICT.pop("defined levels")
LEVELS = levels_listings["pressure levels"]

TOP_LEVEL = 10  # [hPa] This is the highest CFSR level, trim anything higher.
CLAVRX_FILL = 9.999e20


def pressure_to_altitude(pressure):
    """Use surface pressure to converted to km rather than surface geopotential for altitude."""

    P_zero = 101325  # Pa (Pressure at altitude 0)
    T_zero = 288.15  # K (Temperature at altitude 0)
    g = 9.80665      # m/s^2 (gravitational acceleration)
    L = -6.50E-03    # K/m (Lapse Rate)
    R = 287.053      # J/(KgK) (Gas constant for air)

    altitude = (T_zero/L)*((pressure / P_zero)*np.exp(-L*R/g) - 1)

    return altitude


def apply_scale(a, scale_func=None):
    if (scale_func is None) or (scale_func == "no_conversion"):
        return a
    elif scale_func == "fill_bad":
        return a * np.nan
    elif (isinstance(scale_func, float)) or (isinstance(scale_func, int)):
        return a*scale_func
    else:
        raise ValueError("Scale function is not recognized {}".format(scale_func))


def total_precipitable_water(tcwv):
    """Calculate the Total Precipitable Water from the Total Column Water Vapor"""
    g = 9.80665   # Acceleration due to gravity
    rho = 997     # water density in kg/m3
    tpw = tcwv/(g*rho)
    return tpw


def rh_at_sigma(t, td):
    """ Temperature and Dewpoint Temperature -> relative humidity [%].

    Uses August-Roche-Magnus approximation

    Input:
    t: For ERA-5 this is the 2 meter temperature
    td: For ERA-5 this is the 2 meter dewpoint temperature

    Output:
    rh: Relative Humidity
    """
    rh = 100 * (np.exp((17.625 * td) / (243.04 + td)) / np.exp((17.625 * t) / (243.04 + t)))
    return rh


def apply_conversion(units_fn, data, fill):
    """Special handling of converted data apply fill to converted data after function."""
    converted = data.copy()
    converted = apply_scale(converted, scale_func=units_fn)

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


def _hack_snow(data: np.ndarray, mask_sd: Dataset) -> np.ndarray:
    """ Force greenland/antarctica to be snowy like CFSR """

    # Special case: set snow depth missing values to match CFSR behavior.
    frlandice = mask_sd.variables['lsm'][0]  # 0th time index (Land Sea Mask)
    data[frlandice > 0.25] = 100.0
    return data


def _trim_toa(data: np.ndarray) -> np.ndarray:
    """Trim the top of the atmosphere."""
    if len(data.shape) != 3:
        LOG.warning('Warning: why did you run _trim_toa on a non-3d var?')
    # at this point (before _reshape), data should be (level, lat, lon) and
    # the level dim should be ordered surface -> toa
    return data[0:len(LEVELS), :, :]


def make_era5_one_hour(in_files: Dict[str, Path], out_dir: Path):
    """Read input, parse times, and run conversion on one day at a time."""

    era5_ds = dict()

    for file_name_key in in_files.keys():
        era5_ds[file_name_key] = Dataset(in_files[file_name_key])

    try:
        # --- build a list of all times in all input files
        time_set = dict()
        for file_name_key in in_files.keys():
            time_set[file_name_key] = set()
            t_sds = era5_ds[file_name_key].variables['time']
            # this is different for ERA5 versus MERRA2 since there are no units in ERA5 time metadata.
            time_set[file_name_key].add(num2date(t_sds[0], t_sds.units))
        # find set of time common to all input files
        common_times = (time_set['pressureLevels'] &
                        time_set['singleLevel'])

        # these should be single file times, all this
        if len(common_times) != 1:
            raise ValueError('Input files have not produced common times')
        else:
            out_time = common_times.pop()
            LOG.info('    working on time: {}'.format(out_time))
            out_fname = str(out_dir.joinpath(out_time.strftime('era5.%y%m%d%H_F000.hdf')))
            LOG.info(out_fname)
            era5_ds['out'] = SD(out_fname, SDC.WRITE | SDC.CREATE | SDC.TRUNC)  # TRUNC will clobber existing

            # --- prepare input data variables
            output_vars = dict()
            for out_key in OUTPUT_VARS_DICT:
                rsk = OUTPUT_VARS_DICT[out_key]
                if rsk['in_file'] != 'calculate':
                    output_vars[out_key] = MerraConversion(era5_ds, rsk['in_file'], rsk['in_varname'], out_key,
                                                           rsk['out_units'], rsk['units_fn'], rsk['ndims_out'],
                                                           0)
            # to insure that all the variables are filled before attempting the data conversions,
            # read all the output_vars and then convert before setting the hdf variables out.
            for out_key in OUTPUT_VARS_DICT:
                var_fill = output_vars[out_key].fill
                out_data = output_vars[out_key].data

                if out_key == 'total ozone':
                    out_data = total_ozone(out_data, var_fill)
                elif out_key == 'water equivalent snow depth':
                    out_data = _hack_snow(out_data, era5_ds['singleLevel'])
                elif out_key == "height":
                    out_data = pressure_to_altitude(out_data)
                else:
                    out_data = apply_conversion(output_vars[out_key].units_fn, out_data, var_fill)

                output_vars[out_key].update_output(era5_ds, out_data)

            # --- create rh_at_sigma=0.995 from temperature and dewpoint at 2 meters.
            # --- (Even though it says 0.995, that is to comply with old naming of variable)
            temp_at_2m = output_vars['temperature at sigma=0.995'].data
            dp_temp_at_2m = output_vars['dewpoint temperature at sigma=0.995'].data
            rh_sigma = MerraConversion(era5_ds, 'singleLevel', 't2m', 'rh at sigma = 0.995', '%',
                                       None, 2, 0)
            rh_sigma.update_output(era5_ds, rh_at_sigma(temp_at_2m, dp_temp_at_2m))
            # --- handle surface height, static land mask from constants (mask_fn) (no time dependence)
            geopotential_height = MerraConversion(era5_ds, 'mask', 'PHIS', 'surface height', 'km',
                                                  lambda a: a / 9806.6,  # geopotential (m^2 s^-2) => height h/(1000.*g)
                                                  2, 0)
            out_data = apply_conversion(geopotential_height.units_fn, geopotential_height.data,
                                        geopotential_height.fill)
            geopotential_height.update_output(era5_ds, out_data)

            land_mask = MerraConversion(era5_ds, 'singleLevel', 'lsm', 'land mask',
                                        '1=land, 0=ocean', None, 2, 0,)
            land_mask.update_output(era5_ds, land_mask)

            # --- write global attributes
            var = era5_ds['out'].select('temperature')
            nlevel = var.dimensions(full=False)['level']
            nlat = var.dimensions(full=False)['lat']
            nlon = var.dimensions(full=False)['lon']
            setattr(era5_ds['out'], 'NUMBER OF LATITUDES', nlat)
            setattr(era5_ds['out'], 'NUMBER OF LONGITUDES', nlon)
            setattr(era5_ds['out'], 'NUMBER OF PRESSURE LEVELS', nlevel)
            setattr(era5_ds['out'], 'NUMBER OF O3MR LEVELS', nlevel)
            setattr(era5_ds['out'], 'NUMBER OF RH LEVELS', nlevel)
            setattr(era5_ds['out'], 'NUMBER OF CLWMR LEVELS', nlevel)
            lat = era5_ds['out'].select('lat')
            lon = era5_ds['out'].select('lon')
            attr = era5_ds['out'].attr('LATITUDE RESOLUTION')
            attr.set(SDC.FLOAT32, lat.get()[1] - lat.get()[0])
            attr = era5_ds['out'].attr('LONGITUDE RESOLUTION')
            attr.set(SDC.FLOAT32, lon.get()[1] - lon.get()[0])
            attr = era5_ds['out'].attr('FIRST LATITUDE')
            attr.set(SDC.FLOAT32, lat.get()[0])
            attr = era5_ds['out'].attr('FIRST LONGITUDE')
            attr.set(SDC.FLOAT32, lon.get()[0])
            setattr(era5_ds['out'], 'GRIB TYPE', 'not applicable')  # XXX better to just not write this attr?
            setattr(era5_ds['out'], '3D ARRAY ORDER', 'ZXY')  # XXX is this true here?
            setattr(era5_ds['out'], 'MERRA STREAM', "{}".format(era5_ds['ana'].GranuleID.split('.')[0]))
            setattr(era5_ds['out'], 'MERRA History', "{}".format(era5_ds['ana'].History))
            [a.endaccess() for a in [var, lat, lon]]

            era5_ds['out'].end()

    finally:
        for file_name_key in in_files.keys():
            # merra_source_data[k].end()
            LOG.info("Finished {}: {}".format(file_name_key, out_key))

    return out_fname


def build_input_collection(desired_date: datetime, hour_str: str, in_path: Path) -> Dict[str, Path]:
    """Use datetime to build mapping of downloaded input files to process for output."""

    date_str_arg = desired_date.strftime("%Y_%m_%d")
    year = desired_date.strftime("%Y")
    month = desired_date.strftime("%m")
    day = desired_date.strftime("%d")

    if ":" not in hour_str:
        raise ValueError("Hour String must be formatted as HH:mm")
    file_hour = "".join(hour_str.split(":"))
    file_hour = f"{file_hour:0>4}"
    pressure_fn = in_path.joinpath('{}_pressureLevels_hourly_{}.nc'.format(date_str_arg, file_hour))
    single_fn = in_path.joinpath('{}_singleLevel_hourly_{}.nc'.format(date_str_arg, file_hour))

    c = cdsapi.Client()

    if not pressure_fn.exists():
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    'fraction_of_cloud_cover', 'ozone_mass_mixing_ratio', 'relative_humidity',
                    'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content',
                    'specific_humidity', 'specific_snow_water_content', 'temperature',
                    'u_component_of_wind', 'v_component_of_wind', 'geopotential'
                ],
                'pressure_level': LEVELS,
                'month': month,
                'day': day,
                'time': hour_str,
                'year': year,
            },
            pressure_fn)

    if not single_fn.exists():
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    'surface_pressure', 'mean_sea_level_pressure', 'skin_temperature',
                    'geopotential', 'land_sea_mask', 'sea_ice_cover', '2m_temperature',
                    '2m_dewpoint_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
                    'snow_depth', 'total_column_ozone', 'boundary_layer_height',
                    'total_column_water', 'total_column_water_vapour', "total_cloud_cover"
                ],
                'year': year,
                'month': month,
                'day': day,
                'time': hour_str,
            },
            single_fn)

    in_files = { 'pressureLevels': pressure_fn,
                 'singleLevel': single_fn
    }
    return in_files


def process_era5(base_path=None, input_path=None, start_date=None, end_date=None, store_temp=False) -> None:

    out_path_parent = base_path
    try:
        start_dt = datetime.strptime(start_date, '%Y%m%d')
    except ValueError:
        print('usage:\n    python era5clavrx.py 20090101')
        sys.exit()

    if end_date is not None:
        end_dt = datetime.strptime(end_date, '%Y%m%d')
    else:
        end_dt = start_dt

    for dt in pd.date_range(start_dt, end_dt, freq='D'):
        year = dt.strftime("%Y")
        year_month_day = dt.strftime("%Y_%m_%d")
        out_path_full = Path(out_path_parent).joinpath(year)

        try:
            out_path_full.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            msg = "Oops!  {} \n Enter a valid directory with -d flag".format(e)
            raise OSError(msg)

        #for hour in ["0:00", "6:00", "12:00", "18:00"]:
        for hour in ["12:00"]:
            if store_temp:
                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    in_data = build_input_collection(dt, hour, Path(tmp_dir_name))
                    mask_file = str(in_data.pop('mask_file'))
                    LOG.debug("Mask File {}".format(mask_file))
                    out_list = make_era5_one_hour(in_data, out_path_full)
                    LOG.info(', '.join(map(str, out_list)))
            else:
                input_path = Path(input_path).joinpath(year, year_month_day)
                input_path.mkdir(parents=True, exist_ok=True)
                in_data = build_input_collection(dt, hour, input_path)
                out_list = make_era5_one_hour(in_data, out_path_full)
                LOG.info(', '.join(map(str, out_list)))


def argument_parser() -> CommandLineMapping:
    """Parse command line for era5_clavrx.py."""
    parse_desc = (
        """\nRetrieve ERA5 data from Copernicus CDS
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
    process_era5(**parser_args)

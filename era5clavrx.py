# -*- coding: utf-8 -*-
"""Module to convert ERA5 re-analysis data to hdf files for use in CLAVR-x.

usage: era5clavrx.py [-h] [-e end_date] [-t | -i [INPUT_PATH]] [-d [BASE_PATH]] [-v] start_date

Retrieve ERA5 data from GES DISC and process for clavrx input.

positional arguments:
  start_date            Desired processing date as YYYYMMDD

optional arguments:
  -h, --help            show this help message and exit
  -e end_date, --end_date end_date
                        End date as YYYYMMDD not needed when processing one date. (default: None)
  -t, --tmp             Use to store downloaded input files in a temporary location.
                        (default: False)
  -i [INPUT_PATH], --input [INPUT_PATH]
                        Data Input path year/year_month_day subdirs append to path.
                        (default:
                        /apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2/)
  -d [BASE_PATH], --base_dir [BASE_PATH]
                        Parent path used final location year subdirectory appends to this path.
                        (default:
                        /apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2/)
  -v, --verbose         each occurrence increases verbosity 1 level through
                        CRITICAL-ERROR-WARNING-INFO-DEBUG
                        (default: 0)
"""


from __future__ import annotations

import logging
import os
import sys
import tempfile
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import yaml

from merra24clavrx import CommandLineMapping, MerraConversion, apply_conversion

try:
    from datetime import datetime
    from typing import Dict, Union

    import cdsapi
    import numpy as np
    import pandas as pd
    from netCDF4 import Dataset, num2date
    from pyhdf.SD import SD, SDC
except ImportError as e:
    print("Import Error {}".format(e))
    print("Type:  conda activate merra2_clavrx")
    sys.exit(1)

np.seterr(all='ignore')

LOG = logging.getLogger(__name__)

comp_level = 6  # 6 is the gzip default; 9 is best/slowest/smallest file

FOCUS_VAR = ["rh"]
Q_ARR = [0, 0.25, 0.5, 0.75, 1.0]


OUT_PATH_PARENT = '/apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/era5/'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT_DIR, 'yamls', 'ERA5_vars.yaml'), "r") as yml:
    OUTPUT_VARS_DICT = yaml.load(yml, Loader=yaml.Loader)

# this is trimmed to the top CFSR level (i.e., exclude higher than 10hPa)
levels_listings = OUTPUT_VARS_DICT.pop("defined levels")
LEVELS = levels_listings["hPa_levels"]


def total_precipitable_water(tcwv):
    """Calculate the Total Precipitable Water from the Total Column Water Vapor."""
    g = 9.80665   # Acceleration due to gravity
    rho = 997     # water density in kg/m3
    tpw = tcwv/(g*rho)
    return tpw


def rh_on_sigma(t, td):
    """Temperature and Dewpoint Temperature -> relative humidity [%].

    Uses August-Roche-Magnus approximation

    Input:
    t: For ERA-5 this is the 2 meter temperature
    td: For ERA-5 this is the 2 meter dewpoint temperature

    Output:
    rh: Relative Humidity
    """
    rh = 100 * (np.exp((17.625 * td) / (243.04 + td)) / np.exp((17.625 * t) / (243.04 + t)))
    return rh


def check_dataset_times(data_files):
    """Check the times from input datasets to make sure they are concurrent."""
    time_set = dict()
    for file_name_key in data_files.keys():
        time_set[file_name_key] = set()
        t_sds = data_files[file_name_key].variables['time']
        analysis_time = (num2date(t_sds[0], t_sds.units, only_use_python_datetimes=True,
                         only_use_cftime_datetimes=False))
        # this is different for ERA5 versus MERRA2 since there are no units in ERA5 time metadata.
        time_set[file_name_key].add(analysis_time)
    # find set of time common to all input files
    common_times = (time_set['pressureLevels'] &
                    time_set['singleLevel'])

    # these should be single file times, all this
    if len(common_times) != 1:
        raise ValueError('Input files have not produced common times')

    return common_times


def get_input_variables(in_ds: Dict[str, Union[Dataset, SD]]) -> Dict[str, MerraConversion]:
    """Prepare input data variables."""
    output_vars = dict()
    for out_key in OUTPUT_VARS_DICT:
        rsk = OUTPUT_VARS_DICT[out_key]

        output_vars[out_key] = MerraConversion(
            in_ds[rsk['in_file']],
            rsk['in_varname'],
            out_key,
            rsk['out_units'],
            rsk['ndims_out'],
            0)

    return output_vars


def write_global_attributes(out_ds: SD) -> None:
    """Write global attributes."""
    var = out_ds.select('temperature')
    nlevel = var.dimensions(full=False)['level']
    nlat = var.dimensions(full=False)['lat']
    nlon = var.dimensions(full=False)['lon']
    setattr(out_ds, 'NUMBER OF LATITUDES', nlat)
    setattr(out_ds, 'NUMBER OF LONGITUDES', nlon)
    setattr(out_ds, 'NUMBER OF PRESSURE LEVELS', nlevel)
    setattr(out_ds, 'NUMBER OF O3MR LEVELS', nlevel)
    setattr(out_ds, 'NUMBER OF RH LEVELS', nlevel)
    setattr(out_ds, 'NUMBER OF CLWMR LEVELS', nlevel)
    lat = out_ds.select('lat')
    lon = out_ds.select('lon')
    attr = out_ds.attr('LATITUDE RESOLUTION')
    attr.set(SDC.FLOAT32, (lat.get()[1] - lat.get()[0]).item())
    attr = out_ds.attr('LONGITUDE RESOLUTION')
    attr.set(SDC.FLOAT32, (lon.get()[1] - lon.get()[0]).item())
    attr = out_ds.attr('FIRST LATITUDE')
    attr.set(SDC.FLOAT32, (lat.get()[0]).item())
    attr = out_ds.attr('FIRST LONGITUDE')
    attr.set(SDC.FLOAT32, (lon.get()[0]).item())
    setattr(out_ds, 'GRIB TYPE', 'not applicable')  # XXX better to just not write this attr?
    setattr(out_ds, '3D ARRAY ORDER', 'YXZ')  # XXX is this true here?
    [a.endaccess() for a in [var, lat, lon]]

    out_ds.end()


def make_era5_one_hour(in_files: Dict[str, Path], out_dir: Path):
    """Read input, parse times, and run conversion on one day at a time."""
    era5_ds = dict()

    for file_name_key in in_files.keys():
        era5_ds[file_name_key] = Dataset(in_files[file_name_key])

    out_time = (check_dataset_times(era5_ds)).pop()
    LOG.info('    working on time: {}'.format(out_time))
    out_fname = str(out_dir.joinpath(out_time.strftime('era5.%y%m%d%H_F000.hdf')))
    LOG.info(out_fname)
    # TRUNC will clobber existing
    era5_ds['out'] = SD(out_fname, SDC.WRITE | SDC.CREATE | SDC.TRUNC)

    out_vars = get_input_variables(era5_ds)

    # to insure that all the variables are filled before attempting the data conversions,
    # read all the output_vars and then convert before setting the hdf variables out.
    for out_key, rsk in OUTPUT_VARS_DICT.items():
        units_fn = rsk["units_fn"]
        var_fill = out_vars[out_key].fill
        out_data = out_vars[out_key].data

        if out_key == "rh at sigma=0.995":
            dpT = out_vars["dewpoint temperature at sigma=0.995"].data
            out_data = rh_on_sigma(out_data, dpT)

        out_data = apply_conversion(units_fn, out_data, var_fill)

        out_vars[out_key].update_output(era5_ds, "ERA-5->{}".format(rsk["in_file"]), out_data)

    write_global_attributes(era5_ds["out"])

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

    in_files = {'pressureLevels': pressure_fn,
                'singleLevel': single_fn}
    return in_files


def process_era5(base_path=None, input_path=None, start_date=None,
                 end_date=None, store_temp=False) -> None:
    """Read input, parse times, and run conversion."""
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
        except OSError as emsg:
            msg = "Oops!  {} \n Enter a valid directory with -d flag".format(emsg)
            raise OSError(msg)

        for hour in ["0:00", "6:00", "12:00", "18:00"]:
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
                LOG.info(out_list)


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
                       help="Data Input path year/year_month_day append to path.")
    parser.add_argument('-d', '--base_dir', dest='base_path', action='store', nargs='?',
                        type=str, required=False, default=OUT_PATH_PARENT, const=OUT_PATH_PARENT,
                        help="Parent path: year subdirectory appends to this path.")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=2,
                        help='each occurence increases verbosity 1 level through '
                             'ERROR-WARNING-INFO-DEBUG')

    args = vars(parser.parse_args())
    verbosity = args.pop('verbosity', None)

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(format='%(module)s:%(lineno)d:%(levelname)s:%(message)s',
                        level=levels[min(3, verbosity)])

    return args


if __name__ == '__main__':

    parser_args = argument_parser()
    process_era5(**parser_args)

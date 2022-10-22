#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  This code produces CLAVRx compatible files from navgem global model data.
#
#  filter_by_keys is used to select data from the grib into xarray open_dataset
#  however, it is easier to see the data using cfgrib directly.  cfgrib will
#  create a list of loaded xarray Datasets:
#  `import cfgrib
#   cfgrib.open_datasets(<grib_file>)`
#
"""Convert navgem model data into CLAVRx compatible input."""
from __future__ import annotations

import glob
import itertools
import logging
import os
import sys

import yaml

import navgem_nomads_get as navgem_get

try:
    import argparse
    import datetime
    from datetime import timedelta
    from typing import Callable, Dict, List, Optional, TypedDict

    import numpy as np
    import pandas as pd
    import xarray as xr
    from pyhdf.SD import SD, SDC
except ImportError as e:
    print("Import Error {}".format(e))
    print("Type:  conda activate merra2_clavrx")
    sys.exit(1)

from dateutil.parser import parse

from conversion_class import output_dtype
from conversions import CLAVRX_FILL, COMPRESSION_LEVEL

LOG = logging.getLogger(__name__)

OUT_PATH_PARENT = '/apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/navgem/'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT_DIR, 'yamls', 'NAVGEM_nrl_usgodae_vars.yaml'), "r") as yml:
    OUTPUT_VARS_DICT = yaml.load(yml, Loader=yaml.Loader)


class NavgemCommandLineMapping(TypedDict):
    """Type hints for result of the argparse parsing."""

    start_date: datetime.date
    end_date: Optional[str]
    base_path: str
    input_path: str
    forecast_hours: List[int]
    local: List[str]
    model_run: str


class DateParser(argparse.Action):
    """Parse a date from argparse to a datetime."""

    def __call__(self, parser, namespace, values, option_strings=None):
        """Parse a date from argparse to a datetime."""
        setattr(namespace, self.dest, parse(values).date())


def set_dim_names(data_array, ndims_out, out_name, out_sds):
    """Set dimension names in hdf file."""
    if ndims_out in (2, 3):
        coords_dict = {"z": (2, "level"),
                       "x": (1, "lon"),
                       "y": (0, "lat"),
                       }
    else:
        coords_dict = {"z": (0, "level"),
                       "x": (0, "lon"),
                       "y": (0, "lat"),
                       }

    out_sds.dimensions()
    for dim in data_array.dims:
        axis_num, dim_name = coords_dict.get(dim, dim)

        out_sds.dim(axis_num).setname(dim_name)

    msg_str = "Out {} for {} ==> {}."
    msg_str = msg_str.format(out_sds.dimensions(),
                             data_array.name,
                             out_name)
    LOG.info(msg_str)

    return out_sds


def refill(data: xr.DataArray, old_fill: float) -> np.ndarray:
    """Assumes CLAVRx fill value instead of variable attribute."""
    if data.dtype in (np.float32, np.float64):
        data = xr.where(np.isnan(data), CLAVRX_FILL, data)
        data = xr.where(data == old_fill, CLAVRX_FILL, data)
    return data


def update_output(sd, out_name, rsk, data_array, out_fill, data_source):
    """Finalize output variables."""
    out_units = rsk["out_units"]
    ndims_out = rsk["ndims_out"]
    str_template = f"Writing Input name: {data_array.long_name} ==> Output Name: {out_name}"
    LOG.info(str_template)
    data_array = reshape(data_array, out_name, ndims_out)

    out_sds = sd["out"].create(out_name,
                               output_dtype(out_name, data_array.dtype),
                               data_array.shape)
    out_sds.setcompress(SDC.COMP_DEFLATE, value=COMPRESSION_LEVEL)
    set_dim_names(data_array, ndims_out, out_name, out_sds)
    if out_name == "lon":
        out_sds.set(data_array.data)
    else:
        out_data = refill(data_array.data, out_fill)
        out_sds.set(out_data)

    if out_fill is not None:
        out_sds.setfillvalue(CLAVRX_FILL)

    if out_units is None or out_units in ("none", "None"):
        try:
            out_sds.units = data_array.units
        except AttributeError:
            out_sds.units = "1"
    else:
        out_sds.units = out_units

    unit_desc = " in [{}]".format(out_sds.units)

    out_sds.source_data = ("{}->{}{}".format("{}->{}".format(data_source, data_array.name),
                                             data_array.name, unit_desc))

    out_sds.long_name = data_array.long_name
    out_sds.endaccess()


def modify_shape(data_array: xr.DataArray) -> xr.DataArray:
    """Modify shape from dims (level, lat, lon) to (lat,lon,level)."""
    ndim = len(data_array.dims)
    if ndim == 3:
        return data_array.transpose("y", "x", "z")
    if ndim == 2:
        return data_array.transpose("y", "x")
    else:
        return data_array


def reshape(data_array: xr.DataArray,
            out_name: str, ndims_out: int) -> xr.DataArray:
    """Reshape data toa->surface and (lat, lon, level)."""
    if ndims_out == 3:
        # clavr-x needs toa->surface not surface->toa
        if all(np.diff(data_array["isobaricInhPa"] < 0)):
            data_array = data_array.isel(z=slice(None, None, -1))

    return modify_shape(data_array)


def all_equal(iterable):
    """Return True if all the elements are equal to each other."""
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)


def starmap(function, iterable):
    """Itertools apply function to iterable."""
    # starmap(pow, [(2,5), (3,2), (10,3)]) --> 32 9 1000
    for args in iterable:
        yield function(*args)


def build_supplemental_ozone_fn(start_time: datetime.datetime,
                                model_init_hour: str, forecast_hour: str,
                                o3mr_data_crc, o3mr_load_grib_crc) -> str:
    """Use information from ozone mixing ratio yaml setup to build filepath.

    :param start_time: start_time is part of the filename format string and must match yaml
        (so if this variable changes here, it should change in the yaml as well.)
    :param model_init_hour: model_init_hour is part of a filename format string and must match yaml
        (so if this variable changes here, it should change in the yaml as well.)
    :param forecast_hour: forecast_hour is part of a filename format string and must match yaml
        (so if this variable changes here, it should change in the yaml as well.)
    :param o3mr_data_crc: In the yaml, the data_arrays section describe the type of variable
         loaded.  For ozone, there is a special descriptor for the source file type.
         Currently only gfs_hdf has been tested.
    :param o3mr_load_grib_crc: This is the datasets section of the yaml and will contain
         the appropriate filepath and filename pattern for the source files.
    :return: The full filepath based on these input of the ozone mixing ratio supplement file.
    """
    o3mr_format_key = o3mr_data_crc["data_source_format"]
    o3mr_fp = o3mr_load_grib_crc[o3mr_format_key]

    # both had an eval next two lines
    base_dir = eval(o3mr_fp["directory"])
    base_fn = eval(o3mr_fp["file_pattern"])

    # read complementary GFS 03MR here
    full_filepath = os.path.join(base_dir, base_fn)

    msg = f"{start_time}, {model_init_hour}, F{forecast_hour}, {o3mr_format_key} => {full_filepath}"
    LOG.info(msg)

    return full_filepath


def read_gfs(o3mr_fn: str) -> xr.Dataset:
    """Use the GFS ozone mixing ratio.

    :param o3mr_fn: filepath of the gfs file.
    :return: ozone mixing ratio in kg/kg
    """
    gfs_ds = xr.open_dataset(o3mr_fn, engine="cfgrib",
                             filter_by_keys={'typeOfLevel': 'isobaricInhPa',
                                             'shortName': "o3mr"})

    # for predictability, put dims in the CLAVRx order
    gfs_ds = gfs_ds.transpose("latitude", "longitude", "isobaricInhPa")
    gfs_ds = gfs_ds.sortby(gfs_ds["latitude"], ascending=True)
    return gfs_ds


def reformat_levels(datasets_dict):
    """Verify output levels of dataset are on CLAVRx levels."""
    hPa_levels = [1000.0, 975.0, 950.0, 925.0, 900.0, 850.0,
                  800.0, 750.0, 700.0, 650.0, 600.0, 550.0,
                  500.0, 450.0, 400.0, 350.0, 300.0, 250.0,
                  200.0, 150.0, 100.0, 70.0, 50.0, 30.0, 20.0, 10.0]

    for key, ds in datasets_dict.items():
        try:
            new_data = ds.sel(isobaricInhPa=hPa_levels)
            datasets_dict.update({key: new_data})
        except KeyError as kerr:
            msg = "{} for {} in coords {}".format(kerr, key, ds.coords)
            LOG.warning(msg)
    return datasets_dict


def apply_conversion(scale_func: Callable, data: xr.DataArray, fill) -> xr.DataArray:
    """Apply fill to converted data after function."""
    data_attrs = data.attrs
    converted = data.copy(deep=True)
    converted = scale_func(converted)

    if data.dims == converted.dims:
        if fill is not None:
            converted = xr.where(data == fill, fill, converted)
        if np.isnan(data).any():
            converted = xr.where(np.isnan(data), fill, converted)

    converted = converted.astype(np.float32)
    converted = converted.assign_attrs(data_attrs)

    return converted


def write_output_variables(in_datasets, out_vars_setup: Dict):
    """Write variables to file."""
    for var_key, rsk in out_vars_setup.items():
        if var_key in ["rh", "rh_level"]:
            file_key = "rh"
        else:
            file_key = rsk["dataset"]
        print(var_key, rsk)
        var_name = rsk["cfVarName"]
        out_var = in_datasets[file_key][var_name]
        units_fn = rsk["units_fn"]
        if "data_source_format" in rsk.keys():
            source_model = rsk["data_source_format"]
        else:
            source_model = "NAVGEM"

        try:
            var_fill = out_var.fill_value
        except AttributeError:
            var_fill = 1e+20  # match merra2 :/

        # return a new xarray with converted data, otherwise, the process
        # is different for coordinate attributes.
        out_var = apply_conversion(units_fn, out_var, fill=var_fill)

        update_output(in_datasets, var_key, rsk,
                      out_var, var_fill, source_model)


def get_dim_list_string(param: Dict[str]) -> str:
    """Create an attribute string from the dims."""
    dim_list = []
    for dim_name in param.keys():
        if dim_name.lower() in ["lat", "latitude"]:
            out_dim = 'Y'
        elif dim_name.lower() in ["lon", "longitude"]:
            out_dim = 'X'
        elif dim_name.lower() in ["level", "pressure", "rh_level",
                                  "press", "gph", "height"]:
            out_dim = 'Z'
        else:
            out_dim = dim_name
        dim_list.append(out_dim)
    return "".join(dim_list)


def write_global_attributes(out_ds: SD) -> None:
    """Write global attributes."""
    var = out_ds.select('temperature')
    nlevel = var.dimensions(full=False)['level']
    nlat = var.dimensions(full=False)['lat']
    nlon = var.dimensions(full=False)['lon']
    nlevel_rh = (out_ds.select("rh")).dimensions(full=False)['rh_level']
    setattr(out_ds, 'NUMBER OF LATITUDES', nlat)
    setattr(out_ds, 'NUMBER OF LONGITUDES', nlon)
    setattr(out_ds, 'NUMBER OF PRESSURE LEVELS', nlevel)
    setattr(out_ds, 'NUMBER OF O3MR LEVELS', nlevel)
    setattr(out_ds, 'NUMBER OF RH LEVELS', nlevel_rh)
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

    dim_description = get_dim_list_string(var.dimensions())
    setattr(out_ds, '3D ARRAY ORDER', dim_description)  # XXX is this true here?
    [a.endaccess() for a in [var, lat, lon]]

    out_ds.end()


def reorder_dimensions(datasets):
    """Reorder and rename dimensions."""
    for key, ds in datasets.items():
        # rename_dims
        for dim_key in ds.dims:
            dim = ds[dim_key]
            if dim.long_name.lower() == "longitude":
                ds = ds.rename_dims({dim.name: "x"})
            elif dim.long_name.lower() == "latitude":
                ds = ds.rename_dims({dim.name: "y"})
            elif dim.long_name.lower() == "pressure":
                ds = ds.rename_dims({dim.name: "z"})
        datasets[key] = ds

    return datasets


def read_one_hour_navgem(in_files: List[str],
                         out_dir: str,
                         forecast_hour: int):
    """Read input, parse times, and run conversion on one day at a time."""
    datasets = dict()
    timestamps = list()
    model_runs = list()
    out_fname = None

    # build time selection based on forecast hour and model date.
    tdelta = (pd.Timedelta("{} hours".format(forecast_hour))).to_timedelta64()

    levels = None
    for model_file in in_files:
        for ds_key, filter_dict in OUTPUT_VARS_DICT["datasets"].items():
            if ds_key in ["o3mr"]:
                pass
            else:
                ds = xr.open_dataset(model_file, engine="cfgrib",
                                     backend_kwargs={'filter_by_keys': filter_dict})

                # select the forecast time from the steps and squeeze the steps if necessary
                if "step" in ds.dims:
                    model_runs.append(pd.to_datetime(ds.time.data))
                    ds = ds.sel(step=tdelta)

                for coord_key in ds.coords:
                    coord_da = ds.coords[coord_key]
                    if coord_da.long_name.lower() == "pressure":
                        if levels is None:
                            levels = coord_da
                        else:
                            levels = np.concatenate((levels, coord_da))
                LOG.info(filter_dict)
                timestamps.append(pd.to_datetime(ds.valid_time.data))
                # after all work on initial dataset, assign to dictionary.
                datasets.update({ds_key: ds})

        if all_equal(timestamps) and all_equal(model_runs):
            model_init_hour = model_runs[0].strftime("%H")
            forecast_hour = forecast_hour.zfill(3)
            navgem_fn_pattern = f"navgem.%y%m%d{model_init_hour}_F{forecast_hour}.hdf"
            out_fname = timestamps[0].strftime(navgem_fn_pattern)
            out_fname = os.path.join(out_dir, out_fname)
            LOG.info('    working on {}'.format(out_fname))

            o3mr_fn = build_supplemental_ozone_fn(timestamps[0], model_init_hour,
                                                  forecast_hour,
                                                  OUTPUT_VARS_DICT["data_arrays"]["o3mr"],
                                                  OUTPUT_VARS_DICT["datasets"]["o3mr"])
            gfs_o3mr = read_gfs(o3mr_fn)

            datasets.update({"o3mr": gfs_o3mr})
            datasets = reformat_levels(datasets)
            datasets = reorder_dimensions(datasets)

            # TRUNC will clobber existing
            datasets['out'] = SD(out_fname, SDC.WRITE | SDC.CREATE | SDC.TRUNC)

            write_output_variables(datasets, OUTPUT_VARS_DICT["data_arrays"])
        else:
            ts_str = ' '.join(timestamps)
            msg = "Timestamps are not equal: {}".join(ts_str)
            raise ValueError(msg)

    return out_fname


def get_model_run_string(model_date_dt, run_hour):
    """Given a model date and model run hour, create a model run string."""
    model_date = model_date_dt.strftime("%Y%m%d")

    msg = "Model Date {}: {}".format(type(model_date), model_date)
    LOG.debug(msg)
    msg = "Run Hour (model run) {}: {}".format(type(run_hour), run_hour)
    LOG.debug(msg)

    dt_model_run = datetime.datetime.strptime("{} {}".format(model_date, run_hour), "%Y%m%d %H")
    model_run_str = dt_model_run.strftime("%Y%m%d%H")

    return model_run_str


def build_filepath(data_dir, dt: datetime, dir_type="output") -> str:
    """Create output path from in put model run information."""
    year = dt.strftime("%Y")
    year_month_day = dt.strftime("%Y_%m_%d")
    if dir_type == "output":
        this_filepath = os.path.join(data_dir, year)
        LOG.info(f"Making {this_filepath}")
        os.makedirs(this_filepath, exist_ok=True)
    elif dir_type == "input":
        this_filepath = os.path.join(data_dir, year, year_month_day)
    else:
        raise RuntimeError('dir_type options are either ["input", "output"]')

    return this_filepath


def process_navgem(base_path=None, input_path=None, start_date=None,
                   url=None, model_run=None, forecast_hours=None,
                   local=None) -> None:
    """Read input, parse times, and run conversion."""
    if local is not None:
        raise RuntimeError("Local is defined, process_navgem subroutine pulls data.")

    out_list = None
    dt = start_date

    input_path = build_filepath(input_path, dt, dir_type="input")
    os.makedirs(input_path, exist_ok=True)

    # start_time is needed for url string.
    start_time = dt + timedelta(hours=int(model_run))
    LOG.debug("Running for {}".format(start_time.strftime("%Y%m%d%H")))
    url = eval(url)
    soup = navgem_get.create_soup(url)

    out_fp = build_filepath(base_path, dt)
    model_run_str = get_model_run_string(dt, model_run)

    # source_site is ncep (filename format: f'navgem_{navgem_run}f{forecast}.grib2')
    if "nomads" in url:
        in_files = navgem_get.search_date(soup, url, model_run, forecast_hours, out_fp)
    else:
        model_run_dt = datetime.datetime.strptime(model_run_str, "%Y%m%d%H")
        navgem_get.url_search_nrl(soup, url, model_run_dt,
                                  forecast_hours, out_path=input_path)
        in_files = [navgem_get.concat_gribs_in_one(input_path, model_run_str)]

    for forecast_hour in forecast_hours:
        out_list = read_one_hour_navgem(in_files, out_fp, forecast_hour)
    LOG.info(out_list)


def argument_parser() -> NavgemCommandLineMapping:
    """Parse command line for navgem_clavrx.py."""
    parse_desc = (
        """\nProcess navgem data previously downloaded from NCEP nomads or NRL ftp.""")

    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=parse_desc,
                                     formatter_class=formatter)
    group = parser.add_mutually_exclusive_group()

    parser.add_argument('start_date', action=DateParser,
                        default=datetime.datetime.now(), help="Processing date")
    parser.add_argument('-m', '--model_run', default='00',
                        help="Model run hour; i.e. 00, 03, 06, 09, 12...")
    parser.add_argument('-f', '--forecast_hours', nargs='+',
                        default=[3, 6, 9, 12],
                        help="The forecast hours from this model run.")

    group.add_argument('-u', '--url', default=OUTPUT_VARS_DICT["url"],
                       help='alternative url string.')
    parser.add_argument('-i', '--input', dest='input_path', action='store',
                        type=str, required=False, default=None, const=None,
                        help="Input path for the data download.")
    # store_true evaluates to False when flag is not in use (flag invokes the store_true action)
    group.add_argument('-l', '--local', default=None, nargs="*",
                       help="List of files already downloaded and in location of input_path")
    parser.add_argument('-d', '--base_path', action='store', nargs='?',
                        type=str, required=False, default=OUT_PATH_PARENT, const=OUT_PATH_PARENT,
                        help="Parent output path: year subdirectory appends to this path.")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=2,
                        help='each occurrence increases verbosity 1 level through '
                             'ERROR-WARNING-INFO-DEBUG')

    args = vars(parser.parse_args())
    verbosity = args.pop('verbosity', None)

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(format='%(module)s:%(lineno)d:%(levelname)s:%(message)s',
                        level=levels[min(3, verbosity)])

    return args


if __name__ == '__main__':

    parser_args = argument_parser()

    if parser_args["local"] is None:
        process_navgem(**parser_args)
    else:
        inp = parser_args["input_path"]
        inp = build_filepath(inp, parser_args["start_date"], dir_type="input")
        if len(parser_args["local"]) == 0:
            fn = glob.glob(os.path.join(inp, "*.grib"))
        else:
            fn = parser_args["local"]

        dt_in = parser_args["start_date"]
        out_path = build_filepath(parser_args['base_path'],
                                  dt_in)
        fn_paths = (os.path.join(inp, x) for x in fn)
        fn_paths = list(fn_paths)
        out_fnames = []
        for forecast in parser_args["forecast_hours"]:
            out_fnames.append(read_one_hour_navgem(fn_paths, out_path, forecast))
        LOG.info(out_fnames)

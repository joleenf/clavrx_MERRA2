#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  This code produces CLAVRx compatible files from navgem global model data.
#
#  Some of the conversion_class code from MERRA2 and ERA5 is retained.
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
    from datetime import datetime, timedelta
    from typing import Callable, Dict, List

    import numpy as np
    import pandas as pd
    import xarray as xr
    from pyhdf.SD import SD, SDC
except ImportError as e:
    print("Import Error {}".format(e))
    print("Type:  conda activate merra2_clavrx")
    sys.exit(1)

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from conversion_class import CommandLineMapping, ReanalysisConversion
from conversions import CLAVRX_FILL

LOG = logging.getLogger(__name__)

# FOCUS_VAR = ["rh"]
# Q_ARR = [0, 0.25, 0.5, 0.75, 1.0]

OUT_PATH_PARENT = '/apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/navgem/'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT_DIR, 'yamls', 'NAVGEM_nrl_usgodae_vars.yaml'), "r") as yml:
    OUTPUT_VARS_DICT = yaml.load(yml, Loader=yaml.Loader)

levels_listings = OUTPUT_VARS_DICT.pop("defined levels")
hPa_LEVELS = levels_listings["hPa_levels"]
rh_LEVELS = levels_listings["rh_hPa_levels"]


class NavgemConversion(ReanalysisConversion):
    """Builds a class for navgem consistent with old writers."""

    def __init__(self, xr_dataset=None, shortname=None,
                 out_name=None, out_units=None, ndims_out=None) -> None:
        """Based on variable, adjust shape, apply fill and determine dtype."""
        self.nc_dataset = xr_dataset
        self.shortname = shortname
        self.out_name = out_name
        self.out_units = out_units
        self.ndims_out = ndims_out

        self.xrarray = xr_dataset[shortname]

    def __repr__(self):
        """Report the name conversion when creating this object."""
        str_template = "Input name: {} ==> Output Name: {}"
        return str_template.format(self.long_name(), self.out_name)

    def get_units(self):
        """If possible, get units from attrs."""
        return self.nc_dataset.variables[self.shortname].attrs.get("units", "")

    def long_name(self):
        """Return long name from input file."""
        return self[self.shortname].attrs.get("long_name", None)

    def set_dim_names(self, out_sds):
        """Set dimension names in hdf file."""
        if self.ndims_out in (2, 3):
            coords_dict = {"z": (2, "pressure"),
                           "x": (1, "longitude"),
                           "y": (0, "latitude")}
        else:
            coords_dict = {"z": (0, "level"),
                           "x": (0, "longitude"),
                           "y": (0, "latitude")}

        out_sds.dimensions()
        for dim in self.xrarray.dims:
            axis_num, dim_name = coords_dict.get(dim, dim)
            LOG.debug("{} to axis {}, {}".format(self.out_name, axis_num, dim_name))

            out_sds.dim(axis_num).setname(dim_name)

        msg_str = "Out {} for {} ==> {}."
        msg_str = msg_str.format(out_sds.dimensions(),
                                 self.shortname,
                                 self.out_name)
        LOG.debug(msg_str)

        return out_sds

    @staticmethod
    def _refill(data: xr.DataArray, old_fill: float) -> np.ndarray:
        """Assumes CLAVRx fill value instead of variable attribute."""
        if data.dtype in (np.float32, np.float64):
            data = xr.where(np.isnan(data), CLAVRX_FILL, data)
            data = xr.where(data == old_fill, CLAVRX_FILL, data)
        return data

    @staticmethod
    def _reorder_lon(xr_data: xr.DataArray) -> xr.DataArray:
        """Check if lon needs re-ordering to -180 at index 0."""
        start_lon = xr_data['longitude'].isel(x=0)

        if start_lon >= 359.5 or start_lon == 180.0:
            return xr_data.reindex(longitude=list(reversed(xr_data.lon)))

        if start_lon == 0.0:
            xr_data["longitude"] = xr_data["longitude"] - 180.0
            return xr_data
        elif start_lon == -180.:
            return xr_data
        else:
            msg = "NAVGEM lon start is {} neither 0 nor -180.".format(start_lon)
            raise NotImplementedError(msg)

    def _shift_lon(self):
        """Pass the shift_lon from Reanalysis class and just return data again."""
        return self.xrarray.data

    def _modify_shape(self):
        """Modify shape based on output characteristics."""
        if self.ndims_out == 3:
            # clavr-x needs level to be the last dim (make lat,lon,level)
            return self.xrarray.transpose("y", "x", "z").shape
        if self.ndims_out == 2:
            return self.xrarray.transpose("y", "x").shape
        else:
            return self.xrarray.shape

    @property
    def data(self):
        """Wrap get data from xarray using self.data."""
        data = self.xrarray.data
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        return data


def expand_dataset(data, other, fill=np.nan, dim="z"):
    """Expand the dataset along the given dimension and fill with fill."""
    # stack the data
    stacked = data.stack(new_dim=("x", "y"))
    new_shape = (other.sizes[dim], stacked.sizes["new_dim"])
    new_coords = other.coords.update(data.coords["new_dim"])
    new_array = xr.DataArray(np.full_like(new_shape, fill),
                             dims=data.dims,
                             coords=new_coords)

    return new_array


def all_equal(iterable):
    """Return True if all the elements are equal to each other."""
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)


def starmap(function, iterable):
    """Itertools apply function to iterable."""
    # starmap(pow, [(2,5), (3,2), (10,3)]) --> 32 9 1000
    for args in iterable:
        yield function(*args)


def apply_conversion(scale_func: Callable, data: xr.DataArray, fill) -> np.ndarray:
    """Apply fill to converted data after function."""
    converted = data.copy(deep=True)
    converted = scale_func(converted)

    if fill is not None:
        converted = xr.where(data == fill, fill, converted)
    if np.isnan(data).any():
        converted = xr.where(np.isnan(data), fill, converted)

    return converted


def build_output_variables(in_ds: Dict[str, xr.Dataset],
                           out_vars_setup: Dict) -> Dict[str, NavgemConversion]:
    """Prepare output data variables."""
    out_vars = dict()
    for out_key, rsk in out_vars_setup.items():
        file_key = rsk["dataset"] if out_key != "rh" else "rh"
        out_vars["data_object"] = NavgemConversion(
            in_ds[file_key],
            rsk["shortname"],
            out_key,
            rsk['out_units'],
            rsk['ndims_out'],
        )
        out_vars["units_fn"] = rsk["units_fn"]

        yield out_vars


def write_output_variables(in_datasets, out_vars, levels):
    """Write variables to file."""
    while True:
        try:
            current_var = next(out_vars)
            navgem = current_var["data_object"]
            out_var = navgem.xrarray
            file_tag = navgem.out_name
            units_fn = current_var["units_fn"]

            try:
                var_fill = out_var.fill_value
            except AttributeError:
                var_fill = 1e+20  # match merra2 :/

            # return a new xarray with converted data, otherwise, the process
            # is different for coordinate attributes.
            out_var = apply_conversion(units_fn, out_var, fill=var_fill)

            # update levels so that all fields have all levels...(??)
            # Maybe don't need this?
            # if "z" in out_var.dims:
            #     out_var = expand_dataset(levels, fill=var_fill, dim="z")

            # TODO:  Write as xarray Dataset??
            navgem.update_output(in_datasets, "NAVGEM->{}".format(file_tag),
                                 out_var.data, var_fill)

        except StopIteration:
            break


def write_global_attributes(out_ds: SD) -> None:
    """Write global attributes."""
    var = out_ds.select('temperature')
    nlevel = var.dimensions(full=False)['pressure']
    nlat = var.dimensions(full=False)['latitude']
    nlon = var.dimensions(full=False)['longitude']
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


def reorder_dimensions(datasets):
    """Reorder and rename dimensions."""
    for key, ds in datasets.items():
        # rename_dims
        for dim_key in ds.dims:
            if ds[dim_key].long_name.lower() == "longitude":
                ds = ds.rename_dims({ds[dim_key].name: "x"})
            elif ds[dim_key].long_name.lower() == "latitude":
                ds = ds.rename_dims({ds[dim_key].name: "y"})
            elif ds[dim_key].long_name.lower() == "pressure":
                ds = ds.rename_dims({ds[dim_key].name: "z"})

        if len(ds.dims) == 3:
            datasets[key] = ds.transpose("z", "y", "x")
        else:
            datasets[key] = ds.transpose("y", "x")

    return datasets


def read_one_hour_navgem(in_files: List[str],
                         out_dir: str,
                         forecast_hr: int):
    """Read input, parse times, and run conversion on one day at a time."""
    datasets = dict()
    cfgrib_kwargs = OUTPUT_VARS_DICT["datasets"]
    timestamps = list()

    # build time selection based on forecast hour and model date.
    tdelta = (pd.Timedelta("{} hours".format(forecast_hr))).to_timedelta64()

    levels = None
    for model_run in in_files:
        for ds_key, filter_dict in cfgrib_kwargs.items():
            ds = xr.open_dataset(model_run, engine="cfgrib",
                                 backend_kwargs={'filter_by_keys': filter_dict})

            # select the forecast time from the steps and squeeze the steps if necessary
            if "step" in ds.dims:
                ds = ds.sel(step=tdelta)

            for coord_key in ds.coords:
                coord_da = ds.coords[coord_key]
                if coord_da.long_name.lower() == "pressure":
                    if levels is None:
                        levels = coord_da
                    else:
                        levels = np.concatenate((levels, coord_da))

            ts = (pd.to_datetime(ds.valid_time.data).strftime('navgem.%y%m%d%H_F000.hdf'))
            timestamps.append(ts)
            # after all work on initial dataset, assign to dictionary.
            datasets.update({ds_key: ds})
        # broadcast the 3D data to one cube (and replace in the key for isobaricInhPa and rh)
        datasets["isobaricInhPa"], datasets['rh'] = xr.broadcast(datasets['isobaricInhPa'],
                                                                 datasets["rh"])

        datasets = reorder_dimensions(datasets)

        if all_equal(timestamps):
            LOG.info('    working on time: {}'.format(timestamps[0]))
            out_fname = os.path.join(out_dir, (timestamps[0]))
            LOG.info(out_fname)
            # TRUNC will clobber existing
            datasets['out'] = SD(out_fname, SDC.WRITE | SDC.CREATE | SDC.TRUNC)

            out_obj = build_output_variables(datasets, OUTPUT_VARS_DICT["data_arrays"])

            write_output_variables(datasets, out_obj, levels)
        else:
            ts_str = ' '.join(timestamps)
            msg = "Timestamps are not equal: {}".join(ts_str)
            raise ValueError(msg)

        write_global_attributes(datasets["out"])

    return out_fname


def get_model_run_string(model_date_dt, model_run):
    """Given a model date and model run hour, create a model run string."""
    model_date = model_date_dt.strftime("%Y%m%d")
    dt_model_run = datetime.strptime("{} {}".format(model_date, model_run), "%Y%m%d %H")
    model_run_str = dt_model_run.strftime("%Y%m%d%H")

    return model_run_str


def create_full_outpath(base_path, dt, model_run):
    """Create output path from in put model run information."""
    model_run_str = get_model_run_string(dt, model_run)
    year = dt.strftime("%Y")

    out_path = os.path.join(base_path, year, model_run_str)
    os.makedirs(out_path, exist_ok=True)

    return out_path


def process_navgem(base_path=None, input_path=None, start_date=None,
                   url=None, run_hour=None, forecast_hours=None) -> None:
    """Read input, parse times, and run conversion."""
    dt = start_date
    year = dt.strftime("%Y")
    year_month_day = dt.strftime("%Y_%m_%d")

    input_path = os.path.join(input_path, year, year_month_day)
    os.makedirs(input_path, exist_ok=True)

    # start_time is needed for url string.
    start_time = dt + timedelta(hours=int(run_hour))
    LOG.debug("Running for {}".format(start_time.strftime("%Y%m%d%H")))
    url = eval(url)
    soup = navgem_get.create_soup(url)

    model_run_str = get_model_run_string(dt, run_hour)
    model_run = datetime.strptime(model_run_str, "%Y%m%d%H")
    out_path = create_full_outpath(base_path, dt, run_hour)

    # source_site is ncep (filename format: f'navgem_{navgem_run}f{forecast}.grib2')
    if "nomads" in url:
        in_files = navgem_get.search_date(soup, url, run_hour, forecast_hours, out_path)
    else:
        navgem_get.url_search_nrl(soup, url, model_run,
                                  forecast_hours, out_path=input_path)
        in_files = [navgem_get.concat_gribs_in_one(input_path, model_run_str)]

    for forecast_hr in forecast_hours:
        out_list = read_one_hour_navgem(in_files, out_path, forecast_hr)
    # in_data, out_path_full)
    LOG.info(out_list)


def argument_parser() -> CommandLineMapping:
    """Parse command line for navgem_clavrx.py."""
    parse_desc = (
        """\nProcess navgem data previously downloaded from NCEP nomads or NRL ftp.""")

    formatter = ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=parse_desc,
                            formatter_class=formatter)
    group = parser.add_mutually_exclusive_group()

    parser.add_argument('start_date', type=str,
                        default="datetime.now().strftime('%Y%m%d')",
                        help="Desired processing date as YYYYMMDD")
    parser.add_argument('run_hour', default='00',
                        help="Model run hour; i.e. 00, 03, 06, 09, 12...")
    parser.add_argument('-f', '--forecast_hours', nargs='+',
                        default=[3, 6, 12],
                        help="The forecast hours from this model run.")

    group.add_argument('-u', '--url', default=OUTPUT_VARS_DICT["url"],
                       help='alternative url string.')
    # TODO:  Add source type again.  Using url is not a good idea.
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

    args["start_date"] = datetime.strptime(args["start_date"], "%Y%m%d")

    # TODO:  Check for valid input path (maybe even output path)?

    return args


if __name__ == '__main__':

    parser_args = argument_parser()

    if parser_args["local"] is None:
        del parser_args["local"]
        process_navgem(**parser_args)
    else:
        inp = parser_args["input_path"]
        if len(parser_args["local"]) == 0:
            fn = glob.glob(os.path.join(inp, "*.grib"))
        else:
            fn = parser_args["local"]

        out_path = create_full_outpath(parser_args['base_path'],
                                       parser_args["start_date"],
                                       parser_args["run_hour"])
        fn_paths = (os.path.join(inp, x) for x in fn)
        fn_paths = list(fn_paths)
        out_fnames = []
        for forecast in parser_args["forecast_hours"]:
            out_fnames.append(read_one_hour_navgem(fn_paths, out_path, forecast))
        LOG.info(out_fnames)

from __future__ import annotations

import conversions.derived_variables as derive_var
import logging

import numpy as np
import pint
import warnings

from datetime import datetime, timedelta

from conversions.conversion_class import MerraConversion
from netCDF4 import Dataset, num2date
from pyhdf.SD import SD, SDC
from typing import Dict, Generator, KeysView, Tuple, Union


np.seterr(all="ignore")

LOG = logging.getLogger(__name__)

# pint setup
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
# Silence NEP 18 warning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Q_([])


def get_input_data(merra_ds: Dict[str, Union[Dataset, SD]],
                   time_index: Dict[str, int], output_vars_dict) -> Generator[Dict]:
    """Using the mappings from output_vars_dict, map input data to output products."""
    out_vars = dict()

    for out_key, rsk in output_vars_dict.items():
        LOG.info("Get data from %s for %s", rsk["in_file"], rsk["shortname"])
        out_vars["data_object"] = MerraConversion(
            nc_dataset=merra_ds[rsk["in_file"]],
            shortname=rsk["shortname"],
            out_name=out_key,
            out_units=rsk["out_units"],
            ndims_out=rsk["ndims_out"],
            time_ind=time_index[rsk["in_file"]],
            not_masked=True,
            nan_fill=rsk["nan_fill"],
        )
        out_vars["in_file"] = rsk["in_file"]
        out_vars["units_fn"] = rsk["units_fn"]
        if "dependent" in rsk:
            out_vars["dependent"] = dict()
            for support_var_name in rsk["dependent"]:
                sub_field = rsk["dependent"][support_var_name]
                support_obj = MerraConversion(
                    nc_dataset=merra_ds[sub_field["in_file"]],
                    shortname=sub_field["shortname"],
                    out_name=support_var_name,
                    out_units=sub_field["out_units"],
                    ndims_out=sub_field["ndims_out"],
                    time_ind=time_index[sub_field["in_file"]],
                    not_masked=False,   # load these as masked arrays.
                    nan_fill=sub_field["nan_fill"]
                )
                out_vars["dependent"].update({support_var_name: support_obj})

        yield out_vars


def write_output_variables(datasets: Dict[str, Dataset], mask_file: Dataset, out_fields: Generator[Dict]) -> None:
    """Calculate the final output and write to file."""
    if "ana" in datasets.keys():
        file_pressure_levels = datasets["ana"].variables["lev"]
        global_attrs = datasets["ana"]
    else:
        file_pressure_levels = datasets["asm3d"].variables["lev"]
        global_attrs = datasets["asm3d"]

    try:
        source_name = global_attrs.getncattr("Source")
    except KeyError:
        source_name = ""

    pint_unit_levels = Q_(np.asarray(file_pressure_levels), file_pressure_levels.units)

    while True:
        try:
            current_var = next(out_fields)
            out_var = current_var["data_object"]
            file_tag = current_var["in_file"]
            units_fn = current_var["units_fn"]
            out_key = out_var.out_name

            var_fill = out_var.fill
            out_data = out_var.data

            if out_key == "rh":
                temp_k = current_var["dependent"]["masked_temp_k"].data
                out_data = derive_var.qv_to_rh(out_data, temp_k, pint_unit_levels)
                var_fill = out_data.fill_value
                out_data[np.isnan(out_data)] = var_fill
            elif out_key == "rh at sigma=0.995":
                temp_t10m = current_var["dependent"]["masked_temperature_at_sigma"].data
                ps_pa = current_var["dependent"]["surface_pressure_at_sigma"].data
                out_data = derive_var.rh_at_sigma(temp_t10m, ps_pa,
                                                  var_fill, pint_unit_levels, out_data)
                var_fill = out_data.fill_value
                out_data = out_data.filled()
            elif out_key == "water equivalent snow depth":
                out_data = derive_var.hack_snow(out_data, mask_file)
            elif out_key == "land mask":
                out_data = derive_var.merra_land_mask(out_data, mask_file)
            elif out_key == "total ozone":
                out_data = derive_var.total_ozone(out_data, var_fill, pint_unit_levels)
            else:
                out_data = derive_var.apply_conversion(units_fn, out_data, var_fill, p_levels=pint_unit_levels)

            out_var.updateAttr('data', out_data)
            print(f"{source_name}({file_tag})->{out_key}")
            out_var.update_output(datasets, f"{source_name}({file_tag})->{out_key}", out_data, var_fill)

        except StopIteration:
            break


def write_global_attributes(out_sd: SD, info_attrs) -> None:
    """Write Global Attributes.

    :param out_sd: The handle to the output pyhf.SD dataset created.
    :param info_attrs:  Global attributes pulled from input files
        to describe dataset, history and stream (stream if GranuleID is defined)
    :return: None
    """
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

    source_name = info_attrs["Source"]
    history = info_attrs["History"]
    if info_attrs["GranuleID"] is not None:
        stream = info_attrs["GranuleID"].split(".")[0]
        setattr(out_sd, f"{source_name} STREAM",  stream)
    setattr(out_sd, f"{source_name} History",  history)

    for a in [var, lat, lon]:
        a.endaccess()
    out_sd.end()


def get_common_time(datasets: Dict[str, Dataset]):
    """Enforce a 'common' start time among the input datasets."""
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
            print(ds_key, time_hack_offset)
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
            time_set.update({ds_key: analysis_time})
    # find set of time common to all input files
    print(time_set.values())
    ds_common_times = set(time_set.values())

    # if this code has not forced one common time, one of the datasets probably does not match
    ncommon = len(ds_common_times)
    if ncommon != 1:
        print(ds_common_times)

        raise ValueError("Input files have not produced common times: {ncommon}")

    return [dataset_times, ds_common_times]


def get_time_index(file_keys: KeysView[str],
                   file_times: Dict[str, Tuple[int, datetime]],
                   current_time: datetime) -> Dict[str, int]:
    """Determine the time index from the input files based on the output time.

    :rtype: Dict[str, int] time index based on input file.
    """
    # --- determine time index we want from input files
    time_index = dict()
    for file_name_key in file_keys:
        # Get the index for the current timestamp:
        if file_name_key in ["mask", "mask_file"]:
            time_index[file_name_key] = 0
        else:
            time_index[file_name_key] = [i for (i, t) in file_times[file_name_key]
                                         if t == current_time][0]
    return time_index
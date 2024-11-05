import datetime
import functools
import logging
from typing import Dict, Iterator, KeysView, List, Tuple, Union

import numpy as np
from cftime._cftime import real_datetime
from netCDF4 import Dataset, num2date
from pyhdf.SD import SD, SDC

import conversions
import conversions.derived_variables as derive_var
from conversions.conversion_class import MerraConversion

np.seterr(all="ignore")

LOG = logging.getLogger(__name__)

OutVarDictType = Dict[str, Union[MerraConversion, str]]
Complicated_DatasetTimes = Dict[str, List[Tuple[int, real_datetime]]]


def get_input_data(merra_ds: Dict[str, Union[Dataset, SD]],
                   time_index: list, output_vars_dict) -> Iterator[Dict[str, MerraConversion]]:
    """Using the mappings from output_vars_dict, map input data to output products."""
    out_vars = dict()

    for out_key, rsk in output_vars_dict.items():
        LOG.info("Get data from %s for %s", rsk["in_file"], rsk["shortname"])
        filetag = merra_ds[rsk["in_file"]].getncattr("Filename")
        out_vars["data_object"] = MerraConversion(
            data_array=merra_ds[rsk["in_file"]].variables[rsk["shortname"]],
            file_tag=filetag,
            out_name=out_key,
            out_units=rsk["out_units"],
            ndims_out=rsk["ndims_out"],
            unmask=True,
            nan_fill=rsk["nan_fill"],
            time_ind=time_index[rsk["in_file"]]
        )
        if "dependent" in rsk:
            for support_var_name in rsk["dependent"]:
                sub_field = rsk["dependent"][support_var_name]
                var_name = sub_field["shortname"]
                data_array = merra_ds[sub_field["in_file"]].variables[var_name]
                out_vars["data_object"].add_dependent(data_array,
                                                      support_var_name,
                                                      sub_field["nan_fill"],
                                                      time_index[sub_field["in_file"]],
                                                      sub_field["unmask"])

        yield out_vars


def write_output_variables(datasets: Dict[str, Dataset], out_fields: Iterator[Dict], source_name: str) -> None:
    """Calculate the final output and write to file."""
    while True:
        try:
            current_var = next(out_fields)
            out_var = current_var["data_object"]
            out_key = out_var.out_name
            LOG.info(f"Processing {out_key}")

            match out_key:
                case "rh":
                    new_data = derive_var.qv_to_rh(out_var.data, out_var.dependent["masked_temp_k"],
                                                   out_var.dependent["unit_levels"])
                    new_data[np.isnan(new_data)] = new_data.fill_value
                    out_var.updateAttr("fill", new_data.fill_value)
                    out_var.updateAttr("data", new_data.filled())
                    out_var.updateAttr("out_units", "%")
                case "rh at sigma=0.995":
                    temp_t10m = out_var.dependent["masked_temperature_at_sigma"]
                    ps_pa = out_var.dependent["surface_pressure_at_sigma"]
                    new_data = derive_var.rh_at_sigma(temp_t10m, ps_pa,
                                                      out_var.fill, out_var.dependent["unit_levels"], out_var.data)

                    new_data[np.isnan(new_data)] = out_var.fill
                    out_var.updateAttr("data", new_data.filled())
                    out_var.updateAttr("out_units", "%")
                case "water equivalent snow depth":
                    out_var.updateAttr("data", derive_var.hack_snow(out_var.data.magnitude, datasets["mask"]))
                case "land mask":
                    land_mask = derive_var.merra_land_mask(out_var.data, datasets["mask"])
                    # input neither clavrx_fill nor the input data fill make sense for this land mask
                    out_var.updateAttr("fill", None)
                    out_var.updateAttr("data", land_mask.astype(np.float32))
                case "total ozone":
                    new_data = derive_var.total_ozone(out_var.data, out_var.fill, out_var.dependent["unit_levels"])
                    out_var.updateAttr("data", new_data)
                case "total precipitable water":
                    out_var.updateAttr("data", (out_var.data / 10.0))
                case "surface height":
                    out_var.updateAttr("data", conversions.geopotential(out_var.data))
                case _:
                    if out_var.out_units in [1, "degrees_north", "degrees_east"]:
                        # no conversion
                        pass
                    else:
                        print(f"Convert {out_key}[{out_var.get_units}] to {out_var.out_units}")
                        try:
                            out_var.updateAttr("data", out_var.data.to(out_var.out_units))
                        except AttributeError as _e:
                            raise AttributeError(f"Can't convert {out_key}: {out_var.data} to {out_var.out_units}")

            out_var.update_output(datasets, f"{source_name}({out_var.file_tag})->{out_key}")

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

    #source_name = info_attrs["Source"]
    #history = info_attrs["History"]
    #if "GranuleID" in info_attrs.keys():
    #    if info_attrs["GranuleID"] is not None:
    #        stream = info_attrs["GranuleID"].split(".")[0]
    #        setattr(out_sd, f"{source_name} STREAM", stream)
    #setattr(out_sd, f"{source_name} History", history)

    for a in [var, lat, lon]:
        a.endaccess()
    out_sd.end()


def get_common_time(datasets: Dict[str, Dataset], input_type="geosfp"):
    """Enforce a 'common' start time among the input datasets."""
    dataset_times: Complicated_DatasetTimes = dict()
    time_set: Dict[str, real_datetime] = dict()
    keys = list(datasets.keys())
    keys.remove("mask")

    for ds_key in keys:
        dataset_times[ds_key] = []
        time_set[ds_key] = set()
        if "time" in datasets[ds_key].variables.keys():
            t_sds = datasets[ds_key].variables["time"]
            t_units = t_sds.units  # fmt "minutes since %Y-%m-%d %H:%M:%S"
            base_time = datetime.datetime.strptime(
                t_units + " UTC", "minutes since %Y-%m-%d %H:%M:%S %Z"
            )
            # A non-zero time_hack_offset = base_time.minute
            time_hack_offset = base_time.minute
            LOG.debug(ds_key, time_hack_offset)
        else:
            raise ValueError("Couldn't find time coordinate in this file")
        for (i, t) in enumerate(t_sds):
            # format %y doesn't work with gregorian time.
            analysis_time = (num2date(t, t_units, only_use_python_datetimes=True,
                                      only_use_cftime_datetimes=False))
            if analysis_time.minute == time_hack_offset:
                # total hack to deal with non-analysis products being on the half-hour
                analysis_time = analysis_time - \
                                datetime.timedelta(minutes=time_hack_offset)
            # This is just to get the index for a given timestamp later:
            dataset_times[ds_key].append((i, analysis_time))
            time_set[ds_key].add(analysis_time)
    # find set of time common to all input files
    print(time_set.values())
    ds_common_times = functools.reduce(lambda x, y: x & y, time_set.values())

    # if this code has not forced one common time, one of the datasets probably does not match
    ncommon = len(ds_common_times)
    if input_type == "geosfp":
        if ncommon != 1:
            print(ds_common_times)
            raise ValueError("Input files have not produced common times: {ncommon}")
        else:
            # return datetime, not single set value
            common_time = ds_common_times.pop()
    elif input_type == "merra":
        if ncommon != 4:
            print(ds_common_times)
            raise ValueError("Input files have not produced 4 common times: {ncommon}")
        else:
            common_time = ds_common_times

    return [dataset_times, common_time]


def get_time_index(file_keys: KeysView[str],
                   file_times: Complicated_DatasetTimes,
                   current_time: datetime.datetime) -> Dict[str, int]:
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
            time_index[file_name_key] = [index for (index, time_at_index) in file_times[file_name_key]
                                         if time_at_index == current_time][0]
    return time_index

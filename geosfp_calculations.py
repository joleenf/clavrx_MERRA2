from __future__ import annotations

import logging
import os
import sys
import tempfile
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import pint
import yaml

try:
    from datetime import datetime, timedelta
    from typing import Callable, Dict, Generator, KeysView, List, Tuple, Union

    import numpy as np
    from netCDF4 import Dataset, num2date
    from pandas import date_range
    from pint import UnitRegistry
    from pyhdf.SD import SD, SDC

    from conversion_class import CommandLineMapping, ReanalysisConversion
except ImportError as e:
    print("Import Error {}".format(e))
    print("Type:  conda activate merra2_clavrx")
    sys.exit(1)

np.seterr(all="ignore")

LOG = logging.getLogger(__name__)


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


def qv_to_rh(specific_humidity, temp_k, levels: pint.Quantity, press_at_sfc=None):
    """Convert Specific Humidity [kg/kg] -> relative humidity [%]."""
    # See Petty Atmos. Thermo. 4.41 (p. 65), 8.1 (p. 140), 8.18 (p. 147)
    ureg = UnitRegistry()
    levels = np.asarray(levels.to(ureg.pascal))   # [hPa] -> [Pa] when necessary.

    es_tot = total_saturation_pressure(temp_k)

    vapor_pressure = vapor_pressure_approximation(specific_humidity, press_at_sfc, levels)

    relative_humidity = vapor_pressure / es_tot * 100.0  # relative humidity [%]
    relative_humidity[relative_humidity > 100.0] = 100.0  # clamp to 100% to mimic CFSR

    return relative_humidity


def rh_at_sigma(temp10m, sfc_pressure, sfc_pressure_fill, levels: pint.Quantity, data):
    """Calculate the rh at sigma using 10m fields."""
    temp_k = temp10m  # temperature in [K] (Y, X) not in (time, Y, X)

    # pressure in [Pa]
    sfc_pressure[sfc_pressure == sfc_pressure_fill] = np.nan

    rh_sigma = qv_to_rh(data, temp_k, levels, press_at_sfc=sfc_pressure)
    rh_sigma.set_fill_value = sfc_pressure_fill

    return rh_sigma


def dobson_layer(mmr_data_array, level_index, pressure_levels: pint.Quantity):
    """Calculate a dobson layer from a 3D mmr data array given a level index.

    :param pressure_levels: 1D pint Quantity array of pressure levels in hPa from data file.
    :param mmr_data_array: Mass mixing ratio data array
    :param level_index: index of current level
    :return: dobson layer
    """
    ureg = UnitRegistry()
    pressure_levels = np.asarray(pressure_levels.to(ureg.hectopascal))  # ensure hPa

    dobson_unit_conversion = 2.69e16  # 1 DU = 2.69e16 molecules cm-2
    gravity = 9.8  # m/s^2
    avogadro_const = 6.02e23  # molecules/mol
    o3_molecular_weight = 0.048  # kg/mol
    dry_air_molecular_weight = 0.028966  # kg/mol molecular weight of dry air.

    const = 0.01 * avogadro_const / (gravity * dry_air_molecular_weight)
    vmr_j = mmr_data_array[level_index] * dry_air_molecular_weight / o3_molecular_weight
    vmr_j_1 = mmr_data_array[level_index - 1] * dry_air_molecular_weight / o3_molecular_weight
    ppmv = 0.5 * (vmr_j + vmr_j_1)
    delta_pressure = pressure_levels[level_index - 1] - pressure_levels[level_index]
    o3_dobs_layer = ppmv * delta_pressure * const / dobson_unit_conversion

    return o3_dobs_layer


def total_ozone(mmr, fill_value, pressure_levels: pint.Quantity):
    """Calculate total column ozone in dobson units (DU), from ozone mixing ratio in [kg/kg]."""
    nlevels = mmr.shape[0]
    dobson = np.zeros(mmr[0].shape).astype("float32")
    for j in range(1, nlevels):
        good_j = np.logical_not(np.isnan(mmr[j]))
        good_j_1 = np.logical_not(np.isnan(mmr[j - 1]))
        good = good_j & good_j_1
        dobs_layer = dobson_layer(mmr, j, pressure_levels)
        dobson[good] = dobson[good] + dobs_layer[good]

    dobson[np.isnan(dobson)] = fill_value

    return dobson


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


def apply_conversion(scale_func: Callable, data: np.ndarray, fill, p_levels=None) -> np.ndarray:
    """Apply fill to converted data after function."""
    converted = data.copy()

    if scale_func == "total_ozone":
        converted = total_ozone(data, fill, p_levels)
    else:
        converted = scale_func(converted)

        if fill is not None:
            converted[data == fill] = fill

    return converted


def get_common_time(datasets: Dict[str, Dataset]):
    """Enforce a 'common' start time among the input datasets."""
    dataset_times = dict()
    time_set = dict()
    # drop mask_file from this analysis
    if "mask_file" in datasets.keys():
        datasets.pop("mask_file")
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
    if len(ds_common_times) != 1:
        print(ds_common_times)
        raise ValueError("Input files have not produced common times: {len(ds_common_times)}")

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


def get_input_data(merra_ds: Dict[str, Union[Dataset, SD]],
        time_index: Dict[str, int], output_vars_dict) -> Generator[Dict]:
    """Prepare input variables."""
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


def write_output_variables(datasets: Dict[str, Dataset], out_fields: Generator[Dict]) -> None:
    """Calculate the final output and write to file."""
    ureg = UnitRegistry()

    if "ana" in datasets.keys():
        file_pressure_levels = datasets["ana"].variables["lev"]
    else:
        file_pressure_levels = datasets["asm3d"].variables["lev"]

    pint_unit_levels = np.asarray(file_pressure_levels) * ureg(file_pressure_levels.units)

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
                out_data = qv_to_rh(out_data, temp_k, pint_unit_levels)
                var_fill = out_data.fill_value
                out_data[np.isnan(out_data)] = var_fill
            elif out_key == "rh at sigma=0.995":
                temp_t10m = current_var["dependent"]["masked_temperature_at_sigma"].data
                ps_pa = current_var["dependent"]["surface_pressure_at_sigma"].data
                out_data = rh_at_sigma(temp_t10m, ps_pa,
                                       var_fill, pint_unit_levels, out_data)
                var_fill = out_data.fill_value
                out_data = out_data.filled()
            elif out_key == "water equivalent snow depth":
                out_data = _hack_snow(out_data, datasets["mask"])
            elif out_key == "land mask":
                out_data = _merra_land_mask(out_data, datasets["mask"])
            else:
                out_data = apply_conversion(units_fn, out_data, var_fill, p_levels=pint_unit_levels)
            out_var.update_output(datasets, "MERRA2->{}".format(file_tag), out_data, var_fill)

        except StopIteration:
            break


def write_global_attributes(out_sd: SD, info_nc: Dataset) -> None:
    """Write Global Attributes.

    :param out_sd: The handle to the output pyhf.SD dataset created.
    :param info_nc: The "ana" netCDF.Dataset for stream and history
                    information.
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
    setattr(out_sd,
            "MERRA STREAM",
            "{}".format(info_nc.GranuleID.split(".")[0]))
    setattr(out_sd, "MERRA History",
            "{}".format(info_nc.History))
    for a in [var, lat, lon]:
        a.endaccess()
    out_sd.end()


def make_merra_one_day(run_dt: datetime, input_path: Path, out_dir: Path) -> List[str]:
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

        time_inds = get_time_index(in_files.keys(), times, out_time)

        # --- prepare input data variables
        # TODO:  If this will be used, output_vars_dict needs to be added from yaml.
        out_vars = get_input_data(merra_sd, time_inds)
        write_output_variables(merra_sd, out_vars)

        write_global_attributes(merra_sd["out"], merra_sd["ana"])

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
        sh_cmd = (" ".join(cmd))

        raise FileNotFoundError("Download with command: {}.".format(sh_cmd))
        # try:
        #     proc = subprocess.run(cmd, text=True, check=True)
        #     sh_cmd = (" ".join(proc.args))
        # except subprocess.CalledProcessError as proc_error_noted:
        #     raise subprocess.CalledProcessError from proc_error_noted

        # file_list = list(inpath.glob(file_glob))
        #
        # if len(file_list) == 0:
        #     raise FileNotFoundError("{}.".format(sh_cmd))

    return file_list[0]

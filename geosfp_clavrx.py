""" TODO module doc """
import fnmatch
import logging
import numpy as np
import os
import sys
import yaml

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime, timedelta
from glob import glob
from netCDF4 import Dataset, num2date
from pyhdf.SD import SD, SDC
from typing import Callable

from geosfp_calculations import get_input_data, get_time_index, get_common_time
from geosfp_calculations import write_output_variables, write_global_attributes

np.seterr(all='ignore')

comp_level = 6  # 6 is the gzip default; 9 is best/slowest/smallest file

no_conversion = lambda a: a  # ugh why doesn't python have a no-op function...
fill_bad = lambda a: a*np.nan

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT_DIR, 'yamls', 'geosfp.yaml'), "r") as yml:
    OUTPUT_VARS_DICT = yaml.load(yml, Loader=yaml.Loader)

levels_listings = OUTPUT_VARS_DICT.pop("defined levels")
LEVELS = levels_listings["hPa_levels"]


def construct_filepath(base_data_dir, base_out_dir, input_datetime):
    """Construct the filepath given a datetime containing the date and the synoptic hour."""
    year_str = input_datetime.strftime("%Y")
    month_str = input_datetime.strftime("%m")
    day_str = input_datetime.strftime("%d")
    date_str = input_datetime.strftime("%Y_%m_%d")
    date_str_arg = input_datetime.strftime("%Y%m%d")
    inpath = os.path.join(base_data_dir, year_str, input_datetime.strftime("%m_%d_%H"))
    outpath_full = os.path.join(base_out_dir, year_str) + '/'

    if not os.path.isdir(inpath):
        raise FileNotFoundError(f"{inpath} does not exist.")

    if not os.path.isdir(outpath_full):
        os.makedirs(outpath_full)

    return inpath, outpath_full


def get_input_files(filepath, input_datetime):
    """Find the input filse and sort within the filepath."""

    file_start=input_datetime.strftime("%Y%m%d_%H%M")
    time_window_end=(input_datetime + timedelta(minutes=30)).strftime("%Y%m%d_%H%M")

    found_files = os.listdir(filepath)
    mask_file = os.path.join(filepath, f"GEOS.fp.asm.const_2d_asm_Nx.00000000_0000.V01.nc4")
    if not os.path.isfile(mask_file):
        raise FileNotFoundError(mask_file)
    in_files = {"mask_file": mask_file}


    print(f"Processing date: {file_start}")

    fcst="GEOS.fp.fcst"
    version="V01"
    for key in ["flx", "slv", "lnd", "rad"]:
        search_pattern = f"{fcst}.tavg1_2d_{key}_Nx.{file_start[:-2]}+{time_window_end}.{version}.nc4"
        try:
            result = fnmatch.filter(found_files, search_pattern)[0]
        except IndexError as e:
            raise FileNotFoundError(f"{search_pattern} not found at {filepath}")

        in_files.update({key: os.path.join(filepath, result)})


    search_pattern = f"{fcst}.inst3_3d_asm_Np.{file_start[:-2]}+{file_start}.{version}.nc4"
    try:
        result = fnmatch.filter(found_files, search_pattern)[0]
    except IndexError as e:
        raise FileNotFoundError(f"{search_pattern} not found at {filepath}")
    in_files.update({"asm3d": os.path.join(filepath, result)})

    return in_files


def make_one_hour(in_files, out_dir):
    """Read input, parse times, and run conversion on one day at a time."""
    merra_sd = dict()
    for file_name_key, file_name in in_files.items():
        merra_sd[file_name_key] = Dataset(file_name)

    times, common_times = get_common_time(merra_sd)

    out_fnames = []
    for out_time in sorted(common_times):
        #LOG.info("    working on time: %s", out_time)
        out_fname = os.path.join(out_dir, f"{out_time.strftime('geos5.%y%m%d%H_F000.hdf')}")
        out_fnames.append(out_fname)
        merra_sd["out"] = SD(
            out_fname, SDC.WRITE | SDC.CREATE | SDC.TRUNC
        )  # TRUNC will clobber existing

        time_inds = get_time_index(in_files.keys(), times, out_time)

        # --- prepare input data variables
        out_vars = get_input_data(merra_sd, time_inds, OUTPUT_VARS_DICT)
        write_output_variables(merra_sd, out_vars)

        write_global_attributes(merra_sd["out"], merra_sd["ana"])

    return out_fnames


def main(scratch, outpath, in_date, synoptic_run):
    """Construct filepaths, collections and run one run."""
    date_parsed = datetime.strptime(f"{in_date}_{synoptic_run}", '%Y%m%d_%H')
    print(date_parsed)
    inpath, outpath_full = construct_filepath(scratch, outpath, date_parsed)
    in_files = get_input_files(inpath, date_parsed)

    out_files = make_one_hour(in_files, outpath_full)
    print('out_files: {}'.format(list(map(os.path.basename, out_files))))


if __name__ == '__main__':
    scratch = "/ships22/cloud/scratch/7day/GEOS-FP_INPUT/"
    outpath = "/ships22/cloud/Ancil_Data/clavrx_ancil_data/dynamic/geos5"

    try:
        in_date = sys.argv[1]
        synoptic_run = sys.argv[2]
    except IndexError as e:
        print(f"usage:\n    python {sys.argv[0]} 20090101 00")
        print(f"Enter the date of the run and the synoptic run time.")
        exit()

    main(scratch, outpath, in_date, synoptic_run)

"""Load GEOSFP data and pre-process for CLAVRx."""
import datetime
import fnmatch
import logging
import os
import sys

import numpy as np
import yaml

import conversions.file_tools as frw

try:
    from netCDF4 import Dataset
    from pyhdf.SD import SD, SDC
except ImportError as e:
    print("Import Error {}".format(e))
    print("Type:  mamba activate merra2_clavrx")
    sys.exit(1)

FORMAT = '%(asctime)s %(clientip)-15s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
LOG = logging.getLogger("__name__")
LOG.setLevel(logging.INFO)

np.seterr(all='ignore')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT_DIR, 'yamls', 'geosfp.yaml'), "r") as yml:
    OUTPUT_VARS_DICT = yaml.safe_load(yml)


def construct_filepath(base_data_dir, base_out_dir, input_datetime):
    """Construct the filepath given a datetime containing the date and the synoptic hour."""
    year_str = input_datetime.strftime("%Y")
    in_path = os.path.join(base_data_dir, year_str, input_datetime.strftime("%Y_%m_%d_%H"))
    destination = os.path.join(base_out_dir, "geos", year_str)

    if not os.path.isdir(in_path):
        raise FileNotFoundError(f"{in_path} does not exist.")

    if not os.path.isdir(destination):
        os.makedirs(destination)

    return in_path, destination


def get_input_files(filepath, input_datetime):
    """Find the input files and sort within the filepath."""

    file_start = input_datetime.strftime("%Y%m%d_%H%M")
    time_window_end = (input_datetime + datetime.timedelta(minutes=30)).strftime("%Y%m%d_%H%M")

    found_files = os.listdir(filepath)
    mask_file = os.path.join(filepath, "GEOS.fp.asm.const_2d_asm_Nx.00000000_0000.V01.nc4")
    if not os.path.isfile(mask_file):
        raise FileNotFoundError(mask_file)
    in_files = {"mask": mask_file}

    print(f"Processing date: {file_start}")

    fcst = "GEOS.fp.fcst"
    version = "V01"
    for key in ["flx", "slv", "lnd", "rad"]:
        search_pattern = f"{fcst}.tavg1_2d_{key}_Nx.{file_start[:-2]}+{time_window_end}.{version}.nc4"
        try:
            result = fnmatch.filter(found_files, search_pattern)[0]
        except IndexError as _e:
            raise FileNotFoundError(f"{search_pattern} not found at {filepath}")

        in_files.update({key: os.path.join(filepath, result)})

    search_pattern = f"{fcst}.inst3_3d_asm_Np.{file_start[:-2]}+{file_start}.{version}.nc4"
    try:
        result = fnmatch.filter(found_files, search_pattern)[0]
    except IndexError as _e:
        raise FileNotFoundError(f"{search_pattern} not found at {filepath}")
    in_files.update({"asm3d": os.path.join(filepath, result)})

    return in_files


def make_one_hour(in_files, out_dir):
    """Read input, verify times, run conversion on one synoptic run."""
    merra_sd = dict()

    for file_name_key, file_name in in_files.items():
        merra_sd[file_name_key] = Dataset(file_name)

    # just use any dataset key to get ncattrs
    any_key = list(merra_sd.keys())[0]
    file_global_attrs = {"History": None, "Source": None, "GranuleID": None, "Filename": None}
    for attribute_key in file_global_attrs.keys():
        try:
            file_global_attrs.update({attribute_key: merra_sd[any_key].getncattr(attribute_key)})
        except AttributeError:
            pass

    times, out_time = frw.get_common_time(merra_sd)
    # keep this for MERRA2 which operates on 4 time periods at a time.
    time_inds = frw.get_time_index(in_files.keys(), times, out_time)
    model_type = file_global_attrs["Filename"].split(".")[0]

    fn = f"geosfp.{out_time.strftime('%y%m%d%H_F000.hdf')}"
    out_fn = os.path.join(out_dir, fn)

    merra_sd["out"] = SD(
        out_fn, SDC.WRITE | SDC.CREATE | SDC.TRUNC
    )  # TRUNC will clobber existing
    # --- prepare input data variables
    out_vars = frw.get_input_data(merra_sd, time_inds, OUTPUT_VARS_DICT)
    frw.write_output_variables(merra_sd, out_vars, model_type)

    frw.write_global_attributes(merra_sd["out"], file_global_attrs)

    return fn


def main(in_data_dir, final_dir, data_dt: str, run_hour: str):
    """Construct filepaths, collections and run one run."""
    date_parsed = datetime.datetime.strptime(f"{data_dt}_{run_hour}", '%Y%m%d_%H')
    in_data_dir, out_path_full = construct_filepath(in_data_dir, final_dir, date_parsed)
    in_files = get_input_files(in_data_dir, date_parsed)

    out_file = make_one_hour(in_files, out_path_full)
    print(f"out_file: {out_path_full}/{out_file}")


if __name__ == '__main__':
    # Location of Dynamic ancil data (or where the final files will be stored)
    # Set up as an environment variable (DYNAMIC_ANCIL)
    output_loc = os.environ["DYNAMIC_ANCIL"]
    scratch = "/ships22/cloud/scratch/7day/GEOS-FP_INPUT/"

    try:
        in_date = sys.argv[1]
        synoptic_run = sys.argv[2]
    except IndexError as _e:
        print(f"usage:\n    python {sys.argv[0]} 20090101 00")
        print("Enter the date of the run and the synoptic run time.")
        exit()

    main(scratch, output_loc, in_date, synoptic_run)

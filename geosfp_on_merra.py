"""Load MERRA2 data and pre-process for CLAVRx using updated code scheme."""
import datetime
import logging
import os
import sys
from glob import glob as glob

import numpy as np
import yaml

import conversions.file_tools as frw

try:
    from netCDF4 import Dataset
    from pyhdf.SD import SD, SDC
except ImportError as e:
    print("Import Error {}".format(e))
    print("Type:  conda activate merra2_clavrx")
    sys.exit(1)

FORMAT = '%(asctime)s %(clientip)-15s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
LOG = logging.getLogger("__name__")
LOG.setLevel(logging.INFO)

np.seterr(all='ignore')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT_DIR, 'yamls', 'MERRA2_vars.yaml'), "r") as yml:
    OUTPUT_VARS_DICT = yaml.safe_load(yml)

levels_listings = OUTPUT_VARS_DICT.pop("defined levels")
LEVELS = levels_listings["hPa_levels"]


def construct_filepath(base_data_dir, base_out_dir, input_datetime):
    """Construct the filepath given a datetime containing the date and the synoptic hour."""
    year_str = input_datetime.strftime("%Y")
    in_path = os.path.join(base_data_dir, year_str, input_datetime.strftime("%m_%d_%H"))
    destination = os.path.join(base_out_dir, year_str) + '/'

    if not os.path.isdir(in_path):
        raise FileNotFoundError(f"{in_path} does not exist.")

    if not os.path.isdir(destination):
        os.makedirs(destination)

    return in_path, destination


def make_merra_all_hours(in_files, out_dir):
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

    times, out_times = frw.get_common_time(merra_sd, input_type="merra")
    # keep this for MERRA2 which operates on 4 time periods at a time.
    #
    for out_time in out_times:
    #for out_time in {datetime.datetime(2024, 2, 10, 0, 0)}:
        print(out_time)
        model_type = file_global_attrs["Filename"].split(".")[0]

        fn = f"{model_type}.{out_time.strftime('%y%m%d%H_F000.hdf')}"
        out_fn = os.path.join(out_dir, fn)

        merra_sd["out"] = SD(
            out_fn, SDC.WRITE | SDC.CREATE | SDC.TRUNC
        )  # TRUNC will clobber existing
        # --- prepare input data variables
        time_inds = frw.get_time_index(in_files.keys(), times, out_time)
        print(out_time)
        out_vars = frw.get_input_data(merra_sd, time_inds, OUTPUT_VARS_DICT)
        frw.write_output_variables(merra_sd, out_vars)

        frw.write_global_attributes(merra_sd["out"], file_global_attrs)

    return fn


def main_merra(scratch:str, outpath: str, date_dt: datetime.datetime):
    """Construct filepaths, collections and run one run."""
    year_str = date_dt.strftime("%Y")
    date_str = date_dt.strftime("%Y_%m_%d")
    outpath_full = os.path.join(outpath, year_str) + '/'
    scratch = os.path.join(scratch, year_str, date_str)

    try:
        os.makedirs(outpath_full)
    except OSError:
        pass  # dir already exists
    # BTH: Define mask_file here
    mask_file = os.path.join(scratch, "MERRA2_101.const_2d_ctm_Nx.00000000.nc4")
    print("looking at {}".format(mask_file))
    if not os.path.isfile(mask_file):
        mask_file = os.path.join(scratch, f"MERRA2_101.const_2d_ctm_Nx.{date_str_arg}.nc4")
    if not os.path.isfile(mask_file):
        raise FileNotFoundError(mask_file)

    in_files = {"mask": mask_file}

    in_files = {
        'ana': glob(f"{scratch}/MERRA2*ana_Np.{date_str_arg}.nc4")[0],
        'flx': glob(f"{scratch}/MERRA2*flx_Nx.{date_str_arg}.nc4")[0],
        'slv': glob(f"{scratch}/MERRA2*slv_Nx.{date_str_arg}.nc4")[0],
        'lnd': glob(f"{scratch}/MERRA2*lnd_Nx.{date_str_arg}.nc4")[0],
        'asm3d': glob(f"{scratch}/MERRA2*asm_Np.{date_str_arg}.nc4")[0],
        'asm2d': glob(f"{scratch}/MERRA2*asm_Nx.{date_str_arg}.nc4")[0],
        'rad': glob(f"{scratch}/MERRA2*rad_Nx.{date_str_arg}.nc4")[0],
    }
    in_files.update({"mask": mask_file})
    print(in_files)

    out_file = make_merra_all_hours(in_files, outpath_full)
    print(f"out_files: {out_file}")


if __name__ == '__main__':
        home = os.path.expanduser("~")
        scratch = os.path.join(home, "data", "merra_input")
        outpath = os.path.join(home, "data", "merra_output")

        try:
            date_str_arg = sys.argv[1]
            date_parsed = datetime.datetime.strptime(date_str_arg, '%Y%m%d')
        except IndexError as _e:
            print('usage:\n    python merra4clavrx.py 20090101')
            exit()

        main_merra(scratch, outpath, date_parsed)

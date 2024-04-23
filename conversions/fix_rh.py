import datetime
import os
from glob import glob as glob

import numpy as np
import yaml
import sys
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC

import conversions.derived_variables as derive_var
import conversions.file_tools as frw

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT_DIR, 'yamls', 'MERRA2_vars.yaml'), "r") as yml:
    OUTPUT_VARS_DICT = yaml.safe_load(yml)

RH_DICT = dict()
RH_DICT["rh"] = OUTPUT_VARS_DICT.pop("rh")


def recalculate_rh(datasets, out_vars):
    """Recalculate the rh field."""
    old_rh = datasets["out"].select("rh")
    source_data = getattr(old_rh, "source_data")

    try:
        current_var = next(out_vars)
        out_var = current_var["data_object"]

        # Recalculate RH
        new_data = derive_var.qv_to_rh(out_var.data, out_var.dependent["masked_temp_k"],
                                       out_var.dependent["unit_levels"])
        new_data[np.isnan(new_data)] = new_data.fill_value
        out_var.updateAttr("fill", new_data.fill_value)
        out_var.updateAttr("data", new_data.filled())
        out_var.updateAttr("out_units", "%")

        out_var.update_output(datasets, source_data, create=False)

    except StopIteration:
        print("Finished processing variables.")

    return datasets


def read_input_file(ana_file, merra_hdf_dir):
    """Read input, verify times, run conversion on one synoptic run."""
    merra_sd = dict()
    merra_sd["ana"] = Dataset(ana_file)
    merra_sd["mask"] = None

    # just use any dataset key to get ncattrs
    times, out_times = frw.get_common_time(merra_sd, input_type="merra")
    # keep this for MERRA2 which operates on 4 time periods at a time.
    #

    for out_time in out_times:
    #for out_time in {datetime.datetime(2024, 2, 10, 0, 0)}:
        hh = out_time.strftime("%H")
        print(out_time)
        time_inds = frw.get_time_index(merra_sd.keys(), times, out_time)

        out_vars = frw.get_input_data(merra_sd, time_inds, RH_DICT)

        merra_file = os.path.join(merra_hdf_dir, f"merra.{yymmdd}{hh}_F000.hdf")
        if not os.path.exists(merra_file):
            raise FileNotFoundError(f"{merra_file}")
        merra_sd["out"] = SD(
            merra_file, SDC.WRITE | SDC.CREATE)

        merra_sd = recalculate_rh(merra_sd, out_vars)
        merra_sd["out"].end()
        merra_sd.pop("out")


if __name__ == "__main__":
    date_str_arg = datetime.datetime.strptime(sys.argv[1], "%Y%m%d")
    year_str = date_str_arg.strftime("%Y")
    date_str = date_str_arg.strftime("%Y_%m_%d")
    yymmdd = date_str_arg.strftime("%y%m%d")
    yyyymmdd = date_str_arg.strftime("%Y%m%d")
  
    home = os.path.dirname(os.path.expanduser("~"))

    if home == "/home":
        scratch = "/ships22/cloud/Ancil_Data/clavrx_ancil_data/dynamic/MERRA_INPUT/tmp/"
        scratch = os.path.join(scratch, year_str, date_str)
        merra_dir = f"/ships22/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2"
        merra_dir = os.path.join(merra_dir, year_str)
    else:
        scratch = os.path.join(home, "data", "merra_input")
        merra_dir = os.path.join(home, "data", "merra_output", year_str)

    print(f"Looking in {scratch}")

    input_file = glob(f"{scratch}/MERRA2*ana_Np.{yyyymmdd}.nc4")[0]
    read_input_file(input_file, merra_dir)

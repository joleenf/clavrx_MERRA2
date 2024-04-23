import argparse
import os

import datetime
import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC
from pyhdf.error import HDF4Error
import sys

#home = os.path.expanduser("~")
#ddir = os.path.join(home, "data", "merra_output", "2024", "old")
base_dir = "/ships22/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2"

CLAVRX_FILL = 9.999e20


def read_files(fullpath, exit_early=False, print_rh_range=False):
    """Load the hdf and read the files."""
    try:
        d = SD(fullpath, SDC.READ)
    except HDF4Error as e:
        msg=f"{e} for {fullpath}"
        raise HDF4Error(msg)

    if "rh" in d.datasets():
        rh = d.select("rh")
    else:
        print(f"ERROR:  rh not found in {fullpath}")

    pl = d.select("pressure levels")

    data = dict()
    if exit_early:
        # if exiting early, this is being called 
        # to determine if a fix has occured so the calling
        # script in bash wants an exit status
        out_of_range = 0
    else:
        out_of_range=[]

    if print_rh_range:
        print(fullpath)
    for level_index in range(rh.dimensions()["level"]-1, -1, -1):
        a = rh[:, :, level_index]
        a[a == rh.attributes("fill_value")] = np.nan
        if print_rh_range:
            print(pl[level_index], np.min(a), np.max(a))
        if np.max(a) > 100.0:
            data["filename"] = merra_file
            data["max"] = np.max(a)
            if exit_early:
                # if exiting early, this is being called 
                # to determine if a fix has occured so the calling
                # script in bash wants an exit status
                print(f"{merra_file} has not been repaired")
                out_of_range = 1
            else:
                out_of_range.append(data)
            # Can jump out, found a problem case.
            break
    rh.endaccess()
    d.end()

    return out_of_range


def run_years(current_dir):
    """Run out_of_range check over multiple years/files."""
    for year in range(2024, 2023, -1):
        out_of_range = list()
        print(f"Running for {year}")
        current_dir = os.path.join(home, f"{year}")
        for merra_file in os.listdir(current_dir):
            fullpath = os.path.join(current_dir, merra_file)
            out_of_range = read_files(fullpath)

        df = pd.DataFrame(out_of_range)
        df.to_csv(f"/data/Personal/clavrx_ops/MERRA_OUTPUT/badrh{year}.csv")

def create_parser():

    parser = argparse.ArgumentParser(usage='\n\npython %(prog)s <arguments>',
                                     description="Determine if merra rh has reasonable range.")
    parser.add_argument('--synoptic_run', type=str, help='Enter Synoptic Run of Merra File', 
                         choices=["00", "06", "12", "18"], default="00")
    parser.add_argument('--merra_dir', type=str, help='Enter location of merra hdf files',
                        default=base_dir)
    parser.add_argument('--dt', action='store',
                        type=lambda s: datetime.datetime.strptime(s, "%Y%m%d"), required=True,
                        help="date to process in YYYYmmdd format")
    parser.add_argument('--rh_range', help="Print range of rh field.", action=argparse.BooleanOptionalAction)


    args = vars(parser.parse_args())

    return args


if __name__ == "__main__":
    args = create_parser()
    merra_dir = args["merra_dir"]
    date_processed = args["dt"]
    synoptic_run = args["synoptic_run"]
    rh_range = args["rh_range"]

    year = date_processed.strftime("%Y")
    merra_date = date_processed.strftime("%y%m%d")

    merra_file = os.path.join(merra_dir, year, f"merra.{merra_date}{synoptic_run}_F000.hdf")
    res=read_files(merra_file, print_rh_range=rh_range)

    if isinstance(res, int):
        sys.exit(res)

import os

import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC

#home = os.path.expanduser("~")
#ddir = os.path.join(home, "data", "merra_output", "2024", "old")
home = "/ships22/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2"

CLAVRX_FILL = 9.999e20


def read_files(fullpath):
    """Load the hdf and read the files."""
    d = SD(fullpath, SDC.READ | SDC.WRITE)

    if "rh" in d.datasets():
        rh = d.select("rh")
    else:
        print(f"ERROR:  rh not found in {fullpath}")

    pl = d.select("pressure levels")

    data = dict()
    out_of_range=[]
    for level_index in range(rh.dimensions()["level"]-1, -1, -1):
        a = rh[:, :, level_index]
        a[a == rh.attributes("fill_value")] = np.nan
        print(pl[level_index], np.min(a), np.max(a))
        if np.max(a) > 100.0:
            data["filename"] = merra_file
            data["max"] = np.max(a)
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


if __name__ == "__main__":
    merra_dir = "/Users/joleenf/data/merra_output/2024"
    merra_file = os.path.join(merra_dir, "MERRA2_400.24020100_F000.hdf")
    read_files(merra_file)

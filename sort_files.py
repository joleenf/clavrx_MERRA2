import logging
import numpy as np
import os
from glob import glob as glob
from datetime import datetime as dt
from navgem_nomads_get import concat_gribs_to_many

LOG = logging.getLogger(__name__)

data_date = dt(2022,10,11)
dateYMD = data_date.strftime("%Y%m%d")
year = data_date.strftime("%Y")
date_delim = data_date.strftime("%Y_%m_%d")
data_dir = f"/data/Personal/joleenf/navgem/{year}/{date_delim}/nrl_orig"
modelrun = f"{dateYMD}06"
file_endings=[]
start_crc=120

fns = glob(os.path.join(data_dir, "US*"))
for fn in fns:
    file_endings.append(fn[start_crc:])
unique_tags = np.unique(file_endings)
print(unique_tags)

for ue in unique_tags:
    if not ue.isspace() and ue != "":
        grib_name = concat_gribs_to_many(data_dir, modelrun, ue)
        found = glob(grib_name)
        if len(found) == 1:
            pass
        elif len(found) > 1:
            LOG.info("oops, multiple files found: {}".format(' '.join(map(str, found))))
        else:
            LOG.error(f"No files match {ue}")

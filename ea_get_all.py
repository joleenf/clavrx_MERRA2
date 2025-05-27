import argparse
import earthaccess
import sys
import warnings

from datetime import datetime as datetime
from pathlib import Path

# This will work if Earthdata prerequisite files have already been generated
auth = earthaccess.login()

DOI_DICT = {"inst6_3d_ana_Np": "10.5067/A7S6XP56VZWS", 
            "inst6_3d_ana_Nv": "10.5067/IUUF4WB9FT4W",
            "tavg1_2d_slv_Nx": "10.5067/VJAFPLI1CSIV",
            "tavg1_2d_flx_Nx": "10.5067/7MCPBJ41Y0K6",
            "inst3_3d_asm_Np": "10.5067/QBZ6MG944HW0",
            "const_2d_ctm_Nx": "10.5067/ME5QX6Q5IGGU",
            "inst1_2d_asm_Nx": "10.5067/3Z173KIE2TPD",
            "tavg1_2d_lnd_Nx": "10.5067/RKPHT8KC1Y1T",
            "tavg1_2d_rad_Nx": "10.5067/Q9QMY5PBNV1T"
            }



def chomp_args(file_choices):
    parser = argparse.ArgumentParser(usage="\n python %(prog)s -h",
            description="\n Download collection by date and doi. Find doi \
            at https://gmao.gsfc.nasa.gov/reanalysis/merra-2/citing_MERRA-2",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("start_day",
                        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
                        help="First Day in Range in CCYY-MM-DD")
    parser.add_argument("--end_day",
                        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
                        help="Last Day in Range CCYY-MM-DD, \
                              defaults to start_day",
                        default=None)
    parser.add_argument("--collection", nargs="+", choices=file_choices, default="all")
    parser.add_argument("--local_path", "-w", type=Path, default=Path.home(),
                        help="local download path")
    args = parser.parse_args()

    (args.local_path).mkdir(parents=True, exist_ok=True)
    if args.end_day is None:
        args.end_day = args.start_day

    return args


def get_data(start_day, end_day, doi, local_path):
    """Download collection by date based on start_day and end_day
       given in CCYY-MM-DD."""
    # To download multiple files, change the second temporal parameter
    if doi == "10.5067/ME5QX6Q5IGGU":
        # constants file has a fixed date
        full_path = local_path.joinpath(f"MERRA2_101.const_2d_ctm_Nx.{start_day.strftime('%Y%m%d')}.nc4")
        start_day = end_day = "1980-01-01"
    results = earthaccess.search_data(
        doi=doi,
        temporal=(start_day, end_day), # This will stream one granule, but can be edited for a longer temporal extent
        bounding_box=(-180, -90, 180, 90)
    )

    if len(results) == 0:
        warnings.warn(f"No results found for this search {doi}!!!", UserWarning)
    print(local_path)
    print(results)
    downloaded_files = earthaccess.download(
        results,
        local_path=local_path, # Change this string to download to a different path
    )


if __name__ == "__main__":
    choices = [ key for key in DOI_DICT.keys()]
    choices.append("all")

    args = chomp_args(choices)
    if args.collection == "all":
        for value in DOI_DICT.values():
            get_data(args.start_day, args.end_day,
                     doi=value, local_path=args.local_path)
    else:
        for key in args.collection:
            print(f"Trying to get {key}")
            doi = DOI_DICT[key]
            get_data(args.start_day, args.end_day,
                     doi=doi, local_path=args.local_path)

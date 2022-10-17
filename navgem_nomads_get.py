"""Using Beautiful Soup, download data for navgem."""
import logging
import os
import re
import shlex
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime as datetime
from subprocess import PIPE, Popen

import requests

OUT_PATH_PARENT = "/Users/joleenf/data/clavrx/navgem/nrl/"

try:
    from bs4 import BeautifulSoup
except ImportError as ie:
    raise ImportError("{}\n Please activate environment: conda activate merra2_clavrx".format(ie))

LOG = logging.getLogger(__name__)

# Product name model table:  OLD INFO: https://www.usgodae.org/docs/layout/pn_model_tbl.pns.html
MODEL_KEYS = {"NOGAPS": "058"}

PRODUCTS_LIST = ["pres_msl", "grnd_sea_temp", "pres", "rltv_hum", "air_temp", "snw_dpth",
                 "prcp_h20", "grnd_wet", "wnd_ucmp", "wnd_vcmp", "geop_ht",
                 "ttl_snow", "air_temp", "height"]


def download_file(url, filename, destination="data/joleenf/navgem/data/"):
    """Download filename from url to destination."""
    LOG.info("Download {}".format(url))
    response = requests.get(url, verify=False, stream=True)
    try:
        response.raise_for_status()
    except Exception as err:
        msg = "Could not retrieve data {}".format(err)
        warnings.warn(msg)
        return None

    data_dir = "{}/{}".format(destination, filename)
    open(data_dir, "wb").write(response.content)

    return data_dir


def download_url(url):
    """Download file from url given."""
    print("downloading: ", url)
    # assumes that the last segment after the / represents the file name
    # if url is abc/xyz/file.txt, the file name will be file.txt
    file_name_start_pos = url.rfind("/") + 1
    file_name = url[file_name_start_pos:]

    r = requests.get(url, stream=True)
    if r.status_code == requests.codes.ok:
        with open(file_name, 'wb') as f:
            for data in r:
                f.write(data)


def create_soup(URL):
    """Create soup object from page request."""
    page = requests.get(URL, verify=False)
    page.raise_for_status()
    LOG.debug(f"Send {URL} to Soup Parser")
    soup = BeautifulSoup(page.content, "html.parser")

    return soup


def url_search_by_filenames(url_soup, url, get_these_files, out_path):
    """Provide a list of filenames and search within given url for download."""
    list_of_files = []
    for a in url_soup.find_all('a', href=True):
        if a['href'] == a.text and a.text in get_these_files:
            url_fn = url + "/" + a.text
            dl_file = download_file(url_fn, a.text, destination=out_path)
            if isinstance(dl_file, str):
                list_of_files.append(dl_file)
    if list_of_files:
        return list_of_files
    else:
        raise RuntimeError(f"No NAVGEM files loaded with {url}")


def url_search_nrl(url_soup, url, navgem_run_dt, forecast_times, out_path=None):
    """Get NRL data using regex."""
    list_of_files = []
    navgem_run = navgem_run_dt.strftime("%Y%m%d%H")
    if forecast_times is None:
        forecast_times = [3, 6, 9, 12]
    for forecast in forecast_times:
        forecast_time = str(forecast).zfill(3)
        # dataset ID table:  https://www.usgodae.org/docs/layout/pn_dataset_tbl.pns.html
        dataset = "F0RL"  # forecast(now time) field of a given product (RL Stands for Realtime
        for product_name in PRODUCTS_LIST:
            pattern = f"US058G[A-Z][A-Z][A-Z]-GR[12]mdl.0018_0056_" \
                      f"{forecast_time}00{dataset}{navgem_run}_[0-9].*-[0-9].*{product_name}"
            for link in url_soup.findAll("a", {"href": re.compile(pattern)}, href=True):
                a = re.match(pattern, link.text)
                if a is not None:
                    url_fn = url + "/" + link.text
                    list_of_files.append(url_fn)
    if list_of_files:

        LOG.debug(len(list_of_files))
        file_list = ' '.join(list_of_files)

        args = shlex.split("curl --output-dir {} --progress-bar "
                           "--insecure --remote-name-all {}".format(out_path, file_list))

        ip = Popen(args, stdin=PIPE, stdout=PIPE)
        LOG.debug(ip.communicate())
        return list_of_files
    else:
        raise RuntimeError(f"No NAVGEM files loaded with {url}")


def search_date(url_soup, url, navgem_run_dt, forecast_times, output_path="."):
    """From soup, get file list matching date and forecast times."""
    navgem_run = navgem_run_dt.strftime("%Y%m%d%H")
    if forecast_times is None:
        forecast_times = [0, 6, 12, 18]
    get_these_files = []
    for forecast in forecast_times:
        forecast = str(forecast).zfill(2)
        # TODO:  Check if this is correct (This is NOMADS format)
        file_name = f'navgem_{navgem_run}f{forecast}.grib2'
        get_these_files.append(file_name)

    LOG.info(get_these_files)
    downloaded_files = url_search_by_filenames(url_soup, url, get_these_files, output_path)

    return downloaded_files


def concat_gribs_in_one(data_path, model_run):
    """Concat all files in data_path, using model_run to name output file."""
    grib_name = os.path.join(data_path, f'navgem_{model_run}.grib')
    data_path_glob = os.path.join(data_path, "US*")
    os.system(f'cat {data_path_glob} >> {grib_name}')
    return grib_name


def argument_parser():
    """Parse command line for navgem_clavrx.py."""
    parse_desc = (
        """\nProcess navgem data previously downloaded from NCEP nomads server.""")

    formatter = ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=parse_desc,
                            formatter_class=formatter)

    parser.add_argument('-s', '--start_date', type=str,
                        default=datetime.now().strftime('%Y%m%d'),
                        help="Desired processing date as YYYYMMDD")
    parser.add_argument('-r', '--run_hour', action='store',
                        type=str, required=False, default='00',
                        help="Two digit model run hour.")
    parser.add_argument('-f', '--forecast_hours', nargs='+',
                        default=[3, 6, 12],
                        help="The forecast hours.")
    parser.add_argument('-d', '--base_path', action='store', nargs='?',
                        type=str, required=False, default=OUT_PATH_PARENT, const=OUT_PATH_PARENT,
                        help="Parent path: year subdirectory appends to this path.")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=2,
                        help='each occurrence increases verbosity 1 level through '
                             'ERROR-WARNING-INFO-DEBUG')

    args = vars(parser.parse_args())
    verbosity = args.pop('verbosity', None)

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(format='%(module)s:%(lineno)d:%(levelname)s:%(message)s',
                        level=levels[min(3, verbosity)])

    year = args['start_date'][:4]
    run_hour = args['run_hour'].zfill(2)

    model_run = f"{args['start_date']}{run_hour}"

    args['base_path'] = os.path.join(args['base_path'], year, model_run)

    return args


if __name__ == '__main__':

    parser_args = argument_parser()

    model_run = f"{parser_args['start_date']}{parser_args['run_hour']}"
    model_run = datetime.strptime(model_run, "%Y%m%d%H")
    input_date = model_run.strftime("%Y%m%d")
    input_year = model_run.strftime("%Y")
    forecast_hour = parser_args['forecast_hours']
    out_path = parser_args['base_path']

    source_site = "nrl"

    if source_site == "ncep":
        URL = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/fnmoc/prod/navgem.{input_date}"
        soup = create_soup(URL)
        search_date(soup, URL, model_run, forecast_hour)
    else:
        model_run_str = model_run.strftime("%Y%m%d%H")
        url = "https://www.usgodae.org/ftp/outgoing/fnmoc/models/navgem_0.5/"
        URL = f"{url}{input_year}/{model_run_str}"
        soup = create_soup(URL)
        out_path = os.path.join(parser_args["base_path"], input_year, model_run_str)
        os.makedirs(out_path, exist_ok=True)
        files = url_search_nrl(soup, URL, model_run, out_path=out_path)
        concat_to_new = concat_gribs_in_one(out_path, model_run_str, files)
        LOG.info(concat_to_new)

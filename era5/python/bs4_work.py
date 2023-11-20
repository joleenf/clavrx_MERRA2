import sys
from datetime import datetime as datetime

import requests
from bs4 import BeautifulSoup


def download_file(url, filename):
    print("Download {}".format(url))
    response = requests.get(url, verify=False)

    data_dir = "/data/joleenf/navgem/data/{}".format(filename)
    open(data_dir, "wb").write(response.content)


def search_date(soup, URL):

    #date_string = "00F0OF"

    list_of_files = []
    file_endings = [ "pres_bdy", "grnd_temp", "pres", "rltv_hum", "air_temp", "snw_dpth",
                     "prcp_h20", "wnd_ucmp", "wnd_vcmp", "geop_ht", "ttl_snow", "air_temp", "height"]
    for a in soup.find_all('a', href=True):
        print(a)
        #if a['href'] == a.text and date_string in a.text:
        if a['href'] == a.text:
            for file_ending in file_endings:
                if file_ending in a.text:
                    url = URL + a.text
                    list_of_files.append(url)
                    download_file(url, a.text)
    with open("list_of_files_fnmoc_navgem.txt", "w+") as f:
        for fn in list_of_files:
            f.write(fn)
            f.write("\n")

if __name__ == '__main__':
     # Enter directory name of navgem run in CCYYMMDDHH format
     model_run = sys.argv[1]

     print(sys.argv)
     #model_run = "2022081406"

     date_dir = datetime.strptime(model_run, "%Y%m%d%H")
     year = date_dir.strftime("%Y")

     URL="https://www.usgodae.org/ftp/outgoing/fnmoc/models/navgem_0.5/{}/{}/".format(year, model_run)
     #URL = "https://www.usgodae.org/ftp/outgoing/fnmoc/models/navgem_0.5/latest_data/"

     page = requests.get(URL, verify=False)

     print("Send Page to Soup Parser")

     soup = BeautifulSoup(page.content, "html.parser")

     search_date(soup, URL)

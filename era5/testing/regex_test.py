import re
#test_string="US058GMET-GR1mdl.0018_0056_00000F0OF2022102306_0006_000000-000000air_temp"
#test_string="US058GMET-GR1mdl.0018_0056_01200F0OF2022102306_0001_000000-000000pres"
#directory="/data/Personal/joleenf/navgem/2022/2022_10_23/"
#forecast_time="012"
#dataset="F0(\D{2})"
#navgem_run="2022102306"
#product_name="pres"
#pattern = r"US.*?{}.*?{}_(\d+)_(\d+)-(\d+){}".format(forecast_time, navgem_run, product_name)
#print(test_string)
#print(pattern)
#a=(re.search(pattern, test_string))
#print('Here it is', a.group(0))

regex_minimal = r"US.*?{}.*?{}_(\d+)_(\d+)-(\d+){}"
tpw_run = "2022101100"
product_name = "cape"
pattern =  regex_minimal.format(r"(\d+)", tpw_run, product_name)

#with open("test_list.txt") as myfile:
#    for line in myfile:
#        a=(re.search(pattern, line))
#        print(a.group(0))

fn="/home/joleenf/clavrx_MERRA2/testing/october.html"

from bs4 import BeautifulSoup
bs4_soup = BeautifulSoup(open(fn), "html.parser")

list_of_files = []
for link in bs4_soup.find_all(href=re.compile(pattern)):
    #url_fn = url + "/" + link.text
    #list_of_files.append(url_fn)
    list_of_files.append(link.text)

print(list_of_files)

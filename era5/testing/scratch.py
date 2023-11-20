import os

OUT_PATH_PARENT = '/apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/navgem/'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_global(variable):
    print("{} ==> {}".format(ROOT_DIR, variable))


if __name__ == "__main__":
    test_global("Hello world from Main!")
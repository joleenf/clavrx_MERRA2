"""Unit conversions for variables used in merra2 and era5 data."""
import numpy as np


def meter_to_km(data):
    """Convert surface height from m to km."""
    return data / 1000.0


def pa_to_hPa(data):
    """Convert from Pa to hPa."""
    return data/100.0


def no_conversion(data):
    """Return the data without conversion."""
    return data


def scale_tpw(data):
    """Return scale mm to cm."""
    return data/10.0


def fill_bad(data):
    """Fill with np.nan."""
    return data*np.nan


def geopotential(data):
    """Convert geopotential in meters per second squared to geopotential height."""
    # this is height/1000.0*g
    return data / 9806.65
